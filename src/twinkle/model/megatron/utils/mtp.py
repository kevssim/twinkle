# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-model state for multi-token-prediction (MTP) joint training on the megatron backend.

:class:`MtpTrainingState` owns everything MTP that ``MegatronModel`` needs across a run: whether a
pass should feed the MTP heads, a one-shot trainability check, the microbatch count that
normalizes the drained loss, and the drain itself. The model keeps thin same-named methods
over it, so subclass overrides and duck-typed callers (``getattr(model, 'pop_mtp_metrics')``
in swift) keep resolving.
"""
from __future__ import annotations

from typing import Optional


class MtpTrainingState:
    """The MTP training state of one model instance.

    Two pieces of state make this a class rather than a set of free functions: the
    trainability warning must fire once per model (not per pass, not per process), and the
    loss normalizer must be the microbatch count the model's own ``forward_backward``
    chose -- a caller counting the batches it handed over would divide by the wrong
    number.
    """

    def __init__(self):
        # One-shot guard so the "MTP has no trainable parameter" warning fires once, not per step.
        self._trainable_checked = False
        # Microbatch count of the last forward_backward, so pop() can normalize a tracker
        # that accumulates per microbatch. Starts at 1 so a pop before any step does not
        # divide by 0.
        self._last_num_microbatches = 1

    # ── stateless checks ────────────────────────────────────────────────────

    @staticmethod
    def feed_enabled(enable_intent: bool, *, forward_only: bool, task: str, disable_lora: bool) -> bool:
        """Whether this pass should hand next-token targets to the MTP heads.

        The config declares the *intent* (``enable_mtp_training``); only the call site knows
        whether a pass will be backward'ed. That split is necessary, not redundant: megatron's
        ``forward_only`` schedule skips the backward but neither switches the module to eval
        nor disables autograd, so ``self.training`` and ``torch.is_grad_enabled()`` are both
        True on a log-prob pass. Running the MTP block there would cost an extra transformer
        layer plus a full-vocabulary projection per unroll step, and throw every bit of it
        away.
        """
        if not enable_intent:
            return False
        if forward_only:
            # old/reference log-probs. No backward, so the MTP forward would be pure waste.
            return False
        if task != 'causal_lm':
            # A pooling head has no next-token target to give them.
            return False
        if disable_lora:
            # The base-policy pass of a LoRA run; its gradients are discarded by construction.
            return False
        return True

    @staticmethod
    def iter_modules(unwrapped_models):
        """The MTP blocks across model chunks, skipping ranks and chunks that hold none.

        ``mtp`` sits on the language model, which for a multimodal model is one level down. PP
        ranks other than the last have no MTP block at all, and a PeftModel forwards the
        lookup to its base.
        """
        for _model in unwrapped_models:
            mtp = getattr(_model, 'mtp', None)
            if mtp is None:
                language_model = getattr(_model, 'language_model', None)
                mtp = getattr(language_model, 'mtp', None) if language_model is not None else None
            if mtp is not None:
                yield mtp

    @staticmethod
    def warn_if_not_trainable(mtp_modules) -> None:
        """Warn when MTP training is asked for but nothing can receive the gradient.

        LoRA is the way to get here that config validation cannot catch: it freezes every
        base parameter, and the MTP layers are base parameters unless named in
        ``target_modules``. The result is a run that pays for the MTP forward and reports an
        mtp_loss that never moves, which is far harder to notice than an error.
        """
        from twinkle.utils import get_logger

        if not mtp_modules:
            # Expected on every PP rank but the last; those ranks have no MTP block to check.
            return
        if any(p.requires_grad for mtp in mtp_modules for p in mtp.parameters()):
            return
        get_logger().warning('enable_mtp_training is set, but no MTP parameter has requires_grad=True, so the '
                             'MTP loss will be computed and then discarded. With LoRA this means the MTP '
                             'layers are not covered by target_modules.')

    # ── stateful, driven by the model ───────────────────────────────────────

    def training_enabled(self, model, *, forward_only: bool, task: str, disable_lora: bool) -> bool:
        """Gate this pass and, on the first gated-in pass, check the MTP blocks' trainability.

        *model* is the ``MegatronModel`` this instance belongs to: the config intent, the
        unwrapped chunks and the strategy all hang off it.
        """
        enable_intent = getattr(model.strategy.config, 'enable_mtp_training', False)
        if not self.feed_enabled(enable_intent, forward_only=forward_only, task=task, disable_lora=disable_lora):
            return False
        if not self._trainable_checked:
            self._trainable_checked = True
            self.warn_if_not_trainable(list(self.iter_modules(model.strategy.unwrap_model(model.model))))
        return True

    def note_num_microbatches(self, num_microbatches: int) -> None:
        """Record the microbatch count of the forward_backward currently running."""
        self._last_num_microbatches = num_microbatches

    def pop(self, loss_scale: Optional[float] = None) -> dict[str, float]:
        """Drain megatron's MTP loss tracker and return one entry per MTP depth, plus a total.

        Draining rather than reading: the tracker accumulates across microbatches and megatron
        never clears it, so a caller that only read it would report a monotonically growing
        loss.

        Args:
            loss_scale: Multiplier for the accumulated values; defaults to
                ``1 / last_num_microbatches``.

        Returns:
            ``{'mtp_1_loss': ..., ..., 'mtp_loss': <sum>}``, plus ``mtp_{i}_accept`` per depth
            when the megatron version reports acceptance counts. Empty when MTP is off or
            nothing ran.
        """
        from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

        tracker = MTPLossLoggingHelper.tracker
        # mcore >= 0.19 renamed 'values' to 'loss_values' and added acceptance counts.
        loss_key = 'loss_values' if 'loss_values' in tracker else 'values'
        if loss_key not in tracker:
            return {}
        if loss_scale is None:
            loss_scale = 1.0 / max(self._last_num_microbatches, 1)

        if hasattr(MTPLossLoggingHelper, 'reduce_metrics_in_tracker'):
            MTPLossLoggingHelper.reduce_metrics_in_tracker()
        else:
            self._reduce_tracker_legacy(tracker, loss_key)

        losses = (tracker[loss_key] * loss_scale).tolist()
        metrics = {f'mtp_{i + 1}_loss': float(v) for i, v in enumerate(losses)}
        metrics['mtp_loss'] = float(sum(losses))

        correct, total = tracker.get('correct_values'), tracker.get('total_values')
        if correct is not None and total is not None:
            for i, (c, t) in enumerate(zip(correct.tolist(), total.tolist(), strict=True)):
                # The acceptance rate is what decides whether speculative decoding pays off at
                # rollout time, so it is worth surfacing next to the loss rather than only the loss.
                metrics[f'mtp_{i + 1}_accept'] = float(c / t) if t else 0.0

        if hasattr(MTPLossLoggingHelper, 'clean_metrics_in_tracker'):
            MTPLossLoggingHelper.clean_metrics_in_tracker()
        else:
            MTPLossLoggingHelper.clean_loss_in_tracker()
        return metrics

    @staticmethod
    def _reduce_tracker_legacy(tracker, loss_key: str) -> None:
        """The pre-0.19 reduction, which had no ``reduce_metrics_in_tracker`` to call.

        Order matters: the sum across the reduce group first, then the average across the
        group the loss is replicated over. Swapping them averages partial sums.
        """
        import torch.distributed as dist

        values = tracker[loss_key]
        if tracker.get('reduce_group') is not None:
            dist.all_reduce(values, group=tracker['reduce_group'])
        if tracker.get('avg_group') is not None:
            dist.all_reduce(values, group=tracker['avg_group'], op=dist.ReduceOp.AVG)
