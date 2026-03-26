# TransformersModel Full-Parameter Checkpoint Load Design

Date: 2026-03-26
Status: Approved for implementation planning

## Summary

`TransformersModel.save()` already supports saving full-parameter checkpoints, but `TransformersModel.load()` only supports LoRA checkpoints and raises `NotImplementedError` for non-PEFT models. This design fills that gap by adding full-parameter checkpoint loading to the existing `load()` API without changing the public interface.

The goal is a minimal, explicit checkpoint-resume capability for full-parameter training:

- Restore full model weights from checkpoints produced by `Twinkle.save()`
- Optionally restore optimizer and scheduler state when `load_optimizer=True`
- Preserve the current LoRA behavior and API shape
- Keep the design extensible for future `EP + FSDP` checkpoint resume work

This design does not attempt full training-scene restore. It does not persist or recover `cur_step`, `GradScaler`, RNG state, or dataloader position.

## Goals

- Support loading full-parameter checkpoints saved by `TransformersModel.save()`
- Reuse the existing `load(name, output_dir=None, load_optimizer=False, ...)` API
- Keep load path resolution behavior unchanged for local path, `output_dir + name`, and hub path cases
- Restore optimizer and scheduler state when explicitly requested
- Fail fast when the caller requests optimizer restore but the checkpoint is incomplete
- Introduce the loading logic in a place that can later absorb `EP + FSDP` special handling cleanly

## Non-Goals

- Supporting arbitrary HuggingFace `save_pretrained()` directories as a contractual feature
- Full training-state resume
- Changing the LoRA checkpoint format or LoRA load semantics in this task
- Adding new user-facing checkpoint APIs

## Current Problem

Today the save and load paths are asymmetric:

- Full-parameter save already exists in `TransformersModel.save()`
- LoRA save and load both exist
- Full-parameter load does not exist

This means users can save a full-parameter checkpoint but cannot reload it through the same model instance and continue training with the current API.

## Proposed Approach

Use a strategy-level full-state loading API.

### Why this approach

The strategy layer already owns full-state export through `get_full_state_dict()`:

- `AccelerateStrategy.get_full_state_dict()`
- `NativeFSDPStrategy.get_full_state_dict()`

Adding the matching load path there keeps save and load responsibilities symmetrical and leaves room for `EP + FSDP` specific logic later. If the implementation is placed directly inside `TransformersModel.load()`, future strategy-specific branching would accumulate there and make the model wrapper harder to maintain.

## Architecture

### Public API

Keep the public entry point unchanged:

`TransformersModel.load(name, output_dir=None, load_optimizer=False, **kwargs)`

No new arguments are introduced.

### Internal API

Add a new internal method to strategy implementations:

`load_full_state_dict(model, checkpoint_dir)`

Implement it in:

- `AccelerateStrategy`
- `NativeFSDPStrategy`

### Call Flow

`TransformersModel.load()` continues to:

1. Resolve `checkpoint_dir`
2. Unwrap the model
3. Branch by model type

Branch behavior:

- If the unwrapped model is a `PeftModel`, keep the existing LoRA load path
- Otherwise, call `self.strategy.load_full_state_dict(self.model, checkpoint_dir)`

After weights are loaded:

- If `load_optimizer=False`, return successfully
- If `load_optimizer=True`, restore optimizer and scheduler state through `_load_optimizer(...)`

## Checkpoint Semantics

This task only guarantees support for full-parameter checkpoints produced by `Twinkle.save()`.

Expected checkpoint contents for the model portion are whatever `save_pretrained(...)` emitted during `Twinkle.save()`, such as:

- `config.json`
- `model.safetensors` or sharded `model-xxxxx-of-xxxxx.safetensors`
- tokenizer and processor files as already emitted today

Expected optimizer files when `save_optimizer=True` was used:

- `optimizer.pt`
- `scheduler.pt`

The intended behavior is:

- `load_optimizer=False`: restore model weights only
- `load_optimizer=True`: restore model weights and require both optimizer files to exist

## Model Weight Loading

The strategy-level loader should restore weights into the unwrapped model using HuggingFace sharded checkpoint loading utilities. This keeps the implementation compatible with both single-file and sharded checkpoints emitted by `save_pretrained(...)`.

Initial implementation target:

- Use HuggingFace checkpoint loading on the unwrapped model
- Keep strict loading enabled so checkpoint/model mismatches fail clearly

This design intentionally leaves room for `NativeFSDPStrategy.load_full_state_dict(...)` to grow extra logic later if `EP + FSDP` requires parameter redistribution or special handling for expert tensors.

## Optimizer and Scheduler Restore

Optimizer restore remains managed by `TransformersModel._load_optimizer(...)`, but its contract is tightened.

New required behavior:

- If `load_optimizer=True`, both `optimizer.pt` and `scheduler.pt` must exist
- If either file is missing, raise an explicit error
- Do not silently skip missing optimizer artifacts in this mode

This makes checkpoint-resume intent explicit. If a caller asks to resume optimizer state, a partially saved checkpoint should fail immediately rather than appearing to resume while actually restarting optimizer state from scratch.

When `load_optimizer=False`, optimizer artifacts are ignored entirely.

## Error Handling

The implementation should fail loudly and specifically in these cases:

- The checkpoint directory cannot be resolved
- The checkpoint does not contain a compatible full-parameter model state
- The user passes `load_optimizer=True` but `optimizer.pt` is missing
- The user passes `load_optimizer=True` but `scheduler.pt` is missing
- Optimizer or scheduler state cannot be loaded into the already-created optimizer objects

Recommended error style:

- Use targeted exceptions with checkpoint path and missing filename in the message
- Prefer explicit `FileNotFoundError` for missing optimizer artifacts
- Let model state loading mismatches surface as load errors instead of downgrading them to warnings

## Compatibility Notes

### Supported after this change

- Full-parameter checkpoint save -> load within `TransformersModel`
- Optional optimizer/scheduler restore for full-parameter checkpoints
- Existing LoRA load path unchanged

### Intentionally unsupported in this task

- Full fidelity resume of training counters and runtime state
- Arbitrary external HuggingFace checkpoint directories as a promised compatibility target
- Automatic recovery of optimizer state when the caller forgot to save it

## EP + FSDP Readiness

Future `EP + FSDP` checkpoint resume is the main reason this design places full-state loading in the strategy layer.

Expected follow-on benefit:

- `TransformersModel.load()` remains stable
- Dense and standard FSDP can share the same external API
- `NativeFSDPStrategy` can later specialize the load path for expert-parallel parameter handling without pushing that complexity into the model wrapper

This task therefore acts as the compatibility foundation for later `EP + FSDP` resume work even though it does not implement the full `EP + FSDP` behavior yet.

## Testing Plan

Add tests that cover the minimal supported contract:

1. Full-parameter save then load restores model weights successfully
2. Full-parameter save with optimizer artifacts then load with `load_optimizer=True` restores optimizer and scheduler state
3. Full-parameter load with `load_optimizer=True` and missing `optimizer.pt` raises an error
4. Full-parameter load with `load_optimizer=True` and missing `scheduler.pt` raises an error
5. Existing LoRA load flow still passes its regression coverage

Testing emphasis:

- Verify behavior through the public `save()` and `load()` methods
- Use checkpoints produced by `Twinkle.save()` rather than synthetic foreign checkpoint directories
- Keep at least one regression test around LoRA so this task does not unintentionally perturb the PEFT path

## Implementation Outline

1. Add `load_full_state_dict(model, checkpoint_dir)` to `AccelerateStrategy`
2. Add `load_full_state_dict(model, checkpoint_dir)` to `NativeFSDPStrategy`
3. Update `TransformersModel.load()` to dispatch full-parameter loads through the strategy
4. Tighten `_load_optimizer(...)` so explicit optimizer restore requires both files
5. Add test coverage for successful full-parameter load and missing optimizer artifact failures

## Open Follow-Up Work

Potential later extensions, out of scope for this task:

- Persist and restore `cur_step`
- Persist and restore `GradScaler`
- Persist and restore RNG state
- Persist and restore dataloader progress
- Extend `NativeFSDPStrategy.load_full_state_dict(...)` for `EP + FSDP` specific restore requirements
