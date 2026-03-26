import os
from pathlib import Path
from copy import deepcopy

import pytest
import torch
from peft import LoraConfig

from twinkle import Platform
from twinkle.model.transformers import TransformersModel

TEST_MODEL_ID = os.environ.get('TEST_MODEL_ID', 'ms://Qwen/Qwen3-0.6B')


def _get_test_device() -> torch.device:
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return torch.device(Platform.get_local_device(platform='npu'))
    if torch.cuda.is_available():
        return torch.device(Platform.get_local_device(platform='gpu'))
    return torch.device('cpu')

def _build_tiny_model() -> TransformersModel:
    model = TransformersModel(
        model_id=TEST_MODEL_ID,
        strategy='accelerate',
        trust_remote_code=True,
    )
    model.model.to(_get_test_device())
    model._save_tokenizer = lambda *args, **kwargs: None
    return model


def _snapshot_named_parameters(model: TransformersModel) -> dict[str, torch.Tensor]:
    unwrapped = model.strategy.unwrap_model(model.model)
    return {name: param.detach().cpu().clone() for name, param in unwrapped.named_parameters()}


def _snapshot_lora_state(model: TransformersModel, adapter_name: str) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.get_state_dict(adapter_name=adapter_name).items()}


def _seed_optimizer_state(model: TransformersModel) -> None:
    model.set_optimizer('AdamW', lr=1e-3)
    model.set_lr_scheduler('StepLR', step_size=1, gamma=0.5)
    optimizer = model.optimizer_group[''].optimizer
    scheduler = model.optimizer_group[''].lr_scheduler

    for param in model.strategy.unwrap_model(model.model).parameters():
        param.grad = torch.ones_like(param)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _assert_tensor_dict_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        expected_value = expected[key]
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            _assert_tensor_dict_equal(actual_value, expected_value)
        elif isinstance(expected_value, (list, tuple)):
            assert actual_value == expected_value
        elif torch.is_tensor(expected_value):
            assert torch.equal(actual_value.detach().cpu(), expected_value.detach().cpu())
        else:
            assert actual_value == expected_value


def test_load_full_checkpoint_restores_weights(tmp_path: Path):
    model = _build_tiny_model()
    before = _snapshot_named_parameters(model)
    checkpoint_dir = Path(model.save('full-ckpt', output_dir=str(tmp_path)))

    assert (checkpoint_dir / 'model.safetensors').exists()

    with torch.no_grad():
        for param in model.strategy.unwrap_model(model.model).parameters():
            param.add_(1.0)

    model.load(str(checkpoint_dir))
    after = _snapshot_named_parameters(model)

    assert after.keys() == before.keys()
    for name in before:
        assert torch.equal(after[name], before[name])


def test_load_lora_checkpoint_still_restores_adapter_weights(tmp_path: Path):
    model = _build_tiny_model()
    model.add_adapter_to_model(
        'resume_lora',
        LoraConfig(task_type='CAUSAL_LM', r=2, lora_alpha=4, target_modules=['q_proj']),
    )
    before = _snapshot_lora_state(model, 'resume_lora')
    checkpoint_dir = Path(model.save('lora-ckpt', output_dir=str(tmp_path), adapter_name='resume_lora'))

    with torch.no_grad():
        for name, param in model.strategy.unwrap_model(model.model).named_parameters():
            if 'resume_lora' in name:
                param.add_(1.0)

    model.load(str(checkpoint_dir), adapter_name='resume_lora')
    after = _snapshot_lora_state(model, 'resume_lora')

    assert after.keys() == before.keys()
    for name in before:
        assert torch.equal(after[name], before[name])


def test_load_full_checkpoint_with_optimizer_restores_optimizer_and_scheduler(tmp_path: Path):
    model = _build_tiny_model()
    _seed_optimizer_state(model)
    optimizer = model.optimizer_group[''].optimizer
    scheduler = model.optimizer_group[''].lr_scheduler
    expected_optimizer_state = deepcopy(optimizer.state_dict())
    expected_scheduler_state = deepcopy(scheduler.state_dict())
    checkpoint_dir = Path(model.save('optim-ckpt', output_dir=str(tmp_path), save_optimizer=True))

    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.add_(2.0)

    scheduler.step()

    model.load(str(checkpoint_dir), load_optimizer=True)

    _assert_tensor_dict_equal(optimizer.state_dict(), expected_optimizer_state)
    _assert_tensor_dict_equal(scheduler.state_dict(), expected_scheduler_state)


def test_load_full_checkpoint_with_optimizer_requires_optimizer_artifact(tmp_path: Path):
    model = _build_tiny_model()
    _seed_optimizer_state(model)
    checkpoint_dir = Path(model.save('missing-optimizer', output_dir=str(tmp_path), save_optimizer=True))

    (checkpoint_dir / 'optimizer.pt').unlink()

    with pytest.raises(FileNotFoundError, match='optimizer.pt'):
        model.load(str(checkpoint_dir), load_optimizer=True)


def test_load_full_checkpoint_with_optimizer_requires_scheduler_artifact(tmp_path: Path):
    model = _build_tiny_model()
    _seed_optimizer_state(model)
    checkpoint_dir = Path(model.save('missing-scheduler', output_dir=str(tmp_path), save_optimizer=True))

    (checkpoint_dir / 'scheduler.pt').unlink()

    with pytest.raises(FileNotFoundError, match='scheduler.pt'):
        model.load(str(checkpoint_dir), load_optimizer=True)
