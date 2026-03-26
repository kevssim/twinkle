# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
from copy import deepcopy
from typing import Dict, Iterable

import torch


def load_full_state_dict(model, checkpoint_dir) -> None:
    checkpoint_files = _resolve_checkpoint_files(checkpoint_dir)
    state_dict = _load_checkpoint_state_dict(checkpoint_files)
    state_dict = _apply_hf_weight_conversion(model, state_dict)
    model.load_state_dict(state_dict, strict=True)


def _resolve_checkpoint_files(checkpoint_dir) -> list[str]:
    checkpoint_dir = os.fspath(checkpoint_dir)
    safe_index = os.path.join(checkpoint_dir, 'model.safetensors.index.json')
    bin_index = os.path.join(checkpoint_dir, 'pytorch_model.bin.index.json')
    safe_file = os.path.join(checkpoint_dir, 'model.safetensors')
    bin_file = os.path.join(checkpoint_dir, 'pytorch_model.bin')

    if os.path.exists(safe_index):
        return _checkpoint_files_from_index(checkpoint_dir, safe_index)
    if os.path.exists(bin_index):
        return _checkpoint_files_from_index(checkpoint_dir, bin_index)
    if os.path.exists(safe_file):
        return [safe_file]
    if os.path.exists(bin_file):
        return [bin_file]
    raise FileNotFoundError(f'No full model checkpoint found under {checkpoint_dir}')


def _checkpoint_files_from_index(checkpoint_dir: str, index_file: str) -> list[str]:
    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)
    shard_files = sorted(set(index['weight_map'].values()))
    return [os.path.join(checkpoint_dir, shard_file) for shard_file in shard_files]


def _load_checkpoint_state_dict(checkpoint_files: Iterable[str]) -> Dict[str, torch.Tensor]:
    from transformers.modeling_utils import load_state_dict

    merged_state_dict = {}
    for checkpoint_file in checkpoint_files:
        merged_state_dict.update(load_state_dict(checkpoint_file))
    return merged_state_dict


def _apply_hf_weight_conversion(model, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    try:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
    except ImportError:
        return state_dict

    weight_mapping = getattr(model, '_weight_conversions', None)
    if weight_mapping is None:
        weight_mapping = get_model_conversion_mapping(model)
    if not weight_mapping:
        return state_dict

    renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
    pattern_to_converter = {pattern: converter for converter in converters for pattern in converter.source_patterns}

    converted_state_dict = {}
    meta_state_dict = model.state_dict()
    prefix = getattr(model, 'base_model_prefix', None)
    pending_mappings = {}
    unexpected_keys = set()

    for original_key, tensor in sorted(state_dict.items()):
        renamed_key, source_pattern = rename_source_key(original_key, renamings, converters, prefix, meta_state_dict)
        if renamed_key not in meta_state_dict and original_key in meta_state_dict:
            renamed_key, source_pattern = rename_source_key(original_key, [], [], prefix, meta_state_dict)

        if renamed_key in meta_state_dict:
            if source_pattern is not None:
                new_converter = deepcopy(pattern_to_converter[source_pattern])
                mapping = pending_mappings.setdefault(renamed_key, new_converter)
            else:
                mapping = pending_mappings.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                source_pattern = original_key
            mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)
        elif source_pattern is not None:
            mapping = pattern_to_converter[source_pattern]
            for target_pattern in mapping.target_patterns:
                unexpected_keys.add(renamed_key.replace(mapping.target_patterns[0], target_pattern))
        else:
            unexpected_keys.add(renamed_key)

    for first_param_name, mapping in pending_mappings.items():
        realized_value = mapping.convert(first_param_name, model=model, config=model.config)
        for target_name, param in realized_value.items():
            converted_state_dict[target_name] = param[0] if isinstance(param, list) else param

    if unexpected_keys:
        raise RuntimeError(_format_incompatible_keys(model, unexpected_keys=unexpected_keys))

    return converted_state_dict


def _format_incompatible_keys(model, missing_keys: Iterable[str] = (), unexpected_keys: Iterable[str] = ()) -> str:
    missing_keys = tuple(sorted(missing_keys))
    unexpected_keys = tuple(sorted(unexpected_keys))
    error_message = f'Error(s) in loading state_dict for {model.__class__.__name__}:'
    if missing_keys:
        error_message += '\n\tMissing key(s) in state_dict: ' + ', '.join(f'"{key}"' for key in missing_keys) + '.'
    if unexpected_keys:
        error_message += '\n\tUnexpected key(s) in state_dict: ' + ', '.join(f'"{key}"' for key in unexpected_keys) + '.'
    return error_message
