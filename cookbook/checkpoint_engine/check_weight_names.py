# Copyright (c) ModelScope Contributors. All rights reserved.
"""打印 Qwen3.5 的权重名映射，用于定位 full 同步的 tied embedding 报错。

不加载权重（meta device 建骨架），不启动 Ray，只看名字：
  python -m cookbook.checkpoint_engine.check_weight_names --model-path /nas/disk1/Qwen3.5-4B
"""
import argparse

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from twinkle.checkpoint_engine.delta.backends.fsdp import convert_weight_keys

KEYWORDS = ('embed_tokens', 'lm_head')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    architectures = getattr(config, 'architectures', None)
    print(f'architectures        : {architectures}')
    text_config = getattr(config, 'text_config', config)
    print(f'tie_word_embeddings  : {getattr(config, "tie_word_embeddings", None)} '
          f'(text_config: {getattr(text_config, "tie_word_embeddings", None)})')

    # 与 TransformersModel 一致：优先用 config.architectures[0]，而不是 AutoModelForCausalLM
    # （两者命名空间不同：ForConditionalGeneration 带 language_model. 前缀）。
    model_cls = AutoModelForCausalLM
    if architectures:
        model_cls = getattr(transformers, architectures[0], AutoModelForCausalLM)
    print(f'model_cls (同 trainer): {getattr(model_cls, "__name__", model_cls)}')

    with torch.device('meta'):
        if hasattr(model_cls, 'from_config'):
            model = model_cls.from_config(config, trust_remote_code=True)
        else:
            model = model_cls._from_config(config)
    print(f'model class          : {type(model).__name__}')
    mapping = getattr(model, '_checkpoint_conversion_mapping', None)
    print(f'_checkpoint_conversion_mapping: {mapping}')

    state_dict = model.state_dict()
    print(f'\nstate_dict 里的相关键（trainer 侧原始名，共 {len(state_dict)} 个权重）：')
    for name in state_dict:
        if any(k in name for k in KEYWORDS):
            print(f'  {name}')

    converted = convert_weight_keys(state_dict, model)
    print(f'\nconvert_weight_keys 之后（实际发给 vLLM 的名字）：')
    for name in converted:
        if any(k in name for k in KEYWORDS):
            print(f'  {name}')

    print('\nvLLM 报错期望的是 language_model.model.embed_tokens.weight；'
          '若上面没有这个名字，就是导出侧少了 language_model. 前缀。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
