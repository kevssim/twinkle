import torch
from peft import LoraConfig, get_peft_model
from torch import nn


class FakePackedExperts(nn.Module):

    def __init__(self, num_experts=2, hidden=4, intermediate=6, *, is_transposed=False):
        super().__init__()
        self.is_transposed = is_transposed
        if is_transposed:
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, intermediate * 2, hidden))
            self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))
        else:
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate * 2))
            self.down_proj = nn.Parameter(torch.randn(num_experts, intermediate, hidden))

    def forward(self, x, expert_idx=0):
        gate_up = self.gate_up_proj[expert_idx]
        down = self.down_proj[expert_idx]
        if self.is_transposed:
            hidden = torch.nn.functional.linear(x, gate_up)
            gate, up = hidden.chunk(2, dim=-1)
            return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down)

        hidden = torch.nn.functional.linear(x, gate_up.T)
        gate, up = hidden.chunk(2, dim=-1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down.T)


class FakeModel(nn.Module):

    def __init__(self, *, is_transposed=False):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = FakePackedExperts(is_transposed=is_transposed)

    def forward(self, x, expert_idx=0):
        return self.mlp.experts(x, expert_idx=expert_idx)


def test_peft_target_parameter_key_shapes_for_3d_experts():
    model = FakeModel()
    cfg = LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=[],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )

    peft_model = get_peft_model(model, cfg, adapter_name="default")
    state = peft_model.state_dict()
    lora_shapes = {key: tuple(state[key].shape) for key in state if "lora_" in key}

    assert lora_shapes == {
        "base_model.model.mlp.experts.base_layer.lora_A.default.weight": (4, 4),
        "base_model.model.mlp.experts.base_layer.lora_B.default.weight": (12, 4),
        "base_model.model.mlp.experts.lora_A.default.weight": (4, 6),
        "base_model.model.mlp.experts.lora_B.default.weight": (4, 4),
    }
