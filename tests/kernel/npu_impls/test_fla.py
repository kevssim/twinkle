def test_fla_imports():
    from twinkle.kernel.npu_impls.fla import apply_qwen3_5_fla
    assert callable(apply_qwen3_5_fla)


def test_fla_disabled_by_env(monkeypatch):
    monkeypatch.setenv('TWINKLE_NPU_FLA', '0')
    from twinkle.kernel.npu_impls.fla import apply_qwen3_5_fla
    # With env=0, function returns 0 (no-op) without raising
    assert apply_qwen3_5_fla(None) == 0


def test_fla_skips_when_no_torch_npu(monkeypatch):
    import sys
    monkeypatch.setenv('TWINKLE_NPU_FLA', '1')
    monkeypatch.setitem(sys.modules, 'torch_npu', None)  # forces ImportError on import
    from twinkle.kernel.npu_impls import fla as fla_mod
    # Reload-tolerant: should return 0 when torch_npu is missing.
    assert fla_mod.apply_qwen3_5_fla(None) == 0