# Copyright (c) ModelScope Contributors. All rights reserved.
"""Non-device tests for XCCL registration and contract."""
import importlib
import pytest
from twinkle.comm import get_communicator_cls


def test_xccl_backend_is_lazily_registered():
    cls = get_communicator_cls("xccl")
    assert cls.__name__ == "XCCLCommunicator"
    assert cls.backend_name == "xccl"
    assert cls.supports("broadcast") is True
    assert cls.supports("all_reduce") is False
    assert cls.supports("barrier") is True


def test_xccl_backend_module_imports_without_constructing_runtime():
    module = importlib.import_module("twinkle.comm.backends.xccl")
    assert hasattr(module, "XCCLCommunicator")
    assert module.XCCLCommunicator.supports("broadcast")


def test_xccl_backend_rejects_unknown_options_before_runtime():
    cls = get_communicator_cls("xccl")
    with pytest.raises(ValueError, match="does not accept options"):
        cls.create(None, 0, 1, "cpu", unsupported=True)
