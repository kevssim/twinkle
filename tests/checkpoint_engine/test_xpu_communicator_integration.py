# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit coverage for the XPU checkpoint engine communicator integration."""
import pytest
import torch

from twinkle.comm import Communicator
from twinkle.checkpoint_engine.nccl_checkpoint_engine import MasterMetadata, _pg_broadcast
from twinkle.checkpoint_engine.xpu_checkpoint_engine import XCCLCheckpointEngine


class _FakeCommunicator(Communicator):
    def __init__(self):
        super().__init__(0, 2, 'cpu')
        self.barrier_calls = 0
        self.destroy_calls = 0
        self.broadcast_calls = []

    @classmethod
    def _create(cls, rendezvous, rank, world_size, device, *, timeout, **options):
        return cls()

    def barrier(self):
        self.barrier_calls += 1

    def _broadcast(self, tensor, src, async_op):
        self.broadcast_calls.append((tensor, src))
        return None

    def _destroy(self):
        self.destroy_calls += 1


def _metadata():
    return MasterMetadata(zmq_ip="127.0.0.1", zmq_port=12345,
                          nccl_store_host="127.0.0.1", nccl_store_port=23456)


def test_xpu_engine_uses_communicator_and_reuses_group(monkeypatch):
    fake = _FakeCommunicator()
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return fake

    monkeypatch.setattr("twinkle.checkpoint_engine.xpu_checkpoint_engine.create_communicator", create)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    engine = XCCLCheckpointEngine(rebuild_group=False)
    engine.init_process_group(0, 2, _metadata())
    engine.init_process_group(0, 2, _metadata())

    assert len(calls) == 1
    assert calls[0][0][1:4] == (0, 2, torch.device("cuda", 3))
    assert calls[0][1]["backend"] == "xccl"
    assert engine._pg is fake
    assert fake.barrier_calls == 1

    engine.rebuild_group = True
    engine.finalize()
    assert fake.destroy_calls == 1
    assert engine._comm is None
    assert engine._pg is None
    assert engine._group_initialized is False


def test_xpu_engine_skips_communicator_for_non_participating_rank(monkeypatch):
    def fail_create(*args, **kwargs):
        pytest.fail("non-participating trainer rank must not create a communicator")

    monkeypatch.setattr("twinkle.checkpoint_engine.xpu_checkpoint_engine.create_communicator", fail_create)
    engine = XCCLCheckpointEngine()
    engine.init_process_group(-1, 3, _metadata())
    assert engine.rank == -1
    assert engine.world_size == 3
    assert engine._group_initialized is True
    assert engine._comm is None


def test_xpu_engine_connects_zmq_only_for_receiver(monkeypatch):
    fake = _FakeCommunicator()
    connected = []
    monkeypatch.setattr("twinkle.checkpoint_engine.xpu_checkpoint_engine.create_communicator", lambda *args, **kwargs: fake)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(XCCLCheckpointEngine, "_connect_zmq_client", lambda self, metadata: connected.append(metadata))

    engine = XCCLCheckpointEngine()
    engine.init_process_group(1, 2, _metadata())
    assert connected == [_metadata()]
    assert fake.barrier_calls == 1


def test_pg_broadcast_accepts_communicator():
    fake = _FakeCommunicator()
    tensor = torch.zeros(4)
    _pg_broadcast(fake, tensor, src=1)
    assert fake.broadcast_calls == [(tensor, 1)]


def test_pg_broadcast_keeps_raw_process_group_api():
    tensor = torch.zeros(2)
    calls = []

    class _Work:
        def wait(self):
            calls.append(("wait",))

    class _RawGroup:
        def broadcast(self, tensors, options):
            calls.append(("broadcast", tensors, options.rootRank))
            return _Work()

    _pg_broadcast(_RawGroup(), tensor, src=1)
    assert calls[0] == ("broadcast", [tensor], 1)
    assert calls[1] == ("wait",)
