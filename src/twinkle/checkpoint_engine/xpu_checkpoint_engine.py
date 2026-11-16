# Copyright (c) ModelScope Contributors. All rights reserved.
"""Checkpoint weight-sync engine for Kunlunxin XPU (BKCL mounted as torch 'nccl').

Identical to :class:`NCCLCheckpointEngine` except for how the weight-sync group is
built. Constructing a ``ProcessGroupNCCL`` directly deadlocks on XPU when two ranks
share a local device index (every single-card Ray worker reports local index 0), so
the group is built through the backend-neutral ``twinkle.comm`` 'xccl' backend
instead: a stateless ``ProcessGroupXCCL`` with a socket-relay fallback, plus a
store-based readiness barrier that needs no device collective during init. Bucketing,
ZMQ metadata and double buffering are inherited unchanged.
"""
from datetime import timedelta
from typing import Optional

import torch

from twinkle import get_logger
from twinkle.comm import Communicator, Rendezvous, create_communicator
from .nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine

logger = get_logger()

# Upper bound for building the group and for each relay socket transfer.
_GROUP_TIMEOUT = timedelta(seconds=600)


class XCCLCheckpointEngine(NCCLCheckpointEngine):
    """NCCL-checkpoint-engine drop-in for Kunlunxin XPU; group built via ``twinkle.comm``.

    The direct ``ProcessGroupNCCL`` construction (which deadlocks when two ranks share a
    local device index) is replaced by the 'xccl' communicator backend. Everything else
    is inherited unchanged.
    """

    # Distinct from the inherited 'nccl': the mixin keys persistent engine channels by
    # this name, and 'xccl' selects this class on XPU.
    backend_name = 'xccl'

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._comm: Optional[Communicator] = None

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """Build the weight-sync group through the 'xccl' communicator backend.

        Same rendezvous as the NCCL engine (a dedicated TCPStore hosted by the master,
        now owned by the communicator), but the XCCL/relay decision and the readiness
        barrier live in ``twinkle.comm``.
        """
        # Non-participating trainer ranks: record rank and return.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            self._group_initialized = True
            return

        # Fast path: group already initialized, skip all setup.
        if self._group_initialized and not self.rebuild_group:
            return

        if self._comm is None:
            self.rank = rank
            self.world_size = world_size

            device = torch.device('cuda', torch.cuda.current_device())
            self._comm = create_communicator(
                Rendezvous(master_metadata.nccl_store_host, master_metadata.nccl_store_port,
                           group_name=self.group_name),
                rank,
                world_size,
                device,
                backend='xccl',
                timeout=_GROUP_TIMEOUT,
            )
            # BroadcastOperation / _pg_broadcast operate on this; a Communicator is
            # accepted there alongside a raw ProcessGroup.
            self._pg = self._comm
        else:
            assert self.rank == rank, f'rank {rank} != self.rank {self.rank}'
            assert self.world_size == world_size, (
                f'world_size {world_size} != self.world_size {self.world_size}')

        # Receivers connect to master's ZMQ PUB server.
        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)

        # Store-based readiness barrier (no device collective during init).
        self._comm.barrier()

        self._group_initialized = True
        logger.info(f'init_process_group: rank={self.rank}, world_size={self.world_size}')

    def finalize(self):
        """Destroy the communicator on teardown; otherwise keep it for the next sync."""
        if self.rebuild_group and self._comm is not None:
            self._comm.destroy()
            self._comm = None
        super().finalize()
