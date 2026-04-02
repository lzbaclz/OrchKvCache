"""
Block-level hotness scoring for vLLM victim selection.

vLLM 0.7.x requires ALL blocks of a RUNNING sequence to be on GPU
(PagedAttention constraint).  True intra-request partial swap—hot
blocks on GPU, cold blocks on CPU while the sequence runs—requires
kernel changes that are outside the scope of this patch.

What this module DOES provide:
  1. BlockHotnessTracker — per-block EMA scoring
  2. Aggregate request scoring — sum of block scores per sequence,
     giving a finer-grained victim selection signal than progress ratio
  3. Scheduler patch that uses block-level aggregate scores to pick
     the optimal victim when preemption is triggered

Usage:
    from orchkv.vllm_integration.block_level_swap import (
        apply_scheduler_patch,
    )
    apply_scheduler_patch(scheduler_py_path)
    # then launch vLLM with ORCHKV_SWAP=1  ORCHKV_BLOCK_SCORE=1
"""
from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)

try:
    import orchkv_core as _C
except ImportError:
    _C = None

if TYPE_CHECKING:
    from vllm.core.block.interfaces import Block
    from vllm.core.block_manager import SelfAttnBlockSpaceManager
    from vllm.core.scheduler import Scheduler
    from vllm.sequence import SequenceGroup

from vllm.sequence import SequenceStatus
from vllm.utils import Device


# ---------------------------------------------------------------------------
# Block hotness tracker
# ---------------------------------------------------------------------------

class BlockHotnessTracker:
    """EMA-based hotness scoring for individual KV-cache blocks."""

    def __init__(
        self,
        tm_handle: Optional[int] = None,
        ema_lambda: float = 0.3,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
    ):
        self._tm = tm_handle
        self._lambda = ema_lambda
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        self._ema: Dict[int, float] = {}
        self._last_access: Dict[int, int] = {}
        self._access_count: Dict[int, int] = {}
        self._step = 0
        self._max_ema = 1e-9
        self._stats = {"reports": 0, "steps": 0}

    def report_attention(self, block_id: int, score: float) -> None:
        old = self._ema.get(block_id, 0.0)
        self._ema[block_id] = self._lambda * score + (1 - self._lambda) * old
        self._last_access[block_id] = self._step
        self._access_count[block_id] = self._access_count.get(block_id, 0) + 1
        self._stats["reports"] += 1
        if self._tm is not None:
            _C.tm_report_attn(self._tm, block_id, score)

    def step_done(self) -> None:
        self._step += 1
        decay = 1.0 - self._lambda
        new_max = 1e-9
        for bid in self._ema:
            self._ema[bid] *= decay
            if self._ema[bid] > new_max:
                new_max = self._ema[bid]
        self._max_ema = new_max
        self._stats["steps"] += 1
        if self._tm is not None:
            _C.tm_step_done(self._tm)

    def get_score(self, block_id: int) -> float:
        ema = self._ema.get(block_id, 0.0)
        norm_ema = ema / self._max_ema if self._max_ema > 1e-9 else 0.0
        dt = self._step - self._last_access.get(block_id, 0)
        recency = 2.0 ** (-dt / 10.0)
        freq = self._access_count.get(block_id, 0)
        max_freq = max(self._access_count.values()) if self._access_count else 1
        norm_freq = min(freq / max(max_freq, 1), 1.0)
        return (self._alpha * norm_ema
                + self._beta * recency
                + self._gamma * norm_freq)

    def remove_block(self, block_id: int) -> None:
        self._ema.pop(block_id, None)
        self._last_access.pop(block_id, None)
        self._access_count.pop(block_id, None)

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "tracked_blocks": len(self._ema),
                "current_step": self._step}


# ---------------------------------------------------------------------------
# Aggregate request scorer — uses block-level scores for victim selection
# ---------------------------------------------------------------------------

class RequestScorer:
    """Computes per-request aggregate scores from block-level hotness.

    A request's value = sum of hotness scores of all its GPU blocks.
    The request with the LOWEST value is the best preemption candidate.
    """

    def __init__(
        self,
        block_manager: "SelfAttnBlockSpaceManager",
        tracker: BlockHotnessTracker,
    ):
        self._bm = block_manager
        self._tracker = tracker
        self._stats = {
            "score_calls": 0,
            "victims_selected": 0,
        }

    @property
    def tracker(self) -> BlockHotnessTracker:
        return self._tracker

    def score_request(self, seq_group: "SequenceGroup") -> float:
        """Compute aggregate block-level score for a sequence group.

        Returns the mean hotness score across all blocks. Higher = more
        valuable → less likely to be preempted.
        """
        total = 0.0
        count = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            bt = self._bm.block_tables.get(seq.seq_id)
            if bt is None:
                continue
            for block in bt.blocks:
                if block.block_id is not None:
                    total += self._tracker.get_score(block.block_id)
                    count += 1
        self._stats["score_calls"] += 1
        if count == 0:
            return 0.0
        return total / count

    def select_victim(self, candidates) -> int:
        """Return index of lowest-scored candidate in the list."""
        best_idx = len(candidates) - 1
        best_score = float("inf")
        for i, sg in enumerate(candidates):
            s = self.score_request(sg)
            if s < best_score:
                best_score = s
                best_idx = i
        self._stats["victims_selected"] += 1
        return best_idx

    def get_stats(self) -> Dict[str, Any]:
        return {**self._stats, "tracker": self._tracker.get_stats()}


# ---------------------------------------------------------------------------
# Scheduler source-level patch
# ---------------------------------------------------------------------------

_PATCH_MARKER = "# ORCHKV_BLOCK_SCORE_v4"


def apply_scheduler_patch(scheduler_path: str) -> bool:
    """Apply block-level scoring patch to vLLM's scheduler.py.

    Works on VANILLA vLLM 0.7.x (no prior OrchKv patches needed).
    Inserts:
      1. Scheduler.__init__: creates BlockHotnessTracker + RequestScorer
      2. Victim selection: replaces LIFO pop() with block-level scoring

    Idempotent—safe to call multiple times.
    """
    import shutil

    with open(scheduler_path, "r") as f:
        content = f.read()

    if _PATCH_MARKER in content:
        logger.info("Block-score patch already applied")
        return True

    # ---- Patch 1: Add init code after self.prev_prompt ----
    anchor1 = "        # Did we schedule a prompt at previous step?\n        self.prev_prompt = False"
    patch1 = (
        "        # Did we schedule a prompt at previous step?\n"
        "        self.prev_prompt = False\n"
        f"        {_PATCH_MARKER}\n"
        "        self._orchkv_block_scorer = None\n"
        "        if os.environ.get('ORCHKV_BLOCK_SCORE', '0') == '1':\n"
        "            try:\n"
        "                from orchkv.vllm_integration.block_level_swap"
        " import (\n"
        "                    RequestScorer, BlockHotnessTracker)\n"
        "                self._orchkv_block_scorer = RequestScorer(\n"
        "                    block_manager=self.block_manager,\n"
        "                    tracker=BlockHotnessTracker(ema_lambda=0.3))\n"
        "                logger.info(\n"
        "                    'OrchKvCache block-level scoring ENABLED')\n"
        "            except Exception as _e:\n"
        "                logger.warning(\n"
        "                    'OrchKvCache block scoring init failed: %s',"
        " _e)"
    )

    if anchor1 not in content:
        logger.error("Anchor 1 (init) not found")
        return False
    content = content.replace(anchor1, patch1, 1)

    # ---- Patch 2: Replace LIFO victim selection with scoring ----
    anchor2 = (
        "                if running_queue:\n"
        "                    # Preempt the lowest-priority sequence"
        " group.\n"
        "                    victim_seq_group = running_queue.pop()"
    )
    patch2 = (
        "                if running_queue:\n"
        "                    # Preempt the lowest-priority sequence"
        " group.\n"
        "                    _bscorer = getattr(\n"
        "                        self, '_orchkv_block_scorer', None)\n"
        "                    if _bscorer is not None:\n"
        "                        _vidx = _bscorer.select_victim(\n"
        "                            running_queue)\n"
        "                        victim_seq_group = running_queue["
        "_vidx]\n"
        "                        del running_queue[_vidx]\n"
        "                    else:\n"
        "                        victim_seq_group = running_queue.pop()"
    )

    if anchor2 not in content:
        logger.error("Anchor 2 (victim selection) not found")
        return False
    content = content.replace(anchor2, patch2, 1)

    # Write
    bak = scheduler_path + ".pre_blockscore.bak"
    shutil.copy2(scheduler_path, bak)
    with open(scheduler_path, "w") as f:
        f.write(content)
    logger.info("Applied block-score patch (backup: %s)", bak)
    return True


# ---------------------------------------------------------------------------
# Install helper
# ---------------------------------------------------------------------------

def install_block_scoring(
    scheduler: "Scheduler",
    tm_handle: Optional[int] = None,
    ema_lambda: float = 0.3,
) -> Optional[RequestScorer]:
    """Install block-level scoring on an existing scheduler.

    Patches the scheduler source and creates a RequestScorer.
    Requires ORCHKV_SWAP=1 and ORCHKV_BLOCK_SCORE=1 env vars,
    or call this function after engine creation.
    """
    import vllm.core.scheduler as sched_module
    apply_scheduler_patch(sched_module.__file__)

    existing = getattr(scheduler, "_orchkv_block_scorer", None)
    if existing is not None:
        return existing

    tracker = BlockHotnessTracker(tm_handle=tm_handle, ema_lambda=ema_lambda)
    scorer = RequestScorer(
        block_manager=scheduler.block_manager,
        tracker=tracker,
    )
    scheduler._orchkv_block_scorer = scorer
    scheduler.orchkv_enabled = True
    logger.info("Block-level scoring installed on scheduler")
    return scorer
