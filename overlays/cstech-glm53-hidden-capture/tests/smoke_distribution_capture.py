#!/usr/bin/env python3
"""CPU-only build smoke test for the environment-gated capture module."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file

from vllm.v1.worker.gpu import distribution_capture


class IdentityFinalNorm(torch.nn.Module):
    def compute_pre_lm_head_hidden_states(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return hidden_states


def make_batch(computed: int, scheduled: int) -> SimpleNamespace:
    return SimpleNamespace(
        idx_mapping_np=np.array([0]),
        is_prefilling_np=np.array([True]),
        num_computed_prefill_tokens_np=np.array([computed]),
        num_scheduled_tokens=np.array([scheduled]),
        query_start_loc_np=np.array([0, scheduled]),
        req_ids=["capture/request"],
    )


with tempfile.TemporaryDirectory() as temporary_directory:
    os.environ["VLLM_KLD_HIDDEN_CAPTURE_DIR"] = temporary_directory
    distribution_capture._is_capture_rank = lambda: True
    model = IdentityFinalNorm()

    first = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    distribution_capture.capture_pre_lm_head_prompt_hidden_states(
        model, first, make_batch(0, 2), np.array([5])
    )
    second = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    distribution_capture.capture_pre_lm_head_prompt_hidden_states(
        model, second, make_batch(2, 3), np.array([5])
    )

    request_directory = Path(temporary_directory) / "capture_request"
    first_path = request_directory / "hidden.rows-000000-000002.safetensors"
    second_path = request_directory / "hidden.rows-000002-000005.safetensors"
    torch.testing.assert_close(load_file(first_path)["hidden_states"], first)
    torch.testing.assert_close(load_file(second_path)["hidden_states"], second)
