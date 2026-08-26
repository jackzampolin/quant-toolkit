import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from kld_common import (
    canonical_json_sha256,
    sha256_file,
    summarize_kld,
    tokenwise_kld,
)
from replay_prefill_kld import main as replay_main


class KldReplayTests(unittest.TestCase):
    def test_full_artifact_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_dir = root / "reference"
            candidate_dir = root / "candidate"
            replay_dir = root / "replay"
            reference_dir.mkdir()
            candidate_dir.mkdir()

            generator = torch.Generator().manual_seed(5300)
            reference = torch.randn(5, 23, generator=generator, dtype=torch.float32)
            candidate = reference + 0.05 * torch.randn(
                5, 23, generator=generator, dtype=torch.float32
            )
            input_ids = torch.tensor([17, 19, 23, 29, 31, 37], dtype=torch.int32)

            reference_path = reference_dir / "logits_0.safetensors"
            candidate_path = candidate_dir / "candidate_logits_0.safetensors"
            save_file({"logits": reference, "input_ids": input_ids}, str(reference_path))
            save_file({"logits": candidate, "input_ids": input_ids}, str(candidate_path))

            reference_manifest = {
                "schema": "quant-toolkit.prefill-logits.v1",
                "storage_dtype": "float32",
                "windows": [
                    {
                        "index": 0,
                        "file": reference_path.name,
                        "sha256": sha256_file(reference_path),
                    }
                ],
            }
            reference_manifest["manifest_sha256"] = canonical_json_sha256(
                reference_manifest
            )
            (reference_dir / "manifest.json").write_text(
                json.dumps(reference_manifest, indent=2, sort_keys=True) + "\n"
            )

            kld, reference_top1, candidate_top1 = tokenwise_kld(
                reference, candidate
            )
            window = summarize_kld(kld, reference_top1, candidate_top1)
            window.update(
                {
                    "index": 0,
                    "candidate_file": candidate_path.name,
                    "candidate_file_sha256": sha256_file(candidate_path),
                }
            )
            tokenwise_path = candidate_dir / "tokenwise.safetensors"
            save_file(
                {
                    "kld_nats": kld,
                    "kld_bits": kld / math.log(2.0),
                    "reference_top1": reference_top1,
                    "candidate_top1": candidate_top1,
                },
                str(tokenwise_path),
            )
            candidate_summary = {
                "schema": "quant-toolkit.prefill-kld.v1",
                "candidate_storage_dtype": "float32",
                "reference_manifest": reference_manifest,
                "windows": [window],
                "aggregate": summarize_kld(kld, reference_top1, candidate_top1),
                "tokenwise_file": tokenwise_path.name,
                "tokenwise_file_sha256": sha256_file(tokenwise_path),
            }
            candidate_summary["summary_sha256"] = canonical_json_sha256(
                candidate_summary
            )
            (candidate_dir / "summary.json").write_text(
                json.dumps(candidate_summary, indent=2, sort_keys=True) + "\n"
            )

            self.assertEqual(
                replay_main(
                    [
                        "--reference-logits",
                        str(reference_dir),
                        "--candidate-logits",
                        str(candidate_dir),
                        "--output-dir",
                        str(replay_dir),
                    ]
                ),
                0,
            )
            verification = json.loads((replay_dir / "verification.json").read_text())
            self.assertEqual(
                verification["schema"],
                "quant-toolkit.prefill-kld-verification.v1",
            )
            self.assertLessEqual(
                verification["max_tokenwise_kld_difference"], 1e-12
            )


if __name__ == "__main__":
    unittest.main()
