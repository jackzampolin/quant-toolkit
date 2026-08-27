import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from rescale_glm53_nvfp4_input_scales import main


class Glm53Nvfp4InputScaleRescaleTests(unittest.TestCase):
    def test_projection_factors_and_gate_up_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = "model.language_model.layers.3.mlp.experts.0."
            source = root / "source.safetensors"
            output = root / "output.safetensors"
            receipt = root / "receipt.json"
            save_file(
                {
                    prefix + "gate_proj.input_scale": torch.tensor(2.0),
                    prefix + "up_proj.input_scale": torch.tensor(2.0),
                    prefix + "down_proj.input_scale": torch.tensor(3.0),
                },
                source,
            )
            self.assertEqual(
                main(
                    [
                        "--source-scales",
                        str(source),
                        "--output-scales",
                        str(output),
                        "--receipt",
                        str(receipt),
                        "--gate-up-factor",
                        "0.5",
                        "--down-factor",
                        "0.25",
                    ]
                ),
                0,
            )
            tensors = load_file(output)
            self.assertEqual(float(tensors[prefix + "gate_proj.input_scale"]), 1.0)
            self.assertEqual(float(tensors[prefix + "up_proj.input_scale"]), 1.0)
            self.assertEqual(float(tensors[prefix + "down_proj.input_scale"]), 0.75)
            document = json.loads(receipt.read_text())
            self.assertEqual(document["topology"]["input_scale_tensors"], 3)
            self.assertTrue(document["topology"]["gate_up_tied"])


if __name__ == "__main__":
    unittest.main()
