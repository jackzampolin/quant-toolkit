import unittest

import torch

from nvfp4_codec import decode_nvfp4, encode_nvfp4, pack_fp4, unpack_fp4


class Nvfp4CodecTests(unittest.TestCase):
    def test_pack_order_and_codebook(self):
        values = torch.tensor(
            [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]],
            dtype=torch.float32,
        )
        packed = pack_fp4(values)
        self.assertEqual(
            packed.tolist(), [[0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE]]
        )
        torch.testing.assert_close(unpack_fp4(packed), values)

    def test_encode_decode_is_deterministic_and_finite(self):
        weight = torch.linspace(-0.25, 0.25, 64, dtype=torch.float32).reshape(4, 16)
        first = encode_nvfp4(weight)
        second = encode_nvfp4(weight)
        for left, right in zip(first, second, strict=True):
            self.assertTrue(torch.equal(left, right))
        decoded = decode_nvfp4(*first)
        self.assertEqual(decoded.shape, weight.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_secondary_scale_override_is_preserved(self):
        weight = torch.linspace(-0.1, 0.1, 32, dtype=torch.float32).reshape(2, 16)
        override = torch.tensor(1.0e-4, dtype=torch.float32)
        packed, block_scale, scale_2 = encode_nvfp4(weight, scale_2=override)
        self.assertTrue(torch.equal(scale_2, override))
        self.assertTrue(torch.isfinite(decode_nvfp4(packed, block_scale, scale_2)).all())

    def test_rejects_invalid_layout(self):
        with self.assertRaises(ValueError):
            encode_nvfp4(torch.ones(3, 15))


if __name__ == "__main__":
    unittest.main()
