import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("tile_streaming_model.py")
SPEC = importlib.util.spec_from_file_location("tile_streaming_model", MODULE_PATH)
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class TileStreamingModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = MODEL.load_json(MODEL.DEFAULT_PROFILE)
        cls.params = MODEL.ffn_parameters(cls.profile)

    def test_existing_whole_ffn_sensitivity_cost_is_reproduced(self):
        self.assertAlmostEqual(
            MODEL.whole_tensor_latency_us(self.profile, self.params),
            638.29952,
            places=5,
        )

    def test_one_full_tile_has_no_compute_pipeline_gain(self):
        monolithic = MODEL.whole_tensor_latency_us(self.profile, self.params)
        tiled = MODEL.internal_dataflow_latency_us(
            self.profile, self.params, self.params["tokens"], True
        )
        self.assertAlmostEqual(monolithic, tiled, places=6)

    def test_independent_engines_beat_shared_engine_for_multiple_tiles(self):
        shared = MODEL.internal_dataflow_latency_us(self.profile, self.params, 16, False)
        independent = MODEL.internal_dataflow_latency_us(self.profile, self.params, 16, True)
        self.assertLess(independent, shared)

    def test_per_tile_dma_exposes_fixed_latency_penalty(self):
        per_tile, _ = MODEL.end_to_end_stream_latency_us(
            self.profile, self.params, 4, 2, True, "per_tile_dma"
        )
        descriptor, _ = MODEL.end_to_end_stream_latency_us(
            self.profile, self.params, 4, 2, True, "streaming_descriptor"
        )
        self.assertGreater(per_tile, descriptor)

    def test_double_buffer_does_not_hurt_stream_schedule(self):
        single, _ = MODEL.end_to_end_stream_latency_us(
            self.profile, self.params, 16, 1, True, "streaming_descriptor"
        )
        double, _ = MODEL.end_to_end_stream_latency_us(
            self.profile, self.params, 16, 2, True, "streaming_descriptor"
        )
        self.assertLessEqual(double, single)

    def test_all_four_tile_boundaries_have_finite_buffer_effect(self):
        single, _ = MODEL.end_to_end_stream_latency_us(
            self.profile, self.params, 8, 1, True, "streaming_descriptor"
        )
        double, _ = MODEL.end_to_end_stream_latency_us(
            self.profile, self.params, 8, 2, True, "streaming_descriptor"
        )
        self.assertLess(double, single)

    def test_conservative_memory_scales_with_buffer_count(self):
        single = MODEL.conservative_buffer_bytes(self.params, 16, 1)
        double = MODEL.conservative_buffer_bytes(self.params, 16, 2)
        self.assertEqual(double["activation_buffers"], 2 * single["activation_buffers"])
        self.assertEqual(double["resident_low_bit_weights"], single["resident_low_bit_weights"])


if __name__ == "__main__":
    unittest.main()
