"""Tests for Config and SamplingParams validation — no model loading required."""
import os
import pytest
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


@pytest.fixture(scope="module")
def model_path():
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    return MODEL_PATH


# ─── Config ───────────────────────────────────────────────────────────────────

class TestConfig:

    def test_valid_config(self, model_path):
        cfg = Config(model_path, max_model_len=128)
        assert cfg.hf_config is not None
        assert cfg.max_model_len == 128
        assert cfg.kvcache_block_size == 256
        assert cfg.memory_utilization == 0.75

    def test_max_model_len_capped_by_model(self, model_path):
        # Setting max_model_len beyond the model's max_position_embeddings should
        # be silently capped. Also raise max_num_batched_tokens so that its
        # assert does not fire before the cap takes effect.
        large = 50000
        cfg = Config(model_path, max_model_len=large, max_num_batched_tokens=large)
        assert cfg.max_model_len <= cfg.hf_config.max_position_embeddings

    def test_invalid_model_path(self):
        with pytest.raises((AssertionError, Exception)):
            Config("/nonexistent/path/to/model")

    def test_kvcache_block_size_zero_rejected(self, model_path):
        with pytest.raises(AssertionError):
            Config(model_path, kvcache_block_size=0)

    def test_kvcache_block_size_negative_rejected(self, model_path):
        with pytest.raises(AssertionError):
            Config(model_path, kvcache_block_size=-1)

    def test_max_num_batched_tokens_less_than_max_model_len_rejected(self, model_path):
        # max_num_batched_tokens must be >= max_model_len
        with pytest.raises(AssertionError):
            Config(model_path, max_model_len=512, max_num_batched_tokens=128)

    def test_kvcache_block_size_nonstandard(self, model_path):
        # Non-power-of-2 block size should be accepted (Triton constraint removed)
        cfg = Config(model_path, kvcache_block_size=64)
        assert cfg.kvcache_block_size == 64

    def test_memory_utilization_stored(self, model_path):
        cfg = Config(model_path, memory_utilization=0.5)
        assert cfg.memory_utilization == 0.5

    def test_num_kvcache_blocks_initially_unset(self, model_path):
        cfg = Config(model_path)
        assert cfg.num_kvcache_blocks == -1  # set at runtime by model_runner

    def test_device_auto_resolves(self, model_path):
        # "auto" should resolve to a concrete device string, never stay as "auto"
        cfg = Config(model_path, device="auto")
        assert cfg.device in ("cpu", "mps")

    def test_device_cpu_explicit(self, model_path):
        cfg = Config(model_path, device="cpu")
        assert cfg.device == "cpu"

    def test_device_default_is_auto(self, model_path):
        # Default device is "auto", which resolves at post_init time
        cfg = Config(model_path)
        assert cfg.device in ("cpu", "mps")


# ─── SamplingParams ───────────────────────────────────────────────────────────

class TestSamplingParams:

    def test_defaults(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.max_tokens == 64
        assert sp.ignore_eos is False

    def test_custom_values(self):
        sp = SamplingParams(temperature=0.5, max_tokens=32, ignore_eos=True)
        assert sp.temperature == 0.5
        assert sp.max_tokens == 32
        assert sp.ignore_eos is True

    def test_temperature_at_boundary(self):
        # Just above the minimum allowed temperature
        sp = SamplingParams(temperature=1e-9)
        assert sp.temperature == 1e-9

    def test_temperature_zero_rejected(self):
        # Greedy (temp=0) is not supported; the assert fires at <= 1e-10
        with pytest.raises(AssertionError):
            SamplingParams(temperature=0.0)

    def test_temperature_exactly_minimum_rejected(self):
        with pytest.raises(AssertionError):
            SamplingParams(temperature=1e-10)

    def test_temperature_below_minimum_rejected(self):
        with pytest.raises(AssertionError):
            SamplingParams(temperature=1e-11)

    def test_high_temperature_accepted(self):
        sp = SamplingParams(temperature=10.0)
        assert sp.temperature == 10.0

    def test_max_tokens_one(self):
        sp = SamplingParams(max_tokens=1)
        assert sp.max_tokens == 1

    def test_max_tokens_large(self):
        sp = SamplingParams(max_tokens=10000)
        assert sp.max_tokens == 10000
