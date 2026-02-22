import os
import pytest
from nanovllm import LLM

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skipped unless --run-slow is passed)"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Pass --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def llm():
    """Single LLM instance reused across all inference tests in the module.
    Uses a small max_model_len to keep warmup fast on CPU."""
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    return LLM(
        MODEL_PATH,
        max_model_len=128,
        max_num_batched_tokens=256,
        max_num_seqs=4,
    )
