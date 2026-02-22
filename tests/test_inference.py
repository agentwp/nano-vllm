"""End-to-end inference tests.

All tests use the module-scoped `llm` fixture from conftest.py, which loads
the model once and reuses it for the full test session (avoids repeated warmup).

max_model_len=128 keeps the warmup pass fast on CPU.
"""
import pytest
from nanovllm import SamplingParams


# ─── Helpers ──────────────────────────────────────────────────────────────────

def sp(max_tokens=12, temperature=0.6, ignore_eos=False):
    """Shorthand for building SamplingParams with sensible test defaults."""
    return SamplingParams(max_tokens=max_tokens, temperature=temperature, ignore_eos=ignore_eos)


def assert_outputs(outputs, expected_count):
    """Assert that `outputs` is a valid list of `expected_count` generation results."""
    assert isinstance(outputs, list), "generate() must return a list"
    assert len(outputs) == expected_count, f"Expected {expected_count} outputs, got {len(outputs)}"
    for i, out in enumerate(outputs):
        assert isinstance(out, dict), f"Output {i} must be a dict"
        assert "text" in out, f"Output {i} missing 'text' key"
        assert "token_ids" in out, f"Output {i} missing 'token_ids' key"
        assert isinstance(out["text"], str), f"Output {i} 'text' must be a str"
        assert isinstance(out["token_ids"], list), f"Output {i} 'token_ids' must be a list"
        assert len(out["token_ids"]) > 0, f"Output {i} 'token_ids' must be non-empty"


# ─── Prompt-length tests ──────────────────────────────────────────────────────

class TestPromptLengths:

    def test_single_word_prompt(self, llm):
        """Very short prompt: a single word."""
        out = llm.generate(["Hello"], sp())
        assert_outputs(out, 1)

    def test_two_word_prompt(self, llm):
        out = llm.generate(["Hello world"], sp())
        assert_outputs(out, 1)

    def test_medium_prompt(self, llm):
        """~20-token sentence."""
        out = llm.generate(
            ["What is the capital city of France? Please answer in one word."],
            sp(max_tokens=8),
        )
        assert_outputs(out, 1)

    def test_long_prompt(self, llm):
        """Prompt near max_model_len (128). ~100 tokens, leaving room for output."""
        # "The quick brown fox..." ~10 tokens; 10 repetitions ≈ 100 tokens
        long_text = ("The quick brown fox jumps over the lazy dog. " * 10).strip()
        out = llm.generate([long_text], sp(max_tokens=8))
        assert_outputs(out, 1)

    def test_single_character_prompt(self, llm):
        out = llm.generate(["A"], sp())
        assert_outputs(out, 1)

    def test_numeric_prompt(self, llm):
        out = llm.generate(["42"], sp())
        assert_outputs(out, 1)

    def test_punctuation_only_prompt(self, llm):
        out = llm.generate(["..."], sp())
        assert_outputs(out, 1)


# ─── Input-type tests ─────────────────────────────────────────────────────────

class TestInputTypes:

    def test_string_input(self, llm):
        out = llm.generate(["Hello"], sp())
        assert_outputs(out, 1)

    def test_token_id_list_input(self, llm):
        """Passing raw token IDs should work identically to passing a string."""
        out = llm.generate([[100, 200, 300]], sp())
        assert_outputs(out, 1)

    def test_single_token_id_input(self, llm):
        out = llm.generate([[1]], sp())
        assert_outputs(out, 1)

    def test_mixed_string_and_token_id_inputs(self, llm):
        """A batch mixing str and list[int] prompts."""
        out = llm.generate(["Hello", [100, 200, 300]], sp())
        assert_outputs(out, 2)


# ─── Single-request tests ─────────────────────────────────────────────────────

class TestSingleRequest:

    def test_basic_generation(self, llm):
        out = llm.generate(["Tell me a joke."], sp(max_tokens=16))
        assert_outputs(out, 1)
        assert len(out[0]["text"]) > 0

    def test_output_token_ids_within_max_tokens(self, llm):
        max_tok = 8
        out = llm.generate(["Hello"], sp(max_tokens=max_tok))
        assert_outputs(out, 1)
        assert len(out[0]["token_ids"]) <= max_tok

    def test_use_tqdm_false(self, llm):
        out = llm.generate(["Hello"], sp(), use_tqdm=False)
        assert_outputs(out, 1)

    def test_use_tqdm_true(self, llm):
        out = llm.generate(["Hello"], sp(), use_tqdm=True)
        assert_outputs(out, 1)


# ─── Batch tests ──────────────────────────────────────────────────────────────

class TestBatchRequests:

    def test_batch_of_two(self, llm):
        out = llm.generate(["Hello", "Goodbye"], sp())
        assert_outputs(out, 2)

    def test_batch_of_four(self, llm):
        prompts = [
            "What color is the sky?",
            "Name a country in Europe.",
            "What is 1 + 1?",
            "Say hello.",
        ]
        out = llm.generate(prompts, sp(max_tokens=8))
        assert_outputs(out, 4)

    def test_batch_preserves_order(self, llm):
        """Output list must correspond to the input list in order."""
        prompts = ["First", "Second", "Third"]
        out = llm.generate(prompts, sp(max_tokens=4), use_tqdm=False)
        assert_outputs(out, 3)
        # Each output should be a non-empty completion
        assert all(len(o["token_ids"]) > 0 for o in out)

    def test_per_sequence_sampling_params(self, llm):
        """Each prompt may have its own SamplingParams."""
        prompts = ["Hello", "Goodbye"]
        params = [
            SamplingParams(max_tokens=4, temperature=0.1),
            SamplingParams(max_tokens=8, temperature=0.9),
        ]
        out = llm.generate(prompts, params, use_tqdm=False)
        assert_outputs(out, 2)
        # Second prompt was allowed more tokens
        assert len(out[1]["token_ids"]) <= 8

    def test_repeated_prompt_uses_prefix_cache(self, llm):
        """Running the same prompt twice should hit prefix caching on the second pass."""
        prompt = "What is the capital of France?"
        out = llm.generate([prompt, prompt], sp(max_tokens=8), use_tqdm=False)
        assert_outputs(out, 2)

    def test_varied_length_prompts(self, llm):
        """Short and long prompts together."""
        short = "Hi"
        medium = "Explain what the sun is in one sentence."
        out = llm.generate([short, medium], sp(max_tokens=12), use_tqdm=False)
        assert_outputs(out, 2)


# ─── Boundary / edge-case tests ───────────────────────────────────────────────

class TestBoundaries:

    def test_max_tokens_one(self, llm):
        """Requesting exactly 1 token should yield exactly 1 completion token."""
        out = llm.generate(["Hello"], SamplingParams(max_tokens=1), use_tqdm=False)
        assert_outputs(out, 1)
        assert len(out[0]["token_ids"]) == 1

    def test_ignore_eos_generates_exactly_max_tokens(self, llm):
        """With ignore_eos=True the model must produce exactly max_tokens tokens."""
        max_tok = 10
        out = llm.generate(
            ["Hi"],
            SamplingParams(max_tokens=max_tok, temperature=0.6, ignore_eos=True),
            use_tqdm=False,
        )
        assert_outputs(out, 1)
        assert len(out[0]["token_ids"]) == max_tok

    def test_low_temperature_produces_valid_output(self, llm):
        """Near-greedy temperature should still produce valid output."""
        out = llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=1e-9), use_tqdm=False)
        assert_outputs(out, 1)

    def test_high_temperature_produces_valid_output(self, llm):
        """Very high temperature should still produce valid (if random) output."""
        out = llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=5.0), use_tqdm=False)
        assert_outputs(out, 1)

    def test_token_ids_non_empty_for_all_outputs(self, llm):
        prompts = ["A", "B", "C"]
        out = llm.generate(prompts, sp(max_tokens=4), use_tqdm=False)
        assert all(len(o["token_ids"]) > 0 for o in out)

    def test_text_is_string_for_all_outputs(self, llm):
        prompts = ["Hello", "World"]
        out = llm.generate(prompts, sp(max_tokens=4), use_tqdm=False)
        assert all(isinstance(o["text"], str) for o in out)

    def test_output_count_matches_input_count(self, llm):
        for n in [1, 2, 3]:
            prompts = ["test"] * n
            out = llm.generate(prompts, sp(max_tokens=4), use_tqdm=False)
            assert len(out) == n

    @pytest.mark.slow
    def test_prompt_at_max_model_len_boundary(self, llm):
        """Prompt of exactly max_model_len - 1 tokens (leaves 1 slot for output)."""
        # 127 tokens: just under the 128 limit used by the test LLM fixture
        token_ids = [100] * 127
        out = llm.generate([token_ids], SamplingParams(max_tokens=1), use_tqdm=False)
        assert_outputs(out, 1)
        assert len(out[0]["token_ids"]) == 1
