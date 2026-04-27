"""
Quick local test to verify the SSE chunk parsing logic handles all cases correctly.
Simulates the exact vLLM streaming response format for both baseline and speculative decoding.

Run: python3 benchmark/test_parsing.py
"""

import json

def simulate_parse(chunks_raw: list[str], label: str) -> dict:
    """
    Runs the EXACT same parsing logic as load_test.py's send_request() 
    against a list of raw SSE data strings.
    """
    output_text = ""
    chunk_count = 0
    server_output_tokens = None

    for data_str in chunks_raw:
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Check for usage field (sent in the final chunk when stream_options is set).
        # This chunk typically has an empty "choices" array, so we check usage first.
        usage = chunk.get("usage")
        if usage and "completion_tokens" in usage:
            server_output_tokens = usage["completion_tokens"]

        # Extract content from the delta. The usage-only chunk has choices=[]
        # so we must guard against an empty array.
        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        if content:
            output_text += content
            chunk_count += 1

    output_tokens = server_output_tokens if server_output_tokens is not None else chunk_count

    result = {
        "chunk_count": chunk_count,
        "server_output_tokens": server_output_tokens,
        "final_output_tokens": output_tokens,
        "text_length": len(output_text),
    }
    return result


# ── Test 1: Baseline (greedy) — each chunk = 1 token, no usage field ──────────
baseline_chunks = [
    '{"choices":[{"delta":{"content":"Hello"}}]}',
    '{"choices":[{"delta":{"content":" world"}}]}',
    '{"choices":[{"delta":{"content":"!"}}]}',
    '{"choices":[{"delta":{"content":" How"}}]}',
    '{"choices":[{"delta":{"content":" are"}}]}',
    "[DONE]",
]

# ── Test 2: Eagle3 (speculative) — fewer chunks, multi-token content, usage at end ──
eagle3_chunks = [
    '{"choices":[{"delta":{"content":"Hello world"}}]}',       # 2 tokens in 1 chunk
    '{"choices":[{"delta":{"content":"! How are"}}]}',         # 3 tokens in 1 chunk
    '{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}',  # usage-only chunk (empty choices!)
    "[DONE]",
]

# ── Test 3: Server that ignores stream_options (e.g. MLX) — no usage field ────
mlx_chunks = [
    '{"choices":[{"delta":{"content":"Hello"}}]}',
    '{"choices":[{"delta":{"content":" world"}}]}',
    '{"choices":[{"delta":{"content":"!"}}]}',
    "[DONE]",
]

# ── Test 4: Edge case — empty choices mid-stream (shouldn't crash) ────────────
edge_chunks = [
    '{"choices":[{"delta":{"role":"assistant"}}]}',           # role-only delta, no content
    '{"choices":[{"delta":{"content":"Hi"}}]}',
    '{"choices":[]}',                                         # weird empty choices mid-stream
    '{"choices":[{"delta":{"content":" there"}}]}',
    '{"choices":[],"usage":{"completion_tokens":3}}',
    "[DONE]",
]


print("=" * 60)
print("  load_test.py SSE Parsing Verification")
print("=" * 60)

tests = [
    ("Baseline (greedy, no usage)", baseline_chunks, 5, None, 5),
    ("Eagle3 (speculative, with usage)", eagle3_chunks, 2, 5, 5),
    ("MLX (no usage support, fallback)", mlx_chunks, 3, None, 3),
    ("Edge case (empty choices mid-stream)", edge_chunks, 2, 3, 3),
]

all_passed = True
for name, chunks, expect_chunks, expect_server_tokens, expect_final in tests:
    result = simulate_parse(chunks, name)
    
    passed = (
        result["chunk_count"] == expect_chunks
        and result["server_output_tokens"] == expect_server_tokens
        and result["final_output_tokens"] == expect_final
    )
    
    status = "✅ PASS" if passed else "❌ FAIL"
    if not passed:
        all_passed = False
    
    print(f"\n{status}: {name}")
    print(f"  Chunks counted     : {result['chunk_count']} (expected {expect_chunks})")
    print(f"  Server tokens      : {result['server_output_tokens']} (expected {expect_server_tokens})")
    print(f"  Final output_tokens: {result['final_output_tokens']} (expected {expect_final})")

print("\n" + "=" * 60)
if all_passed:
    print("  ALL 4 TESTS PASSED — parsing logic is correct")
else:
    print("  SOME TESTS FAILED — DO NOT RUN THE EXPERIMENT")
print("=" * 60)
