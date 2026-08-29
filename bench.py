#!/usr/bin/env python3
"""Single-request decode benchmark against an OpenAI-compatible endpoint.

Methodology matched to the v100-skinny repo: streaming request, TTFT
(prefill) separated from decode, decode tok/s = (completion_tokens - 1) /
(t_last_token - t_first_token). Sampling profiles: 'ninfer' (temp 0.6,
top-p 0.95, top-k 20, presence 1.0, thinking on) and 'greedy' (temp 0.0).

Usage: bench.py --url http://127.0.0.1:8000 --model NAME [--seeds 1001 2002 ...]
"""
import argparse
import json
import time
import urllib.request

PROMPTS = {
    "math": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b$.",
    "code": "Write a complete red-black tree implementation in Python with insert, delete and search. Code only, no explanation.",
    "prose": "Write a short essay (about 500 words) on why old hardware deserves a second life.",
}

PROFILES = {
    "ninfer": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.0},
    "greedy": {"temperature": 0.0, "top_p": 1.0},
}


def run_one(url: str, model: str, prompt: str, profile: dict, seed: int,
            max_tokens: int, thinking: bool) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
        **profile,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    t_first = None
    t_last = None
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices")
            if choices:
                delta = choices[0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
    ctok = usage["completion_tokens"] if usage else 0
    decode_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    return {
        "completion_tokens": ctok,
        "ttft_s": round((t_first - t_start), 3) if t_first else None,
        "decode_s": round(decode_s, 3),
        "decode_tok_s": round((ctok - 1) / decode_s, 1) if decode_s > 0 and ctok > 1 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 2002, 3003])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--cells", nargs="+", default=["math", "code", "prose"])
    ap.add_argument("--profile", choices=list(PROFILES), default="ninfer")
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    profile = PROFILES[args.profile]
    results = []
    for cell in args.cells:
        for seed in args.seeds:
            r = run_one(args.url, args.model, PROMPTS[cell], profile, seed,
                        args.max_tokens, thinking=not args.no_thinking)
            r.update({"cell": cell, "seed": seed, "label": args.label, "profile": args.profile})
            results.append(r)
            print(json.dumps(r), flush=True)

    for cell in args.cells:
        rs = [r["decode_tok_s"] for r in results if r["cell"] == cell and r["decode_tok_s"]]
        if rs:
            mean = sum(rs) / len(rs)
            spread = (max(rs) - min(rs)) / 2 if len(rs) > 1 else 0.0
            print(f"# {args.label} {cell}: {mean:.1f} ± {spread:.1f} tok/s (n={len(rs)})")


if __name__ == "__main__":
    main()
