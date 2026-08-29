#!/usr/bin/env python3
"""Qualitaetstest: math-Korrektheit (temp 0, thinking) + Diversitaet (Repo-Stil).

math: AIME-Aufgabe mit bekannter Loesung 70 (17_b teilt 97_b -> b+7 | 56 ->
b in {21, 49} -> Summe 70). Bestanden, wenn die finale Antwort 70 nennt.

Diversitaet: 20 Generierungen (Seeds 1-20, temp 0.6) einer offenen Frage;
gezaehlt werden unterschiedliche normalisierte Antworten. Beschaedigte Quants
kollabieren hier auf wenige Wiederholungen (vgl. v100-skinny-README).
"""
import argparse
import json
import re
import urllib.request

MATH_PROMPT = "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b$."
DIVERSITY_PROMPT = ("Name one specific computer hardware product released in the 1990s. "
                    "Reply with only the product name, nothing else.")


def chat(url, model, prompt, temp, seed, max_tokens, thinking):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp, "top_p": 0.95, "seed": seed,
        "max_completion_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    content = chat(args.url, args.model, MATH_PROMPT, 0.0, 1001, 16384, True)
    tail = content.strip()[-300:]
    nums = re.findall(r"\b\d+\b", tail)
    math_ok = "70" in nums[-3:] if nums else False
    print(json.dumps({"label": args.label, "test": "math", "pass": math_ok,
                      "tail": tail[-160:]}), flush=True)

    answers = []
    for seed in range(1, 21):
        a = chat(args.url, args.model, DIVERSITY_PROMPT, 0.6, seed, 40, False)
        norm = re.sub(r"[^a-z0-9 ]", "", a.strip().lower())[:60]
        answers.append(norm)
    distinct = len(set(answers))
    print(json.dumps({"label": args.label, "test": "diversity",
                      "distinct": distinct, "n": len(answers),
                      "sample": sorted(set(answers))[:8]}), flush=True)
    print(f"# {args.label}: math={'PASS' if math_ok else 'FAIL'}, "
          f"Diversitaet {distinct}/20", flush=True)


if __name__ == "__main__":
    main()
