#!/usr/bin/env python3
"""Quantisierungs-Abstandsmessung: wie weit weicht ein Quant von der Referenz ab?

Der bisherige quality.py ist ein Kollaps-Detektor (bestanden/durchgefallen). Er
kann nicht aufloesen, ob 4 Bit gegenueber 8 Bit ein bisschen oder viel kostet.
Dieses Werkzeug misst den Abstand als Zahl.

Verfahren: Teacher Forcing. Beiden Modellen wird DERSELBE feste Text vorgelegt;
per prompt_logprobs liest man an jeder Position die Verteilung ab, die das Modell
fuer das naechste Token gehabt haette. Kein Generieren -- wuerden beide frei
schreiben, liefen sie nach dem ersten abweichenden Token auseinander und man
vergliche zwei Texte statt zweier Modelle.

Zwei Kennzahlen:
  * top1_agree  -- Anteil der Positionen mit identischer Top-1-Wahl. Anschaulich.
  * kl          -- mittlere KL-Divergenz D(ref || test) ueber die Vereinigung der
                   Top-K-Traeger, renormiert. Schlaegt auch an, wenn die Top-1-Wahl
                   noch stimmt, die Verteilung darunter aber schon kippt.

Nur vLLM: llama.cpps Server liefert keine Logprobs ueber den Prompt (der
OpenAI-echo-Weg gibt leere Listen zurueck). Der 8-Bit-Anker ist deshalb der
FP8-Checkpoint in derselben Engine -- was den Vergleich sogar sauberer macht,
weil Engine und Tokenizer konstant bleiben.
"""
import argparse
import json
import math
import urllib.request


def prompt_distributions(url: str, model: str, text: str, topk: int) -> list[dict[str, float]]:
    """Verteilungen (Token-ID -> logprob) an jeder Prompt-Position."""
    body = {
        "model": model,
        "prompt": text,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": topk,
    }
    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    out = []
    for entry in d["choices"][0].get("prompt_logprobs") or []:
        if not entry:          # erste Position hat keinen Kontext
            continue
        out.append({tid: info["logprob"] for tid, info in entry.items()})
    return out


def kl_divergence(ref: dict[str, float], test: dict[str, float]) -> float:
    """D(ref || test) ueber die Vereinigung der Traeger, beide renormiert.

    Beide Seiten liefern nur ihre Top-K, die Traeger sind also abgeschnitten.
    Fehlende Eintraege bekommen den kleinsten beobachteten Wert der jeweiligen
    Seite als Untergrenze -- das unterschaetzt die Divergenz eher, als sie
    aufzublasen.
    """
    support = set(ref) | set(test)
    ref_floor = min(ref.values()) if ref else -30.0
    test_floor = min(test.values()) if test else -30.0
    p = {t: math.exp(ref.get(t, ref_floor)) for t in support}
    q = {t: math.exp(test.get(t, test_floor)) for t in support}
    zp, zq = sum(p.values()), sum(q.values())
    if zp <= 0 or zq <= 0:
        return 0.0
    total = 0.0
    for t in support:
        pi, qi = p[t] / zp, q[t] / zq
        if pi > 0 and qi > 0:
            total += pi * math.log(pi / qi)
    return total


def compare(ref_dists, test_dists) -> dict:
    n = min(len(ref_dists), len(test_dists))
    agree = 0
    kls = []
    for i in range(n):
        r, t = ref_dists[i], test_dists[i]
        if not r or not t:
            continue
        if max(r, key=r.get) == max(t, key=t.get):
            agree += 1
        kls.append(kl_divergence(r, t))
    return {
        "positions": n,
        "top1_agree": round(agree / n, 4) if n else None,
        "kl_mean": round(sum(kls) / len(kls), 5) if kls else None,
        "kl_p95": round(sorted(kls)[int(0.95 * (len(kls) - 1))], 5) if kls else None,
    }


TEXTS = {
    "de_prosa": (
        "Die Dampfmaschine veraenderte im 18. Jahrhundert die Arbeitswelt grundlegend. "
        "Was zuvor von Wasserraedern oder Muskelkraft abhing, liess sich nun unabhaengig "
        "von Standort und Jahreszeit betreiben. Fabriken rueckten naeher an die Staedte, "
        "und die Eisenbahn verkuerzte Entfernungen, die vorher Tagesreisen bedeutet hatten."
    ),
    "code": (
        "def binary_search(items, target):\n"
        "    low, high = 0, len(items) - 1\n"
        "    while low <= high:\n"
        "        mid = (low + high) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            low = mid + 1\n"
        "        else:\n"
        "            high = mid - 1\n"
        "    return -1\n"
    ),
    "fakten": (
        "The mitochondrion is the organelle responsible for oxidative phosphorylation. "
        "It contains its own circular DNA, inherited maternally in most animals, and is "
        "widely believed to descend from an endosymbiotic alphaproteobacterium."
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="vLLM-Endpunkt des Pruef-Quants")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--dump", help="Verteilungen als Referenz hierhin schreiben")
    ap.add_argument("--against", help="frueher gedumpte Referenz laden und vergleichen")
    args = ap.parse_args()

    dists = {name: prompt_distributions(args.url, args.model, text, args.topk)
             for name, text in TEXTS.items()}

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(dists, f)
        print(json.dumps({"label": args.label, "role": "referenz",
                          "positions": {k: len(v) for k, v in dists.items()}}))
        return

    if not args.against:
        ap.error("entweder --dump (Referenz erzeugen) oder --against (vergleichen)")

    with open(args.against) as f:
        ref = json.load(f)
    for name in TEXTS:
        res = compare(ref[name], dists[name])
        print(json.dumps({"label": args.label, "text": name, **res}))


if __name__ == "__main__":
    main()
