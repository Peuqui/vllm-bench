# Benchmark-Kampagne: Qwen3.8-27B über Engines & GPUs (Mini, 2026-08-24)

Auslöser: [v100-skinny](https://github.com/dnv2003/v100-skinny) (NVFP4 auf Volta
per Software). Ziel laut Peuqui: **„Solange vLLM auf unserem System schneller ist
als llama.cpp mit MTP, haben wir gewonnen."**

## Methodik

- Single-Request-Decode, Streaming; **Decode tok/s = (completion_tokens − 1) / (t_letzter − t_erster Token)** — TTFT/Prefill separat, zählt nicht in die Decode-Rate.
- Harness: [bench.py](bench.py). 3 Zellen (math = AIME-artig mit Thinking, code = Red-Black-Tree, prose = Essay), Seeds 1001/2002/3003, max 4096 Tokens.
- Sampling („ninfer-Profil", identisch zum v100-skinny-Repo): temp 0.6, top-p 0.95, top-k 20, presence 1.0, thinking on.
- Angabe: Mittelwert ± halbe Spannweite (n=3).

## Hardware

| Karte | Anbindung | Bandbreite |
|---|---|---|
| 2× Quadro RTX 8000 48 GB (SM75) | M.2-OCuLink, PCIe Gen3 x4 | ~672 GB/s GDDR6 |
| 3× Tesla V100-PCIE-32GB (SM70) | OCuLink bzw. 1× USB4, Gen3 x4 | ~900 GB/s HBM2 |

GPU 3 (V100) ist produktiv durch das Vigilantia-VLM belegt; Tests liefen auf GPU 0 (RTX) bzw. GPU 1+4 (V100, TP=2).

## Ergebnisse: Qwen3.8-27B, Decode tok/s

| # | Engine / Quant / GPU | Speculation | math | code | prose |
|---|---|---|---:|---:|---:|
| 1a | vLLM 0.27.1 · AWQ-INT4 · 1× RTX 8000 | — | 28,5 ± 0,1 | 27,4 ± 0,0 | 27,5 ± 0,2 |
| 1b | vLLM 0.27.1 · AWQ-INT4 · 1× RTX 8000 | MTP k=7 | 51,4 ± 0,8 | 33,0 ± 3,9 | 46,0 ± 1,3 |
| 1c | vLLM 0.27.1 · AWQ-INT4 · 1× RTX 8000 | **MTP k=3** | 52,2 ± 1,1 | 36,3 ± 1,4 | 46,0 ± 1,8 |
| 2 | llama.cpp b10562 (Prod-Config) · **Q8_K_XL** · 1× RTX 8000 | MTP n=3 | 31,5 ± 0,9 | 26,3 ± 1,2 | 34,8 ± 1,5 |
| 3 | v100-skinny · NVFP4-mixed (RadixArk) · 1× RTX 8000 | MTP k=7 | — | — | — |
| 4a | **v100-skinny · NVFP4-mixed (RadixArk) · 2× V100 TP2** | **MTP k=7** | **88,1 ± 1,4** | 58,6 ± 3,8 | **86,3 ± 3,4** |
| 4b | v100-skinny · NVFP4-mixed (RadixArk) · 2× V100 TP2 | MTP k=3 | 80,5 ± 3,5 | 60,2 ± 5,4 | 81,1 ± 0,2 |
| 4c | v100-skinny · NVFP4 all (Unsloth) · 2× V100 TP2 | MTP k=7 | 54,0 ± 0,6 | 34,9 ± 1,3 | 55,9 ± 1,8 |
| 5a | vLLM 0.27.1 · AWQ-INT4 · 2× RTX 8000 TP2 | — | 42,2 ± 0,2 | 40,9 ± 0,0 | 41,1 ± 0,2 |
| 5b | vLLM 0.27.1 · AWQ-INT4 · 2× RTX 8000 TP2 | MTP k=3 | 67,6 ± 0,5 | 40,2 ± 2,3 | 57,7 ± 2,7 |

Station 3 (Fork auf RTX): bootet mit allen Gates, **aber Outputs unbrauchbar** — siehe Befunde.

## Ergebnisse: Qwen3.5-122B-A10B (MoE, 10B aktiv), Decode tok/s

| Engine / Quant / GPU | Speculation | math | code | prose |
|---|---|---:|---:|---:|
| **llama.cpp (Prod) · Q8_K_XL · 5 GPUs Layer-Split** | MTP n=3 | **48,7 ± 0,9** | **47,0 ± 1,9** | **44,3 ± 0,8** |
| vLLM 0.27.1 · GPTQ-Int4 · 2× RTX 8000 TP2 | MTP k=3 | 29,5 ± 0,2 | 27,7 ± 0,5 | 26,7 ± 0,9 |

**Beim 122B-MoE gewinnt llama.cpp deutlich (+65 %)** — trotz doppelter Weight-Bytes (Q8 vs. INT4): Beim 10B-aktiv-MoE zählt die aggregierte Bandbreite aller 5 Karten entlang des Layer-Splits, und vLLMs GPTQ-MoE läuft auf SM75 nur über den langsamen Legacy-Kernel (Marlin-MoE braucht SM80). Das Siegkriterium gilt beim 27B, kippt beim 122B — Modellgröße entscheidet die Engine-Wahl.

## Kernbefunde

1. **Siegkriterium erfüllt:** vLLM (INT4, MTP k=3) schlägt die produktive llama.cpp-Config (Q8, MTP n=3) auf derselben RTX 8000 in jeder Zelle (+32 % bis +66 %). Fußnote: INT4 liest halb so viele Weight-Bytes wie Q8 — Qualität der 4-bit-Artefakte in Stichproben unauffällig, formale Qualitätsmessung offen.
2. **Die beiden V100s sind die schnellste Konfiguration im Haus:** v100-skinny nativ (TP=2, k=7) liefert ~2,8× die Produktions-llama.cpp-Rate und ~1,7× das beste upstream-vLLM-Ergebnis. Repo-Referenz (4× V100-SXM2 + NVLink): 187 tok/s in der math-Zelle — wir erreichen 88 mit halber Kartenzahl ohne NVLink; pro Karte nahezu Parität.
3. **Optimale MTP-Tiefe ist architekturabhängig:** upstream/RTX → k=3 (der einzelne MTP-Layer verliert bei tiefer Verkettung zu viel Akzeptanz); Fork/Volta → k=7 (8-Zeilen-Kachel macht die breite Verifikation fast gratis). Deckt sich mit der Repo-These.
4. **v100-skinny ist tatsächlich Volta-exklusiv:** Auf SM75 (RTX 8000) bootet der Stack nach drei Aufruf-Fixes und die QPN-Decode-GEMMs rechnen korrekt (flüssiger Text), aber es existiert kein funktionierender Prefill-Pfad: FlashQLA-GDN braucht 84 KB Shared Memory (Turing-Limit: 64 KB), die Triton-Ersatzpfade verlieren den Prompt-Zustand, FLEX_ATTENTION bootet nicht (Paged-KV-Blockgröße 784 ≠ 2er-Potenz). Rettung wäre ein Neubau von flash_attn_v100 für SM75 — geparkt.
5. **Mixed schlägt all-FP4 deutlich (~+60 %):** RadixArks Mixed-Layout (FP8-Attention via QPN8) ist auf demselben Stack klar schneller als Unsloths all-NVFP4 (54/35/56) — die Repo-These, dass die FP8-Regionen der Effizienz-Schlüssel sind, bestätigt sich. Der Fork verarbeitet dabei auch compressed-tensors-NVFP4 (nicht nur ModelOpt).
6. **TP=2 skaliert auf ~1,5× (plain):** upstream 28→42 tok/s über die x4-Links (ideal 2,0) — Host-Staging-Allreduce kostet ~25 % des Gewinns. Mit MTP k=3: 67,6 (math); in der code-Zelle hebt sich MTP auf TP=2 fast auf. Unsloth-Analyse: Rückstand kam vom fehlenden QPN-Prepack für compressed-tensors-Attention (Marlin-Fallback im Decode), nicht von der MTP-Akzeptanz — die war dort sogar 98,4 % vs. 85 % (verdächtig hoch, siehe Qualitäts-TODO).
7. **llama.cpp Multi-GPU = Kapazität, vLLM Multi-GPU = Tempo:** llama.cpps `-sm layer` (Prod) beschleunigt Single-Request nicht; vLLMs echtes TP=2 bringt auf den V100s ~+70 % trotz Host-Staging-Allreduce.

## Ausblick: Merge-Projekt „Hierarchisches PP×TP über beide Generationen"

Peuquis Zielarchitektur für große Modelle (DeepSeek-V4-Flash-Klasse): TP=2
innerhalb der RTX-Gruppe + TP=2 innerhalb der V100-Gruppe, PP über die
Generationsgrenze (vLLM-Standard-Gitter 2×2; 5. Karte bleibt beim VLM).
Arbeitspakete: (1) SupportsPP für qwen3_5 aus upstream 0.27 in den Fork
portieren; (2) korrekte SM75-Prefill-Pfade aus upstream zurückportieren;
(3) **Per-Stufe-Attention-Backend — PFLICHT, nicht Kür:** De-Risk-Tests
zeigten, dass der Fork mit Triton-Prefill auf BEIDEN Generationen falsch
rechnet (Turing: Prompt-Verlust; Volta: Mojibake) — jede Stufe braucht ihren
nativen Pfad; (4) ungleiche Layer-Partition (VLLM_PP_LAYER_PARTITION, fertig);
(5) Architektur-Support der Zielmodelle prüfen (Fork-Basis kennt bereits
deepseek_v4_attention-Ops).

## Merge-Projekt: Sitzungsstand 2026-08-24 abends

**PP läuft im Fork** (Weltneuheit-Kandidat Nr. 1): PP=2 auf 2× V100 bootet und
antwortet kohärent — einziger nötiger Fix waren zwei shape-unsichere
Debug-Traces. Gemessen: 32,3 tok/s plain (erwartungsgemäß kein Decode-Gewinn,
reiner Kapazitätsmodus).

**PP+MTP** (Weltneuheit-Kandidat Nr. 2, existiert auch upstream nicht): In 7
Boot-Iterationen von „Config lehnt ab" bis „Warm-Request läuft" gebracht.
Angewandte Patches (alle Python, dokumentiert, fork_patches synchron):
1. gpu_model_runner: 2 Debug-Traces shape-sicher (IntermediateTensors).
2. qwen3_5_mtp: SupportsPP-Interface deklariert + make_empty_intermediate_tensors
   durchgereicht (supports_pp(Qwen3_5MTP)=True).
3. gpu_model_runner: 3 Drafter-Blöcke mit is_last_rank geschützt.
4. gpu_model_runner: drafter=None auf Nicht-Letzt-Ranks (strukturell).
5. qwen3_5_mtp: Drafter-Forward verzweigt nach Input statt globalem PP-Rank.
6. Env: VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=0 (geteiltes Embedding liegt unter
   PP auf der falschen Stufe).

**Verbleibendes Arbeitspaket:** Draft-Token-Verteilung über die Pipeline. Im
Async-Modus hält der Fork Draft-Tokens GPU-seitig auf der letzten Stufe
(_draft_token_ids); vordere Stufen brauchen sie für die Verify-Inputs →
Broadcast implementieren, spiegelbildlich zum existierenden
_pp_broadcast_prev_sampled_token_ids (gleiche Datei, fertige Vorlage samt
Chunked-Prefill-Symmetrie). Sync-Scheduling als Abkürzung scheitert an einem
Kollektiv-Deadlock (RPC-Timeout sample_tokens). Serve-Script hat dafür bereits
den ASYNC_SCHED-Schalter.

## Plattform-Fixe (alle dokumentiert, Original-Repo unangetastet)

| Problem | Fix |
|---|---|
| apt-CUDA nur 12.0, pip-nvcc-Wheel ohne nvcc-Binary | NVIDIA-debs (nvcc/nvvm/crt 12.8.93) per `dpkg -x` ins Projekt entpackt, pip-Runtime-Header dazugelinkt → [run-bootstrap-mini.sh](../v100-skinny/run-bootstrap-mini.sh) |
| 1Cat parst `CUDA_VISIBLE_DEVICES` mit `int()` | numerische Indizes + `CUDA_DEVICE_ORDER=PCI_BUS_ID` |
| Serve-Script hart auf TP=4/512-Zensus | [serve-qwen38-mini.sh](../v100-skinny/scripts/serve-qwen38-mini.sh): TP/K/Backend/Gates parametrisiert |
| NCCL hängt im Init (P2P zwischen Ryzen-Root-Ports defekt) | `NCCL_P2P_DISABLE=1` |
| Custom-Allreduce hängt ebenso (CUDA-IPC-P2P) | `--disable-custom-all-reduce` |
| GPU-frei-Check bricht wegen produktivem VLM auf GPU 3 ab | Check auf `CUDA_VISIBLE_DEVICES` eingeschränkt |

Symptom beider P2P-Hänger: beide Ranks 100 % GPU-„Util" bei ~40 W und 0,5 GB VRAM (Spin-Wait).

## Nebenschauplatz: AIfred-venv aufgeräumt

vLLM 0.16.0 lag funktionsunfähig (torch fehlte) im Produktions-venv; zudem Whisper-Altlasten aus der Vor-Docker-Zeit. Entfernt: vllm, flashinfer, xgrammar, ray, cupy, cutlass-dsl, openai-whisper, faster-whisper, triton, ctranslate2 → venv 10 GB → 7,2 GB. STT läuft unverändert im Docker-Container `whisper-stt`.

## 5-GPU-Mischbetrieb: ~~geklärt (negativ für Single-Engine)~~ ÜBERHOLT

> **Korrektur 2026-08-25:** Der Befund „Fork: kein `SupportsPP` für qwen3_5 →
> PP unmöglich" gilt nicht mehr. Das Merge-Projekt hat PP und PP×TP im Fork
> nachgerüstet; das heterogene 2×2-Gitter (TP=2 RTX + TP=2 V100) läuft
> kohärent mit MTP k=7. Siehe Abschnitt „Sitzung 2026-08-25" unten und
> [MERGE-PROJECT-HANDOVER.md](../v100-skinny/MERGE-PROJECT-HANDOVER.md).
> Der ursprüngliche Text bleibt als Historie stehen:

- **Fork:** kein `SupportsPP` für qwen3_5 → PP unmöglich; TP=5 scheitert an 24 Heads/4 KV-Heads; TP=4 gemischt scheitert an der SM75-Prefill-Numerik. ❌
- **upstream vLLM 0.27:** Wheel enthält keine sm_70-Kernel („no kernel image is available") — V100 damit generell außen vor (das ist exakt die Lücke, die 1Cat-vLLM füllt). ❌
- **llama.cpp:** kann heterogen über alle 5 Karten (Layer-Split, produktiv im Einsatz bei der 122B-Config) — aber Kapazität, kein Single-Request-Tempo. ✅
- **Pragmatische Lösung: Zwei-Engine-Betrieb** — Fork auf 2 V100s (schnellster 27B-Endpoint) + upstream TP=2 auf 2 RTX (z. B. 122B-GPTQ), parallel. Abschluss-Demo geplant.

## Offene Stationen

- [x] 4c: Unsloth-all-NVFP4 auf V100 TP2 — erledigt, siehe Tabelle
- [x] PP=2 auf V100s — **nicht möglich** (Fork ohne SupportsPP)
- [x] vLLM TP=2 über beide RTX 8000 — erledigt (Zeilen 5a/5b)
- [ ] optional: llama.cpp `-sm row` auf 2× RTX (llama.cpps TP-Annäherung)
- [x] Qwen3.5-122B-A10B GPTQ vs. Q8 — erledigt (llama.cpp gewinnt, siehe 122B-Tabelle)
- [x] Zwei-Engine-Demo — gestrichen (User: kein Bedarf; zwei Engines können ohnehin kein gemeinsames Modell tragen)
- [x] Qualitätsvergleich — erledigt, siehe eigene Sektion
- [x] TabbyAPI/ExLlamaV3-Experiment — **definitiv unmöglich auf dieser Hardware:** Das Ampere-Gate in Tabby ist echt. Wheel ohne sm75-Kernel; Source-Build für SM75 scheitert hart an `cp.async`-Instruktionen (Ampere-Silizium-Feature, Fundament der EXL3-Kernel-Pipeline). Gilt für RTX 8000 (SM75) UND V100 (SM70). ExLlama-TP ist damit für diese Kiste vom Tisch; einziger Weg zum Single-Engine-Mischbetrieb bleibt das Merge-Projekt (PP×TP im Fork).
- [x] 122B-GPTQ gelöscht (Platten-Notstand 100 %; per snapshot_download jederzeit wiederholbar)
- [ ] Entscheidung: Tabby-Experiment-Reste löschen (~50 GB: tabby-bench-venv + EXL3 4bpw/6bpw)?

## Qualitätstests (2026-08-24, User-Auftrag)

Methodik: [quality.py](quality.py) — (1) AIME-Aufgabe bei temp 0 mit Thinking,
bekannte Lösung 70; (2) Diversität = unterschiedliche Antworten auf eine offene
Hardware-Frage über 20 Seeds bei temp 0.6. Diversität ist nur RELATIV zur
Q8-Referenz aussagekräftig (Kollaps-Detektor, vgl. v100-skinny-README).

| Konfiguration | math (temp 0) | Diversität |
|---|---|---|
| llama.cpp 27B **Q8** (Referenz) | PASS | 4/20 |
| v100-skinny 27B **NVFP4-mixed** (V100) | PASS | 3/20 |
| vLLM 27B **AWQ-INT4** (RTX) | PASS | 2/20 |
| llama.cpp 122B **Q8** (Referenz) | PASS | 8/20 |
| vLLM 122B **GPTQ-Int4** (RTX TP2) | PASS | 6/20 |

> **Einordnung 2026-08-25:** Dieses Instrument ist ein Kollaps-Detektor. Es trennt
> „lebt" von „kollabiert", kann aber nicht auflösen, ob 4 Bit gegenüber 8 Bit ein
> bisschen oder viel kostet. Das Urteil unten heißt korrekt gelesen: *dieses
> Messgerät* hat keinen Unterschied aufgelöst. Auflösendes Verfahren siehe
> [quality_kl.py](quality_kl.py) und Abschnitt „Sitzung 2026-08-25".

**Urteil: keine messbaren Qualitätsabschläge.** Alle Quants lösen die
math-Aufgabe korrekt, alle Diversitätswerte liegen im Band der jeweiligen
Q8-Referenz mit denselben Antwortfamilien — kein Kollaps-Muster. (122B-Q8
brauchte >6k Thinking-Tokens; erster „FAIL" war ein Token-Limit-Artefakt.)

## Nachtrag: Merge-Projekt gestartet (2026-08-24 abends)

PP=2 läuft im Fork (Trace-Fix), PP+MTP k=7 bootet mit allen Gates und
Runde 0 inkl. Drafting — Deadlock beim Eintritt in Runde 1 (GDN-Spec-Commit
über die PP-Grenze). PP=2 plain auf 2× V100: 32,3 tok/s (Kapazität, kein
Decode-Gewinn — wie erwartet). **Vollständige Übergabe mit allen Patches,
Repro-Kommando, Deadlock-Diagnose und Arbeitsplan:**
[MERGE-PROJECT-HANDOVER.md](../v100-skinny/MERGE-PROJECT-HANDOVER.md)

## Sitzung 2026-08-25: Formatvergleich, faire llama.cpp-Referenz, heterogenes Gitter

Alle Zahlen mit derselben [bench.py](bench.py)-Methodik wie oben (Single-Request-Decode,
ninfer-Profil, Thinking an). **Achtung beim Quervergleich:** die Formatmatrix lief mit
`--max-tokens 1024`, die Gitter-Messungen mit 2048. Die math-Zelle endet vorher von
selbst (~700–900 Tokens) und ist damit vergleichbar; die **code-Zelle läuft in beiden
Fällen ins Limit** und ist zwischen den beiden Blöcken NICHT direkt vergleichbar
(längere Generierung = wachsender KV-Cache = niedrigere Rate).

### Formatmatrix: Qwen3.8-27B, identische Hardware (2× V100, TP=2)

Derselbe Modellstand in drei Quantisierungen, gleiche Topologie, gleiche Seeds (1001/2002).

| Format | Bits/Gewicht | K=0 math | K=0 code | k=7 math | k=7 code | MTP-Akzeptanz |
|---|---|---:|---:|---:|---:|---:|
| **NVFP4** (RadixArk, ModelOpt-mixed) | ~4,5 | 40,5 | 40,4 | **88,2 ± 1,5** | **62,2 ± 1,9** | 39,5 % |
| **FP8** (Qwen offiziell, blockweise) | 8,0 | 14,8 | 14,8 | 44,8 ± 1,5 | 37,2 ± 1,6 | 41,5 % |
| **AWQ-INT4** (cyankiwi) | ~4,25 | 11,0 | 11,0 | 37,0 ± 3,2 | 30,0 ± 5,4 | 40,0 % |
| *llama.cpp Q8_K_XL, MTP n=3* | ~8,3 | — | — | *41,1 ± 0,3* | *31,5 ± 2,7* | — |
| *llama.cpp Q8_K_XL, MTP n=7* | ~8,3 | — | — | *40,8 ± 0,4* | *31,6 ± 0,1* | — |

Alle drei vLLM-Quants bestehen die Kohärenzprobe (17×23 = 391) und tragen MTP-Gewichte.
Die Akzeptanzraten liegen eng beieinander (39–42 %) — kein Ausreißer wie die früher als
„verdächtig hoch" notierten 98,4 %. NVFP4 reproduziert die 88,1 der Vorsitzung exakt.

**Befunde:**

1. **vLLM schlägt llama.cpp auf gleicher Bitbreite** — FP8 44,8/37,2 gegen Q8 41,1/31,5,
   also +9 % (math) und +18 % (code). Bescheiden, aber real, und llama.cpp bekam seine
   beste Konfiguration: **n=7 bringt dort nichts** (40,8/31,6), die Spekulationstiefe ist
   bei n=3 gesättigt. Damit ist der Vergleich erstmals Äpfel mit Äpfeln — die frühere
   Gegenüberstellung 52,2 (vLLM INT4, 1× RTX) gegen 31,5 (llama.cpp Q8, 1× RTX) mischte
   Bitbreite und Engine.
2. **NVFP4 ist auf Volta konkurrenzlos** — 88,2 math sind 2,1× der llama.cpp-Q8-Rate,
   aber eben bei 4 Bit. Der Vorsprung stammt aus den Skinny-Kernels des Forks.
3. **AWQ-INT4 fällt auf Volta durch** — 37,0/30,0, also *langsamer als llama.cpp mit
   der doppelten Bytemenge*. Grund ist Hardware: AWQ lebt von Turings INT4-Tensorcores,
   die Volta nicht besitzt. Auf der RTX lag AWQ in der Vorkampagne bei 52,2 (MTP k=3).
   **Konsequenz fürs heterogene Gitter:** beide Generationen bevorzugen verschiedene
   Formate, aber ein Checkpoint muss beide Stufen bedienen — die Formatwahl richtet sich
   nach der schwächeren Seite, und das ist bei AWQ die V100.
4. **MTP wirkt formatabhängig unterschiedlich stark:** NVFP4 ×2,2, AWQ ×3,4, FP8 ×3,0.
   Je langsamer der Grundpfad, desto mehr holt die Spekulation heraus.

### Dieselbe Matrix auf den beiden RTX 8000 (sm75, TP=2)

Gespiegelt zur V100-Matrix: gleiche Engine (Fork), gleiche Seeds, gleiche Zellen,
gleiche Token-Grenze. Einzige Variable ist die GPU-Generation. Treiber
[run-rtx-matrix.sh](run-rtx-matrix.sh), Rohdaten [rtx-matrix.jsonl](rtx-matrix.jsonl).

| Format | RTX K=0 | RTX k=7 | V100 K=0 | V100 k=7 | Akzeptanz RTX |
|---|---:|---:|---:|---:|---:|
| **NVFP4** | 43,2 | **79,1 ± 2,4** | 40,5 | **88,2 ± 1,5** | 36,6 % |
| **AWQ-INT4** | 12,2 | 38,5 ± 2,7 | 11,0 | 37,0 ± 3,2 | 45,1 % |
| *AWQ, upstream-vLLM (Vorkampagne, 2× RTX)* | *42,2* | *67,6 (k=3)* | — | — | — |

**Befunde:**

1. **AWQ hat auf dieser Maschine keine Nische.** Im Fork ist es auf beiden
   Generationen gleich langsam (38,5 vs. 37,0) — also keine Hardwaregrenze, sondern
   der Fork bedient den AWQ-Pfad nicht. Die frühere Erklärung „AWQ braucht Turings
   INT4-Tensorcores" erklärt nur die V100-Zahl; auf Turing sind sie vorhanden und
   werden trotzdem nicht genutzt. Upstream-vLLM holt auf denselben Karten 67,6 —
   aber **NVFP4 im Fork liefert dort 79,1**, also mehr. Weder Engine noch Generation
   ergeben eine Konstellation, in der AWQ vorne läge.
2. **Ohne MTP ist Turing vorn, mit MTP Volta.** 43,2 vs. 40,5 bei K=0, aber 79,1 vs.
   88,2 bei k=7. Der Spekulationsfaktor ist auf Volta höher (2,18 vs. 1,83), weil die
   V100-Stufe den XQA-Verifier-Pfad bekommt und die RTX-Stufe mit TRITON_ATTN
   verifiziert. Das erklärt, warum das gemischte 2×2 mit 85,0 zwischen beiden landet —
   und dass eine ungleiche Layer-Partition dort kein reines Bandbreiten-, sondern auch
   ein Verifier-Thema ist.
3. **Formatempfehlung für diese Kiste: NVFP4**, ohne Einschränkung. Zusammen mit der
   Verfügbarkeitslage (siehe unten) bleibt kein Argument für AWQ.

### Checkpoint-Verfügbarkeit nach Format (HF-Namenssuche, je 100 Treffer)

| Modell | NVFP4 | AWQ | FP8 | GPTQ/INT4 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B | **7** | 4 | 2 | 3 |
| Qwen3.5-122B-A10B | **15** | 3 | 6 | 4 |
| DeepSeek-V4-Flash | **10** | 1 | 3 | 1 |
| Qwen3.5-397B-A17B | **9** | 2 | 4 | 2 |
| MiniMax-M2.7 | **9** | 2 | 1 | 1 |

Entgegen der verbreiteten Annahme hat **NVFP4 für die aktuelle Modellgeneration die
breitere Auswahl**, teils um Faktor drei bis fünf. AWQ war historisch verbreiteter,
wurde aber überholt — vermutlich durch den Blackwell-Deployment-Druck. Damit entfällt
auch das Verfügbarkeitsargument für einen AWQ-Kernel auf Volta.
*(Einschränkung: Namenssuche, Checkpoints ohne Formatkürzel im Namen fehlen. Die
Richtung ist über fünf Modelle konsistent, die absoluten Zahlen weniger.)*

### Heterogenes PP×TP (Merge-Projekt, Qwen3.8-27B NVFP4)

| Topologie | K | math | code | Bemerkung |
|---|---|---:|---:|---|
| **2×2: TP=2 RTX (Stufe 0) + TP=2 V100 (Stufe 1)** | 7 | **85,0 ± 1,1** | **56,3 ± 1,1** | n=3, max 2048; alle elf Gates grün |
| heterogen TP=1 PP=2 (1× RTX + 1× V100) | 7 | 74,8 ± 0,9 | — | max 2048 |
| V100-PP2 (Regression gegen Vorsitzung) | 7 | 79,8 ± 1,2 | 55,8 ± 1,4 | unverändert |
| 2×2, prose 4096 Tokens | 7 | 78,2 | — | kohärent bis zum Schluss |

Zusätzlich verifiziert: 2731-Token-Prompt (Chunked Prefill) korrekt zusammengefasst,
1481-Token-Aufsatz ohne Drift.

### Checkpoint-Formate: warum FP8 nicht auf den schnellen Pfad kommt

Direkter Blick in die Safetensors-Header:

| Checkpoint | Skalen-Layout | Folge auf dieser Hardware |
|---|---|---|
| Qwen/Qwen3.8-27B-FP8 | `weight_scale_inv`, Form z. B. `[136, 40]` → **Raster aus 128×128-Blöcken** | landet in vLLMs generischer `Fp8Config` → `MarlinFP8ScaledMMLinearKernel` (Weight-only), 14,8 tok/s |
| RadixArk/Qwen3.8-27B-NVFP4 | `weight_scale`, Form `[]` → **ein Skalar pro Matrix** (+ `input_scale`, `weight_scale_2`) | trifft den QPN8/Skinny-Pfad des Forks, 40,5 tok/s |

Der QPN8-Kernel multipliziert am Ende mit genau einem Skalar. Ein blockskaliertes
Checkpoint-Layout zu bedienen hieße, im innersten Schleifenkörper der
`mma.sync.m8n8k4`-Kachelung ein Skalenraster mitzuführen — Kernel-Arbeit, keine
Dispatch-Änderung. **Es ist nicht modell-, sondern formatspezifisch:** Qwen und DeepSeek
skalieren blockweise, NVIDIAs ModelOpt per Tensor. Für Qwen3.8-27B existiert kein
ModelOpt-FP8 (die `nvidia/*`-Pfade liefern 401).

### DeepSeek-V4-Flash: warum das GGUF kleiner ist als NVFP4

| Variante | Größe |
|---|---|
| lokales GGUF UD-Q4_K_XL (Hauptmodell) | **145 GB** |
| + dspark-Draft-Modell Q8_0 (Spekulation) | 11 GB |
| RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 | 164 GB |
| nvidia/DeepSeek-V4-Flash-NVFP4 | 168 GB |
| canada-quant/…-NVFP4-FP8-MTP (mit MTP-Block) | 184 GB |

Der NVFP4-Checkpoint ist **nicht durchgängig 4 Bit**. Seine Config sagt
`quant_algo: MIXED_PRECISION`, `producer: modelopt/dsv4-nvfp4-experts`,
`quantized_layers: layers.N.ffn.experts` und `ignore: ['*.attn.*',
'*.ffn.shared_experts.*', 'head']` über einer Basis aus `quant_method: fp8` mit
`weight_block_size: [128,128]`. Also: **nur die gerouteten Experten sind NVFP4,
Attention, Shared Experts und Head bleiben blockweises FP8.** Im Mittel ~4,7 Bit
gegenüber ~4,1 Bit beim dynamischen GGUF — daher die 16 % Größendifferenz.

**Zwei Konsequenzen:** (a) Der Checkpoint passt nicht — 164 GB gegen ~169 GB nutzbar
über alle fünf Karten, ohne KV-Cache und nur wenn GPU 3 das VLM räumt. (b) Selbst wenn
er passte, liefen Attention und Shared Experts über den langsamen Marlin-Block-FP8-Pfad
— und die feuern bei einem A13B-MoE bei *jedem* Token, während von den gerouteten
Experten nur eine Handvoll aktiv ist. Der blockskalierte FP8-Kernel wäre also nicht nur
für Qwens FP8 nötig, sondern auch Voraussetzung für ein brauchbares DeepSeek-NVFP4.

### Qualitätsmessung: Instrument geschärft

[quality.py](quality.py) ist ein Kollaps-Detektor (eine Matheaufgabe, Diversität über
20 Seeds) — er kann „lebt" von „kollabiert" trennen, aber nicht auflösen, ob 4 Bit
gegenüber 8 Bit ein bisschen oder viel kostet. Sein Urteil „keine messbaren
Qualitätsabschläge" heißt daher: *dieses Messgerät* hat keinen Unterschied aufgelöst.

Neu: [quality_kl.py](quality_kl.py) misst den Abstand als Zahl. Teacher Forcing über
feste Texte (deutsche Prosa, Python-Code, Faktentext), per `prompt_logprobs` an jeder
Position die Verteilung abgelesen; Kennzahlen sind Top-1-Übereinstimmung und mittlere
KL-Divergenz gegen eine Referenz. Das löst mit wenigen hundert Token auf, wofür ein
Benchmark-Score Tausende Aufgaben bräuchte — der Grund, warum Herstellerangaben wie
„unter 1 % auf MMLU-Pro" am Rauschrand liegen.

**Technischer Befund dabei:** llama.cpps Server liefert keine Logprobs über den Prompt
(der OpenAI-Weg mit `echo=true, logprobs=5` gibt leere Listen zurück; das native
`/completion` mit `n_probs` deckt nur generierte Token ab). Der 8-Bit-Anker ist deshalb
der **FP8-Checkpoint in derselben Engine** — was den Vergleich sauberer macht, weil
Engine und Tokenizer konstant bleiben und nur die Quantisierung variiert.

**Ergebnis** (Referenz: FP8 in derselben Engine, 2× V100 TP=2, K=0; KL in nats,
D(FP8 ‖ Quant); Treiber [run-quality-matrix.sh](run-quality-matrix.sh), Rohdaten
[quality-kl.jsonl](quality-kl.jsonl)):

| Text (Positionen) | NVFP4 Top-1 | NVFP4 KL (Mittel / p95) | AWQ-INT4 Top-1 | AWQ-INT4 KL (Mittel / p95) |
|---|---:|---:|---:|---:|
| Deutsche Prosa (78) | 93,6 % | 0,0434 / 0,1098 | 92,3 % | 0,0295 / 0,1015 |
| Python-Code (88) | **100 %** | 0,0028 / 0,0176 | **100 %** | 0,0018 / 0,0078 |
| Faktentext (47) | 91,5 % | 0,0182 / 0,0477 | 91,5 % | 0,0141 / 0,0463 |

**Befunde:**

1. **4 Bit ist messbar anders als 8 Bit, aber der Abstand ist klein.** Ein mittleres
   KL von 0,02–0,04 nats entspricht grob 2–4 % Perplexitäts-Äquivalent. Das ist kein
   Kollaps und rechtfertigt die Sorge „4 Bit fällt deutlich ab" nicht — es ist aber
   auch nicht „verlustfrei", wie die Herstellerkommunikation zu NVFP4 nahelegt.
2. **Der Abstand hängt stark vom Textcharakter ab.** Bei Code stimmen *alle* Top-1-Wahlen
   überein (KL 0,002–0,003): niedrige Entropie, die Fortsetzung ist ohnehin determiniert.
   Bei Prosa und Faktentext weichen 6–9 % der Positionen ab — dort, wo mehrere
   Fortsetzungen legitim sind.
3. **AWQ-INT4 liegt konsistent näher an FP8 als NVFP4** — auf allen drei Texten, sowohl
   im Mittel als auch im p95. Das bestätigt unabhängig, was die Literatur berichtet
   (KLD 0,024 gegen 0,035): INT4 verteilt 16 gleichmäßige Stufen, NVFP4 hat durch das
   Float-Format nur 8 Beträge. NVFP4s Vorteil ist Tempo auf Blackwell, nicht Genauigkeit
   — und auf Volta ist es umgekehrt (NVFP4 88,2 gegen AWQ 37,0 tok/s).

**Grenzen dieser Messung, ausdrücklich:** 213 Positionen sind eine kleine Stichprobe;
die Richtung ist über alle drei Texte konsistent, die absoluten Werte sind es weniger.
Die Referenz ist FP8, nicht BF16 — gemessen wird „Abstand zu 8 Bit", was der Vorgabe
entspricht, aber nicht „Abstand zur Wahrheit". Und eine abweichende Top-1-Wahl ist
nicht automatisch ein Fehler: wo die Verteilung flach ist, ist sie ein Münzwurf.

**Konsequenz für den blockskalierten FP8-Kernel:** Der Aufwand (mehrtägige CUDA-Arbeit
im `mma.sync`-Kern) kaufte gegenüber NVFP4 einen Abstandsgewinn von ~0,02–0,04 nats und
kostete rund die Hälfte der Decode-Rate (88,2 → geschätzt 60–75). Nach dieser Datenlage
lohnt er sich **nicht**. Er würde erst dann wieder interessant, wenn eine
aufgabenbezogene Messung zeigte, dass dieses KL praktisch spürbar ist, oder wenn ein
Zielmodell nur als blockskalierter FP8-/NVFP4-Mixed-Checkpoint verfügbar ist
(DeepSeek-V4-Flash ist genau so ein Fall — siehe oben).


## Rohdaten

`bench-*.jsonl` in diesem Verzeichnis; Server-Logs unter `/home/mp/Projekte/v100-skinny/serve*.log`.
Sitzung 2026-08-25: [format-matrix.jsonl](format-matrix.jsonl) (Treiber:
[run-format-matrix.sh](run-format-matrix.sh)), llama.cpp-Referenz in
`llamacpp-q8-v100*.log`, Gitter-Logs unter `/home/mp/Projekte/v100-skinny/s4-*.log`.


## Sitzung 2026-08-25 (nachmittags): DeepSeek-V4-Flash — llama.cpp-Latte

Gleiche bench.py-Methodik (ninfer, max 4096), Zellen math+code, Seeds
1001/2002, n=2. Produktiv-Config aus llama-swap
(`DeepSeek-V4-Flash-0731-UD-Q4_K_XL`, 5 GPUs Layer-Split 11,12,8,7,5,
dspark-Draft n=5 auf CUDA4):

| Engine · Format | math | code |
|---|---:|---:|
| llama.cpp b10562 · UD-Q4_K_XL + dspark | 40,4 ± 1,1 | 42,8 ± 0,4 |

Das ist die Latte für den vLLM-DeepSeek-Test (RedHatAI NVFP4-FP8,
compressed-tensors: Attention block-FP8 → QPN8-blk, FFN NVFP4).

### PP=5 über alle 5 Karten (27B, Topologie-Nachweis für DeepSeek)

TP=1 PP=5 (RTX,RTX,V100,V100,V100; Partition 16,16,11,11,10), K=7, Graphs,
alle elf Gates grün, bench.py Seeds 1001/2002:

| Topologie | math | code |
|---|---:|---:|
| 2×2-Gitter (Referenz, Session 4) | 85,0 ± 1,1 | 56,3 ± 1,1 |
| PP=5 | 66,8 ± 2,8 | 42,9 ± 2,2 |

PP=5 kostet ~21–24 % gegen das 2×2 (fünf Pipeline-Hops, kein TP innerhalb
der Stufen) — funktioniert aber kohärent. Für DeepSeek (153 GB) ist PP=5
die einzige Topologie, die die Kapazität aller 5 Karten nutzt.
