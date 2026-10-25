# vllm-bench

[English](README.md)

Messwerkzeuge und Ergebnisse einer Benchmark-Kampagne auf einem Rechner mit älteren NVIDIA-Karten
(2× Quadro RTX 8000, sm75, und 3× Tesla V100, sm70): Qwen3.8-27B über Engines (vLLM, llama.cpp),
Quantisierungen (NVFP4, AWQ-INT4, FP8, GGUF Q8) und spekulatives Dekodieren (MTP), August 2026.
Die Frage war, ob vLLM auf diesen Karten llama.cpp mit MTP schlägt.

Die Ergebnisse mit Methode und Hardware stehen in [RESULTS.md](RESULTS.md). Die Werkzeuge werden
für spätere Messungen auf demselben Rechner weiter benutzt.

## Werkzeuge

- **`bench.py`**: Decode-Benchmark mit einer Anfrage gegen einen OpenAI-kompatiblen Endpunkt.
  Streamt die Antwort und trennt die Zeit bis zum ersten Token vom Dekodieren:
  Decode tok/s = (Antwort-Tokens − 1) / (Zeit des letzten Tokens − Zeit des ersten). Drei Prompts
  (Mathe, Code, Prosa), Seeds, Sampling-Profil `ninfer` (Temperatur 0,6, top-p 0,95, top-k 20,
  presence 1,0, Thinking an) oder `greedy`.

  ```bash
  python3 bench.py --url http://127.0.0.1:8000 --model NAME --label lauf1 [--seeds 1001 2002 3003] [--max-tokens 4096]
  ```

- **`quality.py`**: ein Kollaps-Detektor, bestanden oder durchgefallen: eine Mathe-Aufgabe mit
  bekannter Lösung (greedy, Thinking an) und die Vielfalt von 20 Antworten auf eine offene Frage
  (Seeds 1–20); eine beschädigte Quantisierung fällt auf wenige Wiederholungen zusammen.
- **`quality_kl.py`**: wie weit eine Quantisierung von einer Referenz abweicht, als Zahl. Beide
  Modelle bekommen denselben festen Text (Teacher Forcing, kein Generieren); an jeder Stelle wird
  die Verteilung für das nächste Token verglichen: Anteil gleicher Top-1-Wahl und mittlere
  KL-Divergenz über die Top-k. `--dump` schreibt eine Referenz, `--against` vergleicht damit. Nur
  vLLM (der Server von llama.cpp liefert keine Logprobs über den Prompt).
- **`run-format-matrix.sh`**, **`run-quality-matrix.sh`**, **`run-rtx-matrix.sh`**: die Serien der
  Kampagne. Sie starten die Server selbst und benutzen Pfade des Rechners, auf dem sie liefen (den
  vLLM-Fork in `v100-skinny`, den Hugging-Face-Cache); auf einem anderen Rechner erst anpassen.

Die `*.jsonl`-Dateien sind die Rohergebnisse der Läufe, die `*.log`-Dateien die zugehörigen
Server-Logs.

## Star History

![Star History](https://raw.githubusercontent.com/Peuqui/Traffic/main/data/vllm-bench/star-history.svg)

<sub>Täglich erhoben in [Peuqui/Traffic](https://github.com/Peuqui/Traffic), zusammen mit Aufrufen und Klonen. GitHub hat die Stargazer-API am 30.06.2026 auf Repo-Admins beschränkt, deshalb brauchen externe Diagramm-Dienste jetzt ein Token mit Schreibrechten.</sub>

## Lizenz

[MIT](LICENSE)
