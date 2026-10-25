# vllm-bench

[Deutsch](README.de.md)

Measuring tools and results of a benchmark campaign on one machine with older NVIDIA cards
(2× Quadro RTX 8000, sm75, and 3× Tesla V100, sm70): Qwen3.8-27B across engines (vLLM, llama.cpp),
quantisations (NVFP4, AWQ-INT4, FP8, GGUF Q8) and speculative decoding (MTP), August 2026. The
question was whether vLLM on these cards beats llama.cpp with MTP.

The results, with method and hardware, are in [RESULTS.md](RESULTS.md) (German). The tools are
still used for later measurements on the same machine.

## Tools

- **`bench.py`**: single-request decode benchmark against an OpenAI-compatible endpoint. Streams
  the answer and keeps the time to the first token apart from decoding:
  decode tok/s = (completion tokens − 1) / (time of the last token − time of the first). Three
  prompts (math, code, prose), seeds, sampling profile `ninfer` (temperature 0.6, top-p 0.95,
  top-k 20, presence 1.0, thinking on) or `greedy`.

  ```bash
  python3 bench.py --url http://127.0.0.1:8000 --model NAME --label run1 [--seeds 1001 2002 3003] [--max-tokens 4096]
  ```

- **`quality.py`**: a collapse detector, pass or fail: a math task with a known answer (greedy,
  thinking on) and the diversity of 20 answers to an open question (seeds 1–20); a broken
  quantisation collapses to a few repeats.
- **`quality_kl.py`**: how far a quantisation is from a reference, as a number. Both models get the
  same fixed text (teacher forcing, no generation); at every position it compares the distribution
  for the next token: share of identical top-1 choices and mean KL divergence over the top-k.
  `--dump` writes a reference, `--against` compares with one. vLLM only (llama.cpp's server gives
  no logprobs over the prompt).
- **`run-format-matrix.sh`**, **`run-quality-matrix.sh`**, **`run-rtx-matrix.sh`**: the series of
  the campaign. They start the servers themselves and use paths of the machine they ran on
  (the vLLM fork in `v100-skinny`, the Hugging Face cache); on another machine, adapt them first.

The `*.jsonl` files are the raw results of the runs, the `*.log` files the server logs that
belong to them.

## Star History

![Star History](https://raw.githubusercontent.com/Peuqui/Traffic/main/data/vllm-bench/star-history.svg)

<sub>Recorded daily in [Peuqui/Traffic](https://github.com/Peuqui/Traffic), together with views and clones. GitHub restricted the stargazer API to repo admins on 2026-06-30, so external chart services now need a token with write access.</sub>

## License

[MIT](LICENSE)
