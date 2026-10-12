# Gemma-4 Kernel Optimization Competition on Furiosa RNGD

This repository is the competition skeleton code for optimizing Gemma-4-12B-it on Furiosa
RNGD. Stage 1 focused on three decoder-layer kernels; Stage 2 focuses on end-to-end (E2E)
performance. Competitors modify the allowed implementation while preserving the public
interface and the model's output quality.

## What you are optimizing

The three Stage 1 kernels are declared in `src/ops.rs`:

| Kernel | Operation |
|---|---|
| `ops::sliding_project_qkv` | RMSNorm, Q/K/V projection, Q/K RMSNorm, RoPE, and K/V ring-cache writes |
| `ops::sliding_attention_output` | Head broadcast, O projection, post-attention RMSNorm, and residual add |
| `ops::decoder_feedforward` | RMSNorm, GeGLU MLP, post-FF RMSNorm, residual add, and layer gate |

The kernels use the model's existing quantized weights and tensor layouts. Each kernel is
measured across 3 independently-seeded input configurations for the Stage 1 test, not a
single fixed invocation.

Stage 2 covers the complete E2E model-serving path except for the public API endpoint in
`src/api/`. This includes model execution, host orchestration, tokenization, image and
audio preprocessing, and runtime integration.

## Competition format

### Stage 1 — Kernel optimization

The submission period for Stage 1 has ended.

### Stage 2 — End-to-end optimization

Stage 2 is for optimizing the complete Gemma-4-12B-it E2E path, excluding the public API
endpoint in `src/api/`. The harness supplies iteration scheduling and a bounded KV pool;
participants can implement batch calculation through `host::model::Model::forward` and its
device kernels. See [Serving](SERVING.md#batched-serving) for the contract.

The Stage 2 evaluation sends text, image, and audio requests to the server built from your
code and checks two things:

1. **Correctness:** every response meets the threshold of its suite, listed under
   "Grading criteria" below.
2. **Performance:** the server's throughput on a stream of requests that mixes the input
   types. If its TTFT p90 or TPOT p90 exceeds the limit, the submission receives no score.

Values to be finalized:

- **TBD:** submission deadline and tie-break rule.

## Grading criteria

The competition uses one grading policy across both stages. Correctness is a hard gate: a
submission that fails a required correctness check receives no score, even if it is faster.

Every case must meet the threshold of its suite:

| Suite | Metric | Pass | Graded cases |
|---|---|---:|---:|
| `mmlu` | exact answer letter | `1.0` | 4 |
| `gsm8k` | numeric final answer | `1.0` | 2 |
| `ifeval` | every instruction satisfied | `1.0` | 1 |
| `vqav2` | VQA accuracy | `1.0` | 5 |
| `fleurs_en` | word error rate | `≤ 0.2` | 2 |
| `fleurs_ko` | character error rate | `≤ 0.2` | 2 |

Prompts and scoring are adapted from the source datasets; see
[public-test/README.md](public-test/README.md#how-the-cases-differ-from-the-source-datasets).
The server's sampling temperature is fixed at 0 (greedy decoding) so that it generates a
consistent answer for every case; we recommend keeping it.

Submissions are graded by measured throughput (generated tokens/s) on a stream of mixed
request types, subject to two latency limits:

| Limit | Value |
|---|---:|
| TTFT p90 | `≤ 4 s` |
| TPOT p90 | `≤ 100 ms` |

A submission whose TTFT p90 or TPOT p90 exceeds its limit receives no score.

## Rules

1. **Only permitted implementation changes are graded.** Optimizing the E2E path,
   excluding the public API endpoint in `src/api/`, is permitted. `src/api/` is replaced
   by the organizers' copy at grading, so the server must keep the interfaces it depends
   on. Only `src/` is uploaded, so dependencies and the build itself cannot change. If you
   need a change in `src/api/`, ask the organizers: a change that is accepted is applied
   to the baseline for every team.
2. **Serve the provided model.** The server must run Gemma-4-12B-it from the provided
   weights. Replacing the weights with another model's is prohibited.
3. **No input-specific optimization.** Optimizations must be general. They must not target
   specific inputs, such as the benchmark prompts, media, or datasets.
4. **No reuse of computation across requests.** Each request must compute its own KV cache
   and image or audio encoder outputs from its own prompt and media. Reusing results from
   another request, such as through prefix caching or cached encoder outputs, is
   prohibited.
5. **No test-specific behavior.** Performance requests set `ignore_eos`, which makes the
   server generate exactly `max_completion_tokens` tokens. Keep that behavior, and do not use
   the flag, or any other way of telling a performance request from an accuracy request,
   to change how a request is computed.
6. **Fair play.** The rules above do not cover every case. Where a submission raises a
   fairness concern, the organizers will decide how it is handled.

Value to be finalized:

- **TBD:** the boundary of permitted kernel implementation techniques, and the
  reproducibility requirements that apply to them.

Submissions are reviewed, and a submission that violates these rules is disqualified.

## Evaluation commands

### Test on the public cases

```sh
./scripts/stage2/accuracy_test.sh --test mmlu
./scripts/stage2/performance_test.sh --concurrency 4
./scripts/stage2/performance_test.sh --test 256 --concurrency 4
```

These build the model server, submit it through the Arena server, and test it with public
inputs. `accuracy_test.sh` runs a case from `public-test/`, which holds one example per suite,
and reports whether the responses pass. `performance_test.sh` reports TTFT, latency, TPOT,
and throughput.

| Script | `--test` |
|---|---|
| `accuracy_test.sh` | `mmlu`, `gsm8k`, `ifeval`, `vqav2`, `fleurs_en`, `fleurs_ko` |
| `performance_test.sh` | optional: `256`, `1024`, `2048` (text prompt tokens), `image`, `audio` |

Without `--test`, `performance_test.sh` mixes the five inputs into one stream, taking them
in turn (256, 1024, 2048, image, audio). With `--test`, it sends that input alone.
Performance requests set `ignore_eos`, as in grading, so every response has 64 tokens.

`--concurrency N` (default 1) keeps N requests in flight and sends `2 × N` requests, so
that later requests are admitted while others are still running, as in grading. Unlike
grading, there is no warmup. Use `--no-build` to reuse the latest binary or `--no-wait` to
submit without waiting for the result.

**Each test job is limited to 70 seconds**, including server startup. A test that does not
finish within this limit reports no result. The prefill stage was outside the scope of
Stage 1 and remains unoptimized, so optimizing it is recommended as the first step.

### Submit for grading

```sh
cargo binstall moa-submitter-cli   # run again to update an installed version for Stage 2

moa-submitter login
moa-submitter submit         # run from the repository root; auto-detects furiosa-opt-std
moa-submitter submit --concurrency 4   # concurrent requests, default 1
moa-submitter status         # the state of every submission you have made
moa-submitter status <id>    # the results and score of one submission
moa-submitter log <id>       # the complete log, stage by stage
```

`submit` uploads `src/` only, without `src/api/`. Keep the original directory structure.
Run it from the repository root, or pass `--source path/to/furiosa-opt-gemma4-12B`.

Each team may submit up to **4 times per 24 hours**. This limit may be adjusted during the
competition.

A submission is built into a server and evaluated with AI-BMT in two Arena jobs. Both use
the `--concurrency` you chose, and each is limited to 5 minutes including server startup.

- **Accuracy:** all 16 private cases must pass within the 5 minutes. At a concurrency of
  16 or more, the cases are sent `⌊concurrency / 16⌋ + 1` times, and every case must pass
  every time. If accuracy fails, the performance results are still reported.
- **Performance:** the server is measured on one stream of requests that mixes five input
  types: text prompts of 256, 1024, and 2048 tokens (Python code from CrossCodeEval), image
  prompts of 512 tokens (VQAv2), and audio prompts of 768 tokens (FLEURS English and Korean).
  Requests are sent with `concurrency` of them in flight at a time: a new one starts as soon
  as one finishes. Every request sets `ignore_eos` and `max_completion_tokens: 64`, so each
  response has 64 tokens.
  - **Warmup:** `concurrency` requests, taking the five types in turn. They are not
    measured, and the measurement starts after all of them have finished.
  - **Measurement:** `3 × concurrency` requests per type, `15 × concurrency` in total, with
    the five types mixed in a fixed order that is the same for every submission.
  - **Score:** the generated tokens/s over the measured requests. A submission receives no
    score if its TTFT p90 exceeds 4 s or its TPOT p90 exceeds 100 ms over the measured
    requests, or if the job does not finish within the 5 minutes.

A higher concurrency leaves more room to raise throughput, but it also makes the TTFT and
TPOT limits harder to meet.

## Toolchain

The supported development environment is x86_64 Ubuntu 22.04 or newer with GLIBC 2.34 or
newer:

```sh
sudo apt install build-essential libclang-dev
sudo apt install gcc-aarch64-linux-gnu

rustup toolchain install nightly-2026-05-01
cargo +nightly-2026-05-01 install cargo-binstall
cargo +nightly-2026-05-01 binstall cargo-furiosa-opt@0.8.1
cargo install furiosa-schedule-viewer
```

### Furiosa Arena CLI

Configure the Furiosa Arena CLI once before using the test scripts:

```sh
cargo binstall furiosa-arena-cli

furiosa-arena login
```

The cli commands used for troubleshooting are:

| Command | Purpose |
|---|---|
| `furiosa-arena submit <file>` | Submit a script or binary |
| `furiosa-arena status <id>` | Check job state |
| `furiosa-arena logs <id> --follow` | Stream job output |
| `furiosa-arena list` | List your jobs |
| `furiosa-arena cancel <id>` | Cancel a queued or running job |

### Baseline source code per `furiosa-opt-std` version

The baseline is published for each supported `furiosa-opt-std` version. Each version's
baseline code is available under the matching git tag:

| `furiosa-opt-std` | Tag | Checkout |
|---|---|---|
| 0.8.1 | `v0.8.1` | `git checkout v0.8.1` |
| 0.9.0 | `v0.9.0` | `git checkout v0.9.0` |

`main` tracks the latest supported version.

## Source code sharing and competition fairness

Publicly sharing competition source code before the official deadline is strictly
prohibited. If publicly shared code is copied or plagiarized by another participant, both
the code provider and the user will be immediately disqualified from receiving any awards.
If you currently have any competition code publicly accessible, please set it to private or
remove it immediately. Thank you for your active cooperation in maintaining a fair
competition environment.

## References

- [Programming Tensor Contraction Processors](https://developer.furiosa.ai/furiosa-opt/book)
  — mappings, movement, computation, scheduling, and tuning.
- [`furiosa_opt_std` API docs](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/)
  — tensor types, mapping expressions, and engine modules.
- [furiosa-arena-cli docs](https://github.com/kreatinj/furiosa-arena-cli#installation)
- [moa-submitter-cli](https://github.com/micro2026-moa/moa-submitter-cli) — the submission CLI.
- [OPTIMIZATION.md](OPTIMIZATION.md) — the Stage 1 kernel optimization workflow.
- [ARCHITECTURE.md](ARCHITECTURE.md) — repository layout and host/RNGD split.
- [SERVING.md](SERVING.md) — running the model as an HTTP server.
