# XSGLang

XSGLang is a CUDA inference engine built for block-wise generation, branching continuations, activation hooks, and other workflows that are awkward or inefficient in a normal inference stack.

Standard inference engines are often fast, but they expose generation as a black box. You submit a prompt, ask for tokens, and get a completion back. That works for serving, but it becomes restrictive once you want to stop partway through generation, inspect the model state, branch from the current point, or run many related continuations without repeatedly rebuilding the same prefix.

At the other extreme, `transformers.generate()` gives more direct access, but in practice it is slow and awkward for this kind of work. If the goal is to run search, compare branches, inspect top-k behavior, read auxiliary heads, or intervene on activations, it is easy to end up with a Python control loop that throws away most of the performance of the underlying model.

XSGLang is meant to sit in that gap. It keeps the model resident, exposes generation as bounded blocks, and makes continuation control a first-class part of the runtime rather than something wrapped around it from the outside.

## Block-wise generation

The central idea in XSGLang is that generation does not have to be treated as one opaque stream.

Instead of only asking the model to generate until it finishes, you can run a bounded block such as 20, 30, or 40 tokens, then inspect the result before deciding what to do next. That result can include emitted text, top-k ids and logprobs, and model-side outputs such as a value head when the model provides one.

From there you can continue the same state, fork it into children, or apply a different control policy to the next block.

This makes it possible to do things like:

* generate a short block
* inspect the model's local behavior
* branch into several child continuations
* keep all of them warm and ready for the next step
* reuse KV state efficiently instead of restarting each branch from scratch

That is the main use case for this repository.

## What XSGLang is for

XSGLang is built for inference workloads where control matters as much as raw throughput. That includes:

* branching generation and search
* activation capture and intervention
* hook-based experiments
* auxiliary output heads such as value or reward heads
* repeated experiments over a shared prompt prefix
* data generation workflows where many related continuations need to stay live at once

The point is not to wrap an existing server with more Python. The point is to make these workflows part of the engine itself.

## What it is built on

XSGLang is a fork of [Mini-SGLang](https://github.com/sgl-project/mini-sglang), and the underlying serving foundation comes from that project.

Mini-SGLang provides the compact runtime base this fork builds on, including:

* paged KV cache
* prefix reuse
* chunked prefill
* overlap scheduling
* tensor parallelism
* CUDA graph decode
* a codebase that is small enough to read without getting lost immediately

XSGLang keeps that foundation and extends the runtime control surface on top of it.

## What is different here

This repository is centered on the engine rather than on a collection of apps.

The main additions in this fork are around controllable inference:

* bounded block execution
* continuation-oriented offline control
* branching from shared live state
* efficient KV-backed continuation reuse
* hook-aware execution for capture and intervention
* support for auxiliary outputs during generation
* research-oriented workflows that need direct runtime control

The model stays loaded. The continuation stays live. You can advance it a block at a time, inspect it, fork it, and continue from there.

## Repository layout

Main runtime code:

* [`python/minisgl/engine`](./python/minisgl/engine)
* [`python/minisgl/scheduler`](./python/minisgl/scheduler)
* [`python/minisgl/models`](./python/minisgl/models)
* [`python/minisgl/kvcache`](./python/minisgl/kvcache)
* [`python/minisgl/attention`](./python/minisgl/attention)
* [`python/minisgl/server`](./python/minisgl/server)
* [`python/minisgl/llm`](./python/minisgl/llm)
* [`python/minisgl/research`](./python/minisgl/research)

Tests:

* [`tests/core`](./tests/core)
* [`tests/misc`](./tests/misc)
* [`tests/kernel`](./tests/kernel)

Additional documentation:

* [`docs/features.md`](./docs/features.md)
* [`docs/structures.md`](./docs/structures.md)
* [`README_FULL_STRUCTURE.md`](./README_FULL_STRUCTURE.md)

## Installation

Requirements:

* Linux
* Python 3.10+
* NVIDIA GPU with CUDA available to PyTorch

Suggested setup:

```bash
git clone https://github.com/Pi-Willie/XSGLang.git
cd XSGLang
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
```

## Quick start

Run the server:

```bash
python -m minisgl --model Qwen/Qwen3-0.6B
```

Run the interactive shell:

```bash
python -m minisgl --model Qwen/Qwen3-0.6B --shell
```

There are also example scripts in the repository that show block-wise generation, continuation control, branching, and output inspection.

## Testing

Run the main Python tests with:

```bash
PYTHONPATH=python pytest tests/core tests/misc
```

Some kernel tests depend on extra local CUDA or JIT pieces and may not pass in a minimal environment. That is separate from the main Python and runtime test coverage.
