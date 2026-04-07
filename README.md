# Gemma4.java

<p align="center">
  <img src="https://ai.google.dev/static/gemma/images/gemma4_banner.png">
</p>

<div align="center">

![Java 21+](https://img.shields.io/badge/Java-21%2B-007396?logo=java&logoColor=white)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

Fast, zero-dependency, inference engine for [Gemma 4](https://ai.google.dev/gemma) in pure Java.

</div>

----

## Features

- Single file, no dependencies
- [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) parser
- Gemma 4 tokenizer
- Supports all Gemma 4 model families: `E2B`, `E4B`, `31B`, and `26B-A4B` (MoE)
- Mixture of Experts routing and execution
- Sliding Window Attention (SWA) and full-attention layers
- Per-layer KV cache sharing and per-head Q/K RMS normalization
- Supported dtypes/quantizations: `F16`, `BF16`, `F32`, `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`
- Thinking mode control with `--think off|on|inline`
- Matrix-vector kernels using Java's [Vector API](https://openjdk.org/jeps/469)
- CLI with `--chat` and `--instruct` modes
- GraalVM Native Image support
- AOT model preloading for lower time-to-first-token

## Setup

Download GGUF models from Hugging Face:

| Model | Architecture | GGUF Repository |
|-------|-------------|-----------------|
| E2B | Dense, ~5B total params | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) |
| E4B | Dense, ~8B total params | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| 31B | Dense | [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) |
| 26B-A4B | Mixture of Experts (MoE) | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |

#### Optional: pure quantizations

`Q4_0` files are often mixed-quant in practice (for example, `token_embd.weight` and `output.weight` may use `Q6_K`).
A pure quantization is not required, but can be generated from an F32/F16/BF16 GGUF source with `llama-quantize` from [llama.cpp](https://github.com/ggml-org/llama.cpp):

```bash
./llama-quantize --pure ./gemma-4-E2B-it-BF16.gguf ./gemma-4-E2B-it-Q4_0.gguf Q4_0
```

Pick any supported target quantization, for example `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, or `Q8_0`.


## Build and run

Java 21+ is required, in particular for the [`MemorySegment` mmap-ing feature](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.Arena)).

[`jbang`](https://www.jbang.dev/) is a good fit for this use case:
```
jbang Gemma4.java --help
jbang Gemma4.java --model ./gemma-4-E2B-it-Q4_0.gguf --chat
jbang Gemma4.java --model ./gemma-4-E2B-it-Q4_0.gguf --prompt "Explain quantum computing like I'm five"
```

Or run it directly (still via [`jbang`](https://www.jbang.dev/)):
```bash 
chmod +x Gemma4.java
./Gemma4.java --help
```

#### Optional: Makefile

A simple [Makefile](./Makefile) is provided. Run `make jar` to produce `gemma4.jar`.

Run the resulting `gemma4.jar` as follows: 
```bash
java --enable-preview --add-modules jdk.incubator.vector -jar gemma4.jar --help
```

### GraalVM Native Image

Compile with `make native` to produce a `gemma4` executable, then:

```bash
./gemma4 --model ./gemma-4-E2B-it-Q4_0.gguf --chat
```

### AOT model preloading

`Gemma4.java` supports AOT model preloading to reduce parse overhead and time-to-first-token (TTFT).

To AOT pre-load a GGUF model:
```bash
PRELOAD_GGUF=/path/to/model.gguf make native
```

A larger specialized binary is generated with parse overhead removed for that specific model.
It can still run other models with normal parsing behavior.

## Benchmarks

<p align="center">
  <img src="https://github.com/user-attachments/assets/8554de11-c028-4ff4-88b7-b5c0665423da">
</p>

\*\**Hardware specs: AMD Ryzen 9950X 16C/32T 64GB (6400) Linux 6.18.12.*

[GraalVM 25+](https://www.graalvm.org/downloads) is recommended for the absolute best performance (JIT mode), it provides partial, but good support for the [Vector API](https://openjdk.org/jeps/469), also in Native Image.

By default, the "preferred" vector size is used, it can be force-set with `-Dllama.VectorBitSize=0|128|256|512`, `0` means disabled.

## License

Apache 2.0
