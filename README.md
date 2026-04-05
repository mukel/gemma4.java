# Gemma4.java

Fast [Gemma 4](https://ai.google.dev/gemma) inference (dense and MoE) implemented in pure Java.

## Features

 - Single file, no dependencies
 - [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) parser
 - Gemma 4 tokenizer
 - Support all Gemma 4 models: `E4B`, `E2B`, `31B` and `26B-A4B` (MoE), 
 - **Mixture of Experts (MoE) support** with expert routing
 - Sliding Window Attention (SWA) with full attention layers
 - Per-layer KV cache sharing
 - Per-head Q/K RMS normalization
 - Support `F16`, `BF16`, `F32` weights + `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` quantizations
 - Fast matrix-vector multiplication routines using Java's [Vector API](https://openjdk.org/jeps/469)
 - Simple CLI with `--chat` and `--instruct` modes
 - GraalVM's Native Image support
 - AOT model pre-loading for instant time-to-first-token

## Setup

Download GGUF quantized models from HuggingFace:

| Model | Architecture | GGUF Repository |
|-------|-------------|-----------------|
| E2B | Dense, ~5B total params | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) |
| E4B | Dense, ~8B total params | [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) |
| 31B | Dense | [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF) |
| 26B-A4B | Mixture of Experts (MoE) | [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |

#### Optional: Pure quantizations

In the wild, `Q8_0` quantizations are fine, but `Q4_0` quantizations are rarely pure e.g. the `token_embd.weights`/`output.weights` tensor are quantized with `Q6_K`, instead of `Q4_0`.  
A **pure** quantization, although not required, can be generated from a high precision (F32, F16, BF16) .gguf source 
with the `llama-quantize` utility from [llama.cpp](https://github.com/ggml-org/llama.cpp) as follows (pass `--pure` to avoid mixing quantizations):

```bash
./llama-quantize --pure ./gemma-4-E2B-it-BF16.gguf ./gemma-4-E2B-it-Q4_0.gguf Q4_0
```

Pick any of the supported quantizations: `Q4_0`, `Q4_1`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`.


## Build and run

Java 21+ is required, in particular for the [`MemorySegment` mmap-ing feature](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/nio/channels/FileChannel.html#map(java.nio.channels.FileChannel.MapMode,long,long,java.lang.foreign.Arena)).

[`jbang`](https://www.jbang.dev/) is a perfect fit for this use case, just:
```
jbang Gemma4.java --help
```

Or execute directly, also via [`jbang`](https://www.jbang.dev/):
```bash 
chmod +x Gemma4.java
./Gemma4.java --help
```

#### Optional: Makefile

A simple [Makefile](./Makefile) is provided, run `make jar` to produce `gemma4.jar`.

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

`Gemma4.java` supports AOT model preloading, enabling **0-overhead, instant inference, with minimal TTFT (time-to-first-token)**.

To AOT pre-load a GGUF model:
```bash
PRELOAD_GGUF=/path/to/model.gguf make native
```

A specialized, larger binary will be generated, with no parsing overhead for that particular model.
It can still run other models, although incurring the usual parsing overhead.

## Performance

[GraalVM 25+](https://www.graalvm.org/downloads) is recommended for the absolute best performance, it provides partial, but good support for the [Vector API](https://openjdk.org/jeps/469).

By default, the "preferred" vector size is used, it can be force-set with `-Dgemma4.VectorBitSize=0|128|256|512`, `0` means disabled.

## License

Apache 2.0
