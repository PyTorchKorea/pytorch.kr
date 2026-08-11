---
layout: blog_detail
title: "ExecuTorch에서 Muse Glimmer로 구현하는 빠른 온디바이스 에이전틱 AI"
author: ExecuTorch Team at Meta
ext_author: Junghwan Park (박정환)
category: ["pytorch.org", "translation"]
date: 2026-08-10 12:00:00
org_title: "Fast, On Device Agentic AI with Muse Glimmer on ExecuTorch"
org_link: https://pytorch.org/blog/fast-ondevice-agentic-ai-with-executorch/
---

![ExecuTorch에서 Muse Glimmer로 구현하는 빠른 온디바이스 에이전틱 AI 대표 이미지 / Fast, On Device Agentic AI with Muse Glimmer on ExecuTorch](/assets/blog/2026-08-10-fast-ondevice-agentic-ai-with-executorch/hero.png){:style="width:100%"}

오늘 Meta가 [Muse Glimmer를 공개했습니다](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model). 온디바이스(on-device) 에이전틱(agentic) 워크플로우를 위해 Meta의 Muse Spark에서 증류한, 매개변수 300억 개 규모의 오픈 웨이트(open-weight) 모델입니다. 이와 함께 ExecuTorch는 NVIDIA GPU와 Apple 실리콘 기반 Mac에서 Muse Glimmer를 실행할 수 있도록 엔드투엔드(end-to-end) 지원을 추가합니다.
> Today, Meta [introduced Muse Glimmer](https://research.meta.ai/blog/introducing-muse-glimmer-open-agentic-model), an open-weight, 30-billion-parameter model distilled from Meta's Muse Spark for on-device agentic workflows. Alongside, ExecuTorch is adding end-to-end support for running Muse Glimmer on NVIDIA GPUs and Macs with Apple silicon.

## 왜 ExecuTorch인가? / Why ExecuTorch?

대부분의 로컬 AI 프레임워크는 모델을 Python이 아닌 다른 언어로 다시 구현합니다. LLM이 표준적인 텍스트 트랜스포머이던 시절에는 이 방식이 잘 확장됐지만, 오늘날의 모델은 점점 더 복잡해지고 있습니다. 새로운 아키텍처, 멀티모달 입출력, 낮은 지연 시간을 위한 [DFlash](https://arxiv.org/abs/2602.06036)(확산 기반 병렬 추측 디코딩) 같은 고급 디코딩 알고리즘이 그렇습니다. 이런 것들을 백엔드마다 다시 구현하는 방식은 확장되지 않습니다.
> Most local AI frameworks rewrite models in other non-Python languages. That scaled well when LLMs were standard text transformers, but today's models are becoming more complex – novel architectures, multimodal inputs and outputs, advanced decoding algorithms like [DFlash](https://arxiv.org/abs/2602.06036) (parallel diffusion-based speculative decoding) for low latency. Reimplementing these across different backends doesn't scale.

[ExecuTorch](https://pytorch.org/blog/introducing-executorch-1-0/)는 다른 접근을 택합니다. 머신러닝 엔지니어와 연구자는 모델(과 그 디코딩 전략)을 PyTorch로 구현하기만 하면 됩니다. 배포할 준비가 되면 ExecuTorch로 익스포트(export)하고, 백엔드별 저수준화(lowering)는 프레임워크가 알아서 처리합니다. CUDA에서는 Triton으로, Apple 실리콘에서는 MLX 네이티브 구현과 커스텀 Metal 구현으로 내려갑니다. 사전 컴파일(ahead-of-time compilation)은 개별 연산만이 아니라 실행 경로 전체를 엔드투엔드로 최적화합니다.
> [ExecuTorch](https://pytorch.org/blog/introducing-executorch-1-0/) takes a different approach. As machine learning engineers and researchers, you implement the model (and its decoding strategy) in PyTorch. Once you're ready for deployment, you export to ExecuTorch, and the framework handles backend-specific lowering, Triton on CUDA, MLX-native and custom Metal on Apple silicon. Ahead-of-time compilation optimizes the full execution path end-to-end, not just individual ops.

Muse Glimmer의 텍스트·이미지 입력, GGUF 직접 익스포트, K-quant 네이티브 실행, 128K 이상의 토큰 컨텍스트, DFlash 추측 디코딩(speculative decoding) 기능도 이런 방식으로 제공합니다. 미리 빌드된 PTE 아티팩트 번들도 공개했으니, 이를 내려받아 지원되는 NVIDIA GPU나 Apple 실리콘 기반 Mac에서 ExecuTorch 런타임으로 실행할 수 있습니다.
> This is how we ship Muse Glimmer's text and image inputs, direct GGUF export, native K-quant execution, 128K+-token context, and DFlash speculative decoding features. We have released prebuilt PTE artifact bundles that you can download and run on supported NVIDIA GPUs or Macs with Apple silicon using the ExecuTorch runtime.

## 빠르게 시작하기 / Quickstart

### PTE 받기 / Getting the PTEs

PTE는 ExecuTorch Python 스택이 모델의 PyTorch 그래프로부터 사전에 만들어내는 직렬화된 아티팩트로, 대상 백엔드에 맞춰 최적화되어 있습니다.
> A PTE is the serialized artifact produced ahead of time from a model's PyTorch graph by the ExecuTorch Python stack, and optimized for a target backend.

#### 내려받기(권장) / Download (Preferred)

NVIDIA CUDA와 Apple 실리콘(Metal)용으로 검증한 PTE를 Hugging Face에 공개했습니다. 텍스트 전용 아티팩트와 텍스트+이미지 아티팩트가 모두 있으며, 각각 DFlash 추측 디코딩을 적용한 것과 적용하지 않은 것이 있습니다. [여기](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE/tree/main)에서 내려받으세요.
> We have published verified PTEs on Hugging Face for NVIDIA CUDA and Apple Silicon (Metal). This includes text-only and text-plus-image artifacts, with and without DFlash speculative decoding. Download them here: [link](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE/tree/main).

#### 직접 빌드하기 / Build your own

미리 빌드된 PTE로 시작하는 것이 가장 빠르게 실행해 보는 방법입니다. 직접 빌드하려면 ExecuTorch의 Muse Glimmer [README](https://github.com/pytorch/executorch/blob/main/examples/models/muse-glimmer/README.md)를 따라 백엔드와 모달리티(modality), 컨텍스트 길이, DFlash 사용 여부를 선택하면 됩니다. ExecuTorch는 `torch.export` 기반 사전 컴파일 스택을 통해 공개된 GGUF 체크포인트에서 곧바로 익스포트합니다. CUDA 익스포트는 감지된 GPU 아키텍처에 맞춰 Triton 커널을 컴파일하고 자동 튜닝(autotune)합니다. 최상의 결과를 얻으려면 아티팩트를 실행할 GPU와 같은 아키텍처에서 익스포트하세요.
> Starting with a prebuilt PTE is the fastest way to get running. To build your own, follow the ExecuTorch Muse Glimmer [README](https://github.com/pytorch/executorch/blob/main/examples/models/muse-glimmer/README.md), and select the backend, modality, context length, and whether to use DFlash. ExecuTorch exports directly from the released GGUF checkpoints through its torch.export-based ahead-of-time stack. CUDA export compiles and autotunes Triton kernels for the detected GPU architecture. For the best results, export on the same GPU architecture that will run the artifact.

## PTE 실행하기 / Executing the PTEs

### 1. 런타임 빌드하기 / 1. Build the runtime

ExecuTorch는 이 모델 러너(runner)를 위해 [CUDA](https://docs.pytorch.org/executorch/stable/backends/cuda/cuda-overview.html)와 [MLX 백엔드](https://pytorch.kr/blog/2026/executorch-mlx-delegate/) 양쪽의 CMake 프리셋을 함께 제공합니다. [여기](https://docs.pytorch.org/executorch/stable/getting-started.html)의 ExecuTorch 설치 안내를 따른 뒤, 선택한 PTE에 맞춰 추측 디코딩을 적용하거나 적용하지 않은 러너를 CMake로 빌드하세요. DFlash를 쓰는 러너와 쓰지 않는 러너 모두 텍스트와 이미지 모달리티를 지원하며, 에이전틱 사용 사례를 위한 ExecuTorch의 예제 `llm_server` 와도 호환됩니다.
> ExecuTorch ships CMake presets for both the [CUDA](https://docs.pytorch.org/executorch/stable/backends/cuda/cuda-overview.html) and [MLX backends](https://pytorch.org/blog/running-pytorch-models-on-apple-silicon-gpus-with-the-executorch-mlx-delegate/) for this model runner(s). Follow ExecuTorch installation instructions [here](https://docs.pytorch.org/executorch/stable/getting-started.html), and then use CMake to build the runners with or without speculative decoding for the PTE you selected. Both, with and without DFlash, runners support text and image modalities and are compatible with the example llm\_server in ExecuTorch for agentic use cases.

```bash
# 런타임 빌드

# ExecuTorch를 설치한 뒤, 사용할 백엔드에 맞춰 러너를 빌드합니다:

$ cd examples/models/muse-glimmer
$ cmake --workflow --preset muse-glimmer-cuda # macOS에서는 muse-glimmer-mlx 사용

# 이 명령으로 solo_runner, dflash_runner, 서빙 워커가 빌드됩니다.
```

### 2. PTE 실행하기 / 2. Run the PTEs

러너를 빌드했다면, 아래는 PTE를 실행하는 몇 가지 예시입니다.
> Here are some examples of how to run the PTEs, once you have built the runners.

```bash
# 예시 1: 명령줄에서 단독 실행

$ PROMPT='<|start|>user<|message|>Describe this image: <img><|eot|><|start|>assistant'

$ cmake-out/examples/models/muse-glimmer/dflash_runner \
--model_path artifacts/dflash-vision/model.pte \
--data_path artifacts/dflash-vision/aoti_cuda_blob.ptd \
--tokenizer_path assets/hf/tokenizer.json \
--image_path image.jpg --prompt "$PROMPT" \
--block_length 4 --n_draft 3 --temperature 0 --max_new_tokens 256
```

```bash
# 예시 2, 1/2 단계: 에이전트 서버 시작

$ python -m executorch.examples.models.muse_glimmer.serving.serve \
--model-path artifacts/dflash-vision/model.pte \
--data-path artifacts/dflash-vision/aoti_cuda_blob.ptd \ # cuda에서만 필요
--tokenizer-path assets/hf/tokenizer.json --hf-tokenizer assets/hf \
--worker-bin cmake-out/examples/models/muse-glimmer/muse_glimmer_worker \
--tool-parser atem --max-context 131072

# API는 http://127.0.0.1:8000/v1 에서 제공됩니다

# 예시 2, 2/2 단계: 에이전트 시작 (Pi를 예로 사용)

$ pi \
--provider muse-glimmer-local \
--model muse-glimmer \
--thinking high \
--tools read,bash,edit,write

# 이 명령은 로컬 muse glimmer 서버를 사용해 pi 에이전트를 자동으로 시작합니다.
# 먼저 ~/.pi/agent/models.json 에 muse_glimmer-local 을 등록하세요.
# 자세한 내용은 README.md 를 참고하세요.
```

## 가능해진 사용 사례 / Use cases enabled

### 추측 디코딩 적용 여부에 따른 Muse Glimmer의 이미지 이해 / Image understanding with Muse Glimmer, with and without speculative decoding

![M5 Pro에서 진행한 Muse Glimmer 텍스트-이미지 입력 실험 / Muse Glimmer text-image input experiment on M5 Pro](/assets/blog/2026-08-10-fast-ondevice-agentic-ai-with-executorch/Muse-Glimmer-text-image-input-experiment-on-M5-Pro.gif){:style="width:100%"}
*그림 1: M5 Pro(64GiB)에서 진행한 Muse Glimmer 텍스트-이미지 입력 실험. Solo는 21.6 tok/s를 기록한 반면, 추측 디코딩을 적용한 구성(DFlash)은 33.0 tok/s에 도달해 품질 저하 없이 52.8%의 성능 향상을 보였습니다 / Figure 1: Muse Glimmer text-image input experiment on M5 Pro (64 GiB). Solo achieves 21.6 tok/s, while our speculative decoding set up (DFlash) reaches 33.0 tok/s, a 52.8% performance improvement without quality regression*

### ExecuTorch로 Pi 코딩 에이전트를 구동하는 Muse Glimmer / Muse Glimmer powering Pi Coding Agent through ExecuTorch

![Pi 코딩 에이전트를 사용해 M5 Pro에서 동작하는 Muse Glimmer 에이전트 파이프라인 / Muse Glimmer agent pipeline on an M5 Pro using the Pi coding agent](/assets/blog/2026-08-10-fast-ondevice-agentic-ai-with-executorch/Muse-Glimmer-agent-pipeline-on-an-M5-Pro-using-the-Pi-coding-agent.gif){:style="width:100%"}
*그림 2: Pi 코딩 에이전트를 사용해 M5 Pro(64GB)에서 동작하는 Muse Glimmer 에이전트 파이프라인. 이 에이전트는 새를 테마로 한 게임을 만들면서, 긴 추론으로 세부 사항을 반복해 다듬고, 도구를 호출해 파일을 만들고, 필요한 패키지를 설치하고, 테스트를 작성해 실행하며, 다음 단계와 추가 요구사항을 사용자에게 먼저 물어봅니다 / Figure 2: Muse Glimmer agent pipeline on an M5 Pro (64 GB) using the Pi coding agent. The agent creates a bird-themed game, iteratively refining details through extended reasoning, calling tools to create files, installing required packages, writing and running tests, and proactively asking the user about next steps and additional requirements*

## 성능 / Performance

![NVIDIA A100과 Apple Mac에서 컨텍스트 길이를 달리한 텍스트 전용 입력으로 측정한 ExecuTorch에서의 Muse Glimmer 성능 / Muse Glimmer performance on ExecuTorch using text-only input with varying context on NVIDIA A100 and Apple Mac](/assets/blog/2026-08-10-fast-ondevice-agentic-ai-with-executorch/Muse-Glimmer-performance-on-ExecuTorch-using-text-only-input-with-varying-context-on-NVIDIA-A100-and-Apple-Mac.png){:style="width:100%"}
*그림 3: 컨텍스트 길이를 달리한 텍스트 전용 입력으로 측정한 ExecuTorch에서의 Muse Glimmer 성능. NVIDIA A100(RTX 카드를 대신하는 기준)과 M5-max를 탑재한 Apple Mac에서, 코딩 프롬프트를 사용해 DFlash 적용 여부에 따른 프리필과 디코드 성능을 초당 토큰 수로 측정했습니다. 이 코딩 프롬프트는 해당 모델에서 수용률(acceptance rate)도 좋은 편인데, 디코드 차트에서 이를 확인할 수 있습니다 / Figure 3: Muse Glimmer performance on ExecuTorch using text-only input with varying context on NVIDIA A100 (as a proxy for RTX cards) and Apple Mac with an M5-max measuring prefill and decode performance in tokens/second with and without DFlash using coding prompt, which also has a good acceptance rate for this model, as seen in the decode charts*

## 내부 동작 / Under the Hood

이제 Muse Glimmer는 NVIDIA GPU와 Apple 실리콘 GPU 양쪽에서 ExecuTorch 위에서 엔드투엔드로 동작합니다. 이를 위해 구현한 핵심 기능과 최적화를 몇 가지 소개합니다.
> Muse Glimmer now runs end-to-end on ExecuTorch on both NVIDIA GPUs and Apple Silicon GPUs. Here are some of the key capabilities and optimizations we built.

### DFlash 추측 디코딩 지원 / Enabling DFlash speculative decoding

- 가중치를 공유해 타깃 모델과 드래프트 모델의 상호 운용을 최적화했고, 둘을 하나의 PTE로 익스포트했습니다.
- DFlash의 블록 차원은 동적으로 익스포트되므로, 하나의 PTE로 런타임에 블록 길이를 선택할 수 있습니다.
- 런타임은 그리디 디코딩(greedy decoding)과 거부 샘플링(rejection sampling)을 모두 지원합니다.

> - We optimized target and draft interoperability through weight sharing, exporting both into a single PTE.
> - The DFlash block dimension is exported dynamically, allowing one PTE to support runtime-selectable block lengths.
> - The runtime supports both greedy decoding and rejection sampling.

### GGUF 로딩과 K-quant 지원 / Supporting GGUF loading and k-quant

- Muse Glimmer와 함께 공개된 GGUF에서 곧바로 익스포트합니다.
- CUDA에서는 Q4\_K/Q5\_K/Q6\_K를 dp4a GEMV 커널을 사용하는 패킹된 INT4/5/6으로 매핑하고, MLX에서는 재패킹하거나 융합한 Metal 커널로 매핑합니다.
- MLX에서는 성능을 위해, 재패킹 시점에 스케일과 최솟값이 동일한 인접 서브블록들을 병합이 무손실인 경우에 한해 최대 128까지 더 큰 그룹 크기로 합칩니다.

> - We export straight from the GGUF released with the Muse Glimmer.
> - We map Q4\_K/Q5\_K/Q6\_K to packed INT4/5/6 with dp4a GEMV kernels on CUDA, and to repacked or fused Metal kernels on MLX.
> - On MLX, for performance, at repack time we merge adjacent sub-blocks whose scale and min are identical into a larger group size, up to 128, whenever the merge is lossless.

### 에이전틱 하네스(agentic harness)와 LLM 서빙 / Agentic harness and LLM serving

- 두 백엔드에 추가한 세션별 가변 상태 재바인딩(mutable-state rebinding) 덕분에, 모델을 한 번만 로드해도 서로 격리된 여러 대화를 처리할 수 있습니다.
- 추론(reasoning) 라우팅을 갖춘 Harmony 채팅 템플릿을 추가했습니다.
- 한 턴에 여러 번 호출하는 경우까지 포함해, 모델의 XML 도구 호출 형식을 처리하는 파서를 추가했습니다.

> - One model load serves multiple isolated conversations, through per-session mutable-state rebinding we added to both backends.
> - We added Harmony chat templating with reasoning routing.
> - We added a parser for the model's XML tool-call format, including multiple calls in one turn.

### 백엔드별 성능 최적화 / Backend-specific performance optimizations

**CUDA**

- 디코드를 CUDA 그래프로 캡처해, 커널마다 발생하던 실행 오버헤드를 한 번의 제출로 줄였습니다.
- 패킹된 K-quant 커널은 배치가 작은 디코드를 가속하고, 길이를 인식하는 split-K FlashDecoding++ 경로는 단일 토큰 디코드와 작은 DFlash 검증 블록을 최적화합니다.

> **CUDA**
> - We capture decode into a CUDA graph, reducing per-kernel launch overhead into one submission.
> - Packed K-quant kernels accelerate low-batch decode, while length-aware split-K FlashDecoding++ paths optimize single-token decode and small DFlash verification blocks.

**MLX**

- RMSNorm, RoPE, SDPA, KV 캐시 갱신, 양자화된 선형 연산은 MLX 네이티브 구현이나 커스텀 Metal 구현으로 저수준화됩니다.
- GGUF K-quant 가중치는 재패킹한 MLX 네이티브 연산이나 융합한 Metal 커널을 사용합니다.

> **MLX**
> - RMSNorm, RoPE, SDPA, KV-cache updates, and quantized linear operations are lowered to MLX-native or custom Metal implementations.
> - GGUF K-quant weights use either repacked MLX-native operations or fused Metal kernels.

### 긴 컨텍스트 지원 / Supporting long context

Muse Glimmer는 128K가 넘는 토큰 컨텍스트를 지원하며, KV 캐시가 늘어나는 방식도 효율적입니다. 전체 52개 계층 중 13개만 글로벌이고, 나머지 39개는 슬라이딩 윈도우 방식입니다. ExecuTorch는 이를 효율적으로 지원해, 엣지 기기에서도 긴 컨텍스트를 쓰는 사용 사례를 현실적인 선택지로 만들어 줍니다.
> Muse Glimmer supports a 128K+ token context, and is efficient in how its KV-cache grows: only 13 of its 52 layers are global; the other 39 are sliding-window. ExecuTorch supports this efficiently, making the long context use cases practical on edge devices.

## 다음 단계 / What's next

- 이번 첫 릴리즈는 텍스트와 이미지 입력을 지원하며, 영상 입력은 아직 지원하지 않습니다. 현재 작업이 진행 중입니다.
- 현재로서는 세션 간 프리픽스 공유(prefix sharing)나 체크포인팅, 연속 배칭(continuous batching)을 지원하지 않습니다. ExecuTorch를 에이전틱 워크플로우에 더 알맞게 만들기 위해 모두 활발히 작업하고 있습니다.

> - This initial release supports text and image inputs; video input is not yet supported. It is a work in progress.
> - No cross-session prefix sharing or checkpointing or continuous batching as of now. These are all actively being worked on to make ExecuTorch even more suitable for agentic workflows.

ExecuTorch로 Muse Glimmer를 사용해 보고, [Discord](https://discord.com/invite/Dh43CKSAdc)에서 의견을 들려주세요. 문제가 생기면 언제든 GitHub [이슈](https://github.com/pytorch/executorch/issues/new/choose)를 열어 주세요.
> Try Muse Glimmer with ExecuTorch and let us know what you think on [Discord](https://discord.com/invite/Dh43CKSAdc). If you run into any issues, feel free to open a Github [Issue](https://github.com/pytorch/executorch/issues/new/choose).

## 참고 자료 / References

[ExecuTorch의 Muse Glimmer](https://github.com/pytorch/executorch/blob/main/examples/models/muse-glimmer/README.md) | [Hugging Face의 Muse Glimmer](https://huggingface.co/meta-models) | [ExecuTorch 문서](https://docs.pytorch.org/executorch/main/getting-started.html) | [GitHub의 ExecuTorch](https://github.com/pytorch/executorch)
> [Muse Glimmer in ExecuTorch](https://github.com/pytorch/executorch/blob/main/examples/models/muse-glimmer/README.md) | [Muse Glimmer on Hugging Face](https://huggingface.co/meta-models) | [ExecuTorch Documentation](https://docs.pytorch.org/executorch/main/getting-started.html) | [ExecuTorch on Github](https://github.com/pytorch/executorch)
