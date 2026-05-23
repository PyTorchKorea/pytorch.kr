---
layout: blog_detail
title: "ExecuTorch MLX 델리게이트로 Apple Silicon GPU에서 PyTorch 모델 실행하기"
author: ExecuTorch Team
category: ["pytorch.org", "translation"]
org_title: "Running PyTorch Models on Apple Silicon GPUs with the ExecuTorch MLX Delegate"
org_link: https://pytorch.org/blog/running-pytorch-models-on-apple-silicon-gpus-with-the-executorch-mlx-delegate/
---

**TL;DR: ExecuTorch MLX 델리게이트 소개 / TL;DR: Introducing the ExecuTorch MLX Delegate**

- 새로운 MLX 델리게이트는 Apple의 MLX 프레임워크를 사용해 Apple Silicon Mac에서 PyTorch 모델의 GPU 가속 추론(inference)을 최적화된 형태로 수행할 수 있게 합니다.
- 이 델리게이트는 PyTorch 2 export 스택과 매끄럽게 통합되며, 다양한 양자화(quantization) 옵션(BF16, FP16, FP32, 2/4/8비트 affine, NVFP4)을 지원합니다.
- 밀집(dense) 트랜스포머(Llama, Qwen, Gemma), 희소(sparse) Mixture-of-Experts, 그리고 오프라인 및 실시간 전사(transcription)를 위한 음성-텍스트 모델(Whisper, Voxtral, Parakeet) 등 다양한 모델을 지원합니다.
- *참고: MLX 델리게이트는 현재 실험적(experimental) 단계입니다.*

> - The new MLX delegate enables optimized, GPU-accelerated inference for PyTorch models on Apple Silicon Macs, using Apple's MLX framework.
> - The delegate seamlessly integrates with the PyTorch 2 export stack and supports a wide range of quantization options (BF16, FP16, FP32, 2/4/8-bit affine, NVFP4).
> - It supports various models, including dense transformers (Llama, Qwen, Gemma), sparse Mixture-of-Experts, and speech-to-text models (Whisper, Voxtral, Parakeet) for both offline and real-time transcription.
> - *Note: The MLX delegate is currently experimental.*

![ExecuTorch MLX 델리게이트 / ExecuTorch MLX Delegate](/assets/blog/2026-05-18-executorch-mlx-delegate/1.png){:style="width:100%"}

Apple Silicon은 로컬에서 대규모 언어 모델(LLM)을 실행하는 플랫폼으로 인기를 얻고 있습니다. 지금까지 macOS에서 [ExecuTorch](https://github.com/pytorch/executorch) 사용자는 XNNPACK이나 AOTI Metal 백엔드 같은 CPU 기반 백엔드만 사용할 수 있었습니다. 이번에 출시한 MLX 델리게이트는 Apple의 [MLX](https://github.com/ml-explore/mlx) 프레임워크를 통해 Apple Silicon Mac에 완전히 최적화된 GPU 가속 추론을 제공합니다.
> Apple Silicon has become a popular platform for running large language models locally. Until now, [ExecuTorch](https://github.com/pytorch/executorch) users on macOS were limited to CPU-based backends like XNNPACK or the AOTI Metal backend. Now we've released the MLX delegate, which brings fully optimized GPU-accelerated inference to Apple Silicon Macs through Apple's [MLX](https://github.com/ml-explore/mlx) framework.

이번 글에서는 MLX 델리게이트가 무엇인지, 왜 이를 ExecuTorch 백엔드로 만들었는지, 그리고 오늘 무엇을 실행할 수 있는지 살펴봅니다.
> In this post we'll cover what the MLX delegate is, why we built it as an ExecuTorch backend, and what you can run with it today.

**참고:** MLX 델리게이트는 현재 실험적이며 활발히 개발 중입니다. API와 지원 기능은 변경될 수 있습니다.
> **Note:** The MLX delegate is currently experimental and under active development. APIs and supported features may change.

## MLX 델리게이트란? / What is the MLX Delegate?

MLX 델리게이트는 Apple Silicon GPU에서 PyTorch 모델을 컴파일하고 실행하는 새로운 ExecuTorch 백엔드입니다. 표준 ExecuTorch 파이프라인을 사용해 모델을 export하면, 델리게이트가 나머지 작업을 처리합니다. 그래프를 분할하고, 최적화된 포맷으로 직렬화(serialize)한 뒤, 런타임에 연산을 MLX의 Metal GPU 커널로 디스패치(dispatch)하는 일까지 모두 수행합니다.
> The MLX delegate is a new ExecuTorch backend that compiles and runs PyTorch models on Apple Silicon GPUs. You export your model using the standard ExecuTorch pipeline, and the delegate handles the rest: partitioning the graph, serializing it into an optimized format, and dispatching operations to MLX's Metal GPU kernels at runtime.

사용자 관점에서 워크플로우는 다른 ExecuTorch 백엔드와 동일합니다.
> From the user's perspective, the workflow is the same as any other ExecuTorch backend:

1. `torch.export`로 모델을 내보냅니다.
2. `MLXPartitioner`와 함께 `to_edge_transform_and_lower`로 로워링(lower)합니다.
3. 생성된 `.pte` 파일을 ExecuTorch 런타임에서 실행합니다.

> 1. Export your model with `torch.export`
> 2. Lower it with `to_edge_transform_and_lower` using the `MLXPartitioner`
> 3. Run the resulting `.pte` file with the ExecuTorch runtime

이 델리게이트는 현재 약 90개의 ATen 연산을 지원하며, 트랜스포머 추론에 필요한 전체 연산 범위를 포함합니다. 양자화된 matmul, 멀티헤드 어텐션(multi-head attention), 회전 위치 임베딩(rotary position embeddings), mixture-of-experts 라우팅, 순환 상태 공간(recurrent state-space) 연산 등이 포함됩니다.
> The delegate currently supports around 90 ATen ops, covering the full range of operations needed for transformer inference: quantized matmul, multi-head attention, rotary position embeddings, mixture-of-experts routing, recurrent state-space operations, and more.

## 왜 ExecuTorch 델리게이트로 만들었는가? / Why Build This as an ExecuTorch Delegate?

Apple Silicon에서 모델을 실행할 수 있는 훌륭한 도구는 이미 존재합니다. MLX 자체의 `mlx-lm`도 그 중 하나입니다. 그런데 왜 또 다른 도구를 만들었을까요? 세 가지 이유가 있습니다.
> There are already excellent tools for running models on Apple Silicon, including MLX's own `mlx-lm`. So why build another one? Three reasons:

**성능(Performance).** MLX 델리게이트는 macOS에서 기존 ExecuTorch 델리게이트 대비 생성형 AI 워크로드에서 3~6배 높은 처리량(throughput)을 달성합니다. 추론을 MLX의 최적화된 Metal 커널로 옮기면 채팅이나 실시간 전사 같은 ExecuTorch 애플리케이션에서 의미 있는 차이를 만들어냅니다.
> **Performance.** The MLX delegate achieves 3-6x higher throughput on generative AI workloads compared to existing ExecuTorch delegates on macOS. Moving inference to MLX's optimized Metal kernels makes a meaningful difference for ExecuTorch applications like chat and real-time transcription.

**PyTorch 2 통합(PyTorch 2 integration).** 델리게이트는 PyTorch 2 export 스택에 직접 연결됩니다. 그래프 캡처에는 `torch.export`를, 양자화에는 TorchAO를 사용하는데, 이는 다른 모든 ExecuTorch 백엔드에서 사용하는 도구와 동일합니다. `torch.export`로 export할 수 있는 모델이라면 MLX에서도 실행할 수 있습니다. PyTorch에 새로운 모델이나 양자화 기법이 추가되면, 추가 작업 없이 MLX 델리게이트에서도 사용할 수 있게 됩니다.
> **PyTorch 2 integration.** The delegate plugs directly into the PyTorch 2 export stack. It uses `torch.export` for graph capture and TorchAO for quantization, the same tools used by every other ExecuTorch backend. If you can export a model with `torch.export`, you can run it on MLX. When new models or quantization techniques land in PyTorch, they become available to the MLX delegate without additional work.

**이식 가능한 애플리케이션(Portable applications).** ExecuTorch는 모든 백엔드에 걸쳐 단일한 런타임 API를 제공합니다. ExecuTorch C++ 또는 Python 런타임을 기반으로 만든 애플리케이션은 MLX, XNNPACK, CoreML, Vulkan, CUDA 중 어떤 백엔드로 export된 모델이든 애플리케이션 코드를 변경하지 않고 실행할 수 있습니다.
> **Portable applications.** ExecuTorch provides a single runtime API across all backends. An application built against the ExecuTorch C++ or Python runtime can run models exported for MLX, XNNPACK, CoreML, Vulkan, or CUDA without changing application code.

## 양자화 및 데이터 타입 지원 / Quantization and Dtype Support

델리게이트는 온디바이스(on-device) 추론에 기대할 만한 정밀도(precision) 및 양자화 옵션을 지원합니다.
> The delegate supports the precision and quantization options you'd expect for on-device inference:

- 가중치와 활성화 값에 대한 **BF16, FP16, FP32**
- TorchAO의 `quantize_` API를 통한 **2, 4, 8비트 affine 양자화**. XNNPACK 및 Vulkan 백엔드와 동일한 양자화 방식을 사용하므로, 하나의 양자화된 모델 정의로 여러 백엔드를 타겟팅할 수 있고, 런타임에 사용 가능한 백엔드에서 실행되는 fat PTE 파일의 길도 열립니다.
- NVIDIA FP4 데이터 타입을 사용하는 **NVFP4 양자화**
- 임베딩 계층과 언어 모델 헤드가 가중치를 공유하는 모델을 위한 **양자화된 묶인 임베딩(tied embedding)**

> - **BF16, FP16, and FP32** for weights and activations
> - **2, 4, and 8-bit affine quantization** via TorchAO's `quantize_` API. This uses the same quantization scheme as the XNNPACK and Vulkan backends, which means a single quantized model definition can target multiple backends, and opens the door to fat PTE files that run on whichever backend is available at runtime.
> - **NVFP4 quantization** using NVIDIA's FP4 data type
> - **Tied quantized embeddings** for models that share weights between the embedding layer and the language model head

## 어떤 모델을 실행할 수 있나? / What Models Can I Run?

다양한 아키텍처에 걸쳐 델리게이트를 검증했습니다.
> We've validated the delegate across a range of architectures:

### 대규모 언어 모델 / Large Language Models

**밀집(Dense) 트랜스포머**는 전체 KV 캐시와 슬라이딩 윈도우(sliding window) 캐시 모두를 지원하며 그대로 동작합니다.
> **Dense transformers** work out of the box, with support for both full KV caches and sliding window caches:

- Llama 3.2 1B
- Qwen 3 (0.6B, 1.7B, 4B)
- Phi-4 mini (3.8B)
- 슬라이딩 윈도우 어텐션을 사용하는 Gemma 3 (1B, 4B)

> - Llama 3.2 1B
> - Qwen 3 (0.6B, 1.7B, 4B)
> - Phi-4 mini (3.8B)
> - Gemma 3 (1B, 4B) with sliding window attention

**희소(Sparse) Mixture-of-Experts** 모델은 GPU에서 토큰을 올바른 전문가(expert)로 효율적으로 라우팅하는 커스텀 gather 연산을 통해 지원됩니다.
> **Sparse Mixture-of-Experts** models are supported through custom gather operations that efficiently route tokens to the correct experts on the GPU:

- Qwen 3.5 35B-A3B: top-8 라우팅을 사용하는 256개 전문가로, GatedDeltaNet 선형 어텐션 계층과 전체 SDPA 어텐션 계층을 결합

> - Qwen 3.5 35B-A3B: 256 experts with top-8 routing, combining GatedDeltaNet linear attention layers with full SDPA attention layers

### 음성-텍스트 변환 / Speech-to-Text

**오프라인 전사(Offline transcription)** 모델은 전체 오디오 녹음을 처리하여 전사 결과를 반환합니다.
> **Offline transcription** models process a complete audio recording and return the transcript:

- OpenAI Whisper (tiny ~ large-v3-turbo)
- 단어 수준 타임스탬프를 제공하는 NVIDIA Parakeet TDT (0.6B)
- Mistral Voxtral (3B)

> - OpenAI Whisper (tiny through large-v3-turbo)
> - NVIDIA Parakeet TDT (0.6B) with word-level timestamps
> - Mistral Voxtral (3B)

**실시간 스트리밍 전사(Real-time streaming transcription)** 는 오디오가 도착하는 대로 작은 청크 단위로 처리하여, 실시간 사용 사례를 가능하게 합니다.
> **Real-time streaming transcription** processes audio in small chunks as it arrives, enabling live use cases:

- 라이브 마이크 입력, 링 버퍼(ring buffer) KV 캐시, 슬라이딩 윈도우 어텐션을 사용하는 Mistral Voxtral Realtime (4B)

> - Mistral Voxtral Realtime (4B) with live microphone input, ring buffer KV caches, and sliding window attention

![Voxtral Realtime 데모 / Voxtral Realtime demo](/assets/blog/2026-05-18-executorch-mlx-delegate/to_make_gif_cropped.gif){:style="width:100%"}

### 더 폭넓은 커버리지 / Broader Coverage

이러한 대표 모델들 외에도, 백엔드 테스트 스위트를 통해 30개 이상의 추가 모델이 검증되었으며, 밀집 트랜스포머, 인코더-디코더 아키텍처, 비전(vision) 모델 등을 포함합니다.
> Beyond these flagship models, over 30 additional models have been validated through our backend test suites, covering dense transformers, encoder-decoder architectures, and vision models.

## 시작하기 / Getting Started

지원되는 각 모델마다 export 및 추론 방법을 상세히 안내하는 README가 제공됩니다.
> Each supported model has a README with detailed export and inference instructions:

- [HuggingFace를 통한 LLM](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/backends/mlx/examples/llm): optimum-executorch를 사용한 Llama, Qwen, Gemma를 다룹니다
- [export_llm을 통한 LLM](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/llama): Hydra 기반 파이프라인으로 Phi-4와 Stories 110M을 다룹니다
- [Qwen 3.5 MoE](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/qwen3_5_moe): `--backend mlx`로 희소 MoE export를 다룹니다
- [Voxtral Realtime](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/voxtral_realtime): 스트리밍 및 오프라인 음성-텍스트 변환을 다룹니다
- [Parakeet](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/parakeet): 타임스탬프를 포함한 음성 인식을 다룹니다
- [Whisper](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/backends/mlx/examples/whisper): OpenAI 음성 인식 모델을 다룹니다

> - [LLMs via HuggingFace](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/backends/mlx/examples/llm): covers Llama, Qwen, and Gemma using optimum-executorch
> - [LLMs via export_llm](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/llama): covers Phi-4 and Stories 110M using the Hydra-based pipeline
> - [Qwen 3.5 MoE](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/qwen3_5_moe): covers the sparse MoE export with `--backend mlx`
> - [Voxtral Realtime](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/voxtral_realtime): covers streaming and offline speech-to-text
> - [Parakeet](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/examples/models/parakeet): covers speech recognition with timestamps
> - [Whisper](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/backends/mlx/examples/whisper): covers OpenAI's speech recognition models

델리게이트 아키텍처, 지원 연산, 개발 가이드에 대한 개요는 [MLX Delegate README](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/backends/mlx)를 참고하세요.
> For an overview of the delegate architecture, supported operations, and development guide, see the [MLX Delegate README](https://github.com/pytorch/executorch/tree/89600b3954c08f9224df0ef295232f4c835e46a9/backends/mlx).

여러분에게 가장 중요한 모델과 사용 사례가 무엇인지 듣고 싶습니다. 문제가 발생하거나 기능 요청이 있다면 [ExecuTorch GitHub 저장소](https://github.com/pytorch/executorch)에 이슈를 등록하거나 [Discord 채널](https://discord.com/invite/Dh43CKSAdc)에 참여해 주세요.
> We'd love to hear what models and use cases matter most to you. If you run into issues or have feature requests, please open an issue on the [ExecuTorch GitHub repo](https://github.com/pytorch/executorch) or join our [Discord Channel](https://discord.com/invite/Dh43CKSAdc).
