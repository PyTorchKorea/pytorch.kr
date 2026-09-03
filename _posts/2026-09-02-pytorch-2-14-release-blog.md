---
layout: blog_detail
title: "PyTorch 2.14 출시 공지"
author: PyTorch Foundation
author_type: Organization
ext_author: Junghwan Park (박정환)
image: /assets/blog/2026-09-02-pytorch-2-14-release-blog/hero.png
category: ["pytorch.org", "translation"]
date: 2026-09-02 12:00:00
org_title: "PyTorch 2.14 Release Blog"
org_link: https://pytorch.org/blog/pytorch-2-14-release-blog/
---

![PyTorch 2.14 출시 공지 대표 이미지 / PyTorch 2.14 Release Blog](/assets/blog/2026-09-02-pytorch-2-14-release-blog/hero.png){:style="width:100%"}

PyTorch® 2.14([릴리즈 노트](https://github.com/pytorch/pytorch/releases/tag/v2.14.0))의 출시를 발표하게 되어 기쁩니다!
> We are excited to announce the release of PyTorch® 2.14 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0))!

PyTorch 2.14 릴리즈에는 다음과 같은 변경 사항이 포함되어 있습니다:
> The PyTorch 2.14 release features the following changes:

- **NVGEMM이 CuTeDSL로 생성한 CUTLASS 커널을 Inductor에 제공하며,** 에필로그(epilogue) 융합, 스케일링된(scaled) GEMM 및 NVFP4 GEMM, 그룹 리덕션(grouped-reduction) 에필로그를 Triton, ATen과 나란히 오토튜닝(autotuning)합니다
- **PyTorch Distributed를 위한 새로운 nccl2 백엔드**는 torchcomms에서 이식된 것으로, 논블로킹(nonblocking) 커뮤니케이터와 즉시(eager) 커뮤니케이터 분할을 갖춘 완전한 집합 통신(collective) 계약을 구현합니다
- **내결함성(fault tolerance)이 c10d의 일급(first-class) 개념이 되어,** 프로세스 그룹을 제자리에서 재구성(in-place reconfiguration)할 수 있고, 단방향(one-sided) RMA 윈도우를 제공하며, NCCL에서만이 아니라 어떤 백엔드에서도 동작하는 Flight Recorder를 제공합니다
- **Apple Silicon이 네이티브 선형대수를 갖추어,** Jacobi 커널 기반 SVD와 `eigh`, QR, Cholesky를 지원하고, 여기에 5부에 걸친 리덕션 재작성과 MPSGraph에서 Metal 커널로의 추가 이전이 더해집니다
- **torch.switch가 torch.cond를 다중 분기(multi-way branching)로 일반화하고,** 이제 torch.while\_loop를 CUDA 그래프에 캡처할 수 있습니다
- **@dynamic\_spec을 통한 선언적(declarative) 동적 shape(dynamic shape)** 이 torch.compile, torch.export, make\_fx 전반에서 공유됩니다
- **더 넓은 플랫폼 지원**: ROCm 7.14 휠은 TheRock pip SDK로 만들어지고, Intel XPU는 네이티브 그래프 캡처를 추가하며, Inductor는 Rubin(sm\_107)을 타겟팅합니다
- **복소수 값 텐서에 대한 실험적 torch.compile 지원:** 선택적으로 활성화하는 이 지원은 지원되는 복소수 연산을 실수부와 허수부 계산으로 분해하여, 컴파일러 백엔드가 더 많은 복소수 워크로드를 최적화할 수 있게 합니다.

> - **NVGEMM brings CuTeDSL-generated CUTLASS kernels to Inductor,** with epilogue fusion, scaled and NVFP4 GEMM, and grouped-reduction epilogues autotuned alongside Triton and ATen
> - **A new nccl2 backend for PyTorch Distributed,** ported from torchcomms, implementing the full collective contract with nonblocking communicators and eager communicator splitting
> - **Fault tolerance becomes a first-class c10d concept,** with in-place process-group reconfiguration, one-sided RMA windows, and a Flight Recorder that works for any backend rather than only NCCL
> - **Apple Silicon gains native linear algebra,** including Jacobi-kernel SVD, eigh, QR and Cholesky, alongside a five-part reduction rewrite and a further MPSGraph to Metal kernel migration
> - **torch.switch generalizes torch.cond to multi-way branching,** and torch.while\_loop can now be captured in a CUDA graph
> - **Declarative dynamic shapes via @dynamic\_spec,** shared across torch.compile, torch.export and make\_fx
> - **Broader platform support**: ROCm 7.14 wheels are produced from the TheRock pip SDK, Intel XPU adds native graph capture, and Inductor targets Rubin (sm\_107)
> - **Experimental torch.compile support for complex-valued tensors:** Opt-in support decomposes supported complex operations into real and imaginary computations, enabling compiler backends to optimize more complex-number workloads.

이번 릴리즈는 PyTorch 2.13 이후 487명의 기여자로부터 2,995회의 커밋으로 구성되었습니다. 헌신적인 커뮤니티의 기여에 진심으로 감사드립니다. 언제나 그렇듯, 새로운 버전을 사용해 보시고 문제가 있으면 보고해 주시면 2.14를 개선하는 데 도움이 됩니다. PyTorch 2 시리즈를 시작하는 방법에 대한 자세한 정보는 [시작하기](https://pytorch.org/get-started/locally/) 페이지에서 확인할 수 있습니다.
> This release is composed of 2,995 commits from 487 contributors since PyTorch 2.13. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.14. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/locally/) page.

이번 릴리즈에 대해 궁금한 점이 있다면 Q&A 웨비나에서 물어봐 주세요. 2026년 9월 17일 목요일에 진행되며, Andrey Talman(Meta), Natalia Gimelshein(Meta), Joe Spisak(Reflection AI), Chris Gottbrath(Gottbrath Tech, 모더레이터)가 참여해 2.14 릴리즈를 개괄하고 PyTorch와 이번 릴리즈의 새로운 기능에 대한 커뮤니티의 질문에 답변합니다. [지금 등록하세요](https://pytorch.org/event/pytorch-2-14-release-live-qa/).
> Bring any questions you may have about this release to our Q&A Webinar. It will be on Thursday, September 17, 2026 and will feature Andrey Talman (Meta), Natalia Gimelshein (Meta), Joe Spisak (Reflection AI), and Chris Gottbrath (Gottbrath Tech, moderator) who will share an overview of the 2.14 release and provide answers to community questions about PyTorch and the new capabilities in this release. [Register today](https://pytorch.org/event/pytorch-2-14-release-live-qa/).

2026년 10월 20-21일 미국 캘리포니아 산호세에서 열리는 [PyTorch Conference North America](https://hubs.la/Q04vXWRN0)에서 전 세계 PyTorch 커뮤니티와 만나보세요. 컴파일러와 런타임 작업, 분산 통신, 디바이스 이식성, 릴리즈 엔지니어링, CI, 관측 가능성(observability), 가속기 통합, 기여자 인프라 등을 아우르는 세션에서 PyTorch 프레임워크의 새로운 소식을 살펴볼 수 있습니다. PyTorch Conference는 학습, 추론, 커널, 애플리케이션, 책임 있는 AI에서 실제 문제를 푸는 엔지니어, 연구자, 메인테이너가 모이는 자리입니다.
> Connect with the global PyTorch community at the upcoming [PyTorch Conference North America](https://hubs.la/Q04vXWRN0) October 20-21, 2026, in San Jose, CA. Explore what's new with PyTorch framework with sessions spanning compiler and runtime work, distributed communication, device portability, release engineering, CI, observability, accelerator integration, contributor infrastructure, and much more. PyTorch Conference is the place for engineers, researchers and maintainers solving real problems in training, inference, kernels, applications and responsible AI to convene.

2.x 시리즈 전반에 걸쳐 PyTorch는 연구 중심의 프레임워크에서, 대규모 프로덕션 학습 및 추론을 위한 통합된 하드웨어에 구애받지 않는(hardware-agnostic) 플랫폼으로 진화해 왔습니다. [PyTorch 2.12](https://pytorch.kr/blog/2026/pytorch-2-12-release-blog/)는 디바이스에 구애받지 않는(device-agnostic) `torch.accelerator.Graph` API와 마이크로스케일링(Microscaling) 양자화 내보내기(export) 지원을 추가했습니다. [PyTorch 2.13](https://pytorch.kr/blog/2026/pytorch-2-13-release-blog/)은 Apple Silicon에 FlexAttention을 도입했고, Inductor에 CuTeDSL 코드 경로를 제공했으며, 대규모 클러스터 학습을 위한 torchcomms를 선보였습니다.
> Throughout the 2.x series, PyTorch has been evolving from a research-first framework into a unified, hardware-agnostic platform for production training and inference at scale. [PyTorch 2.12](https://pytorch.org/blog/pytorch-2-12-release-blog/) added a device-agnostic `torch.accelerator.Graph` API and Microscaling quantization export support. [PyTorch 2.13](https://pytorch.org/blog/pytorch-2-13-release-blog/) landed FlexAttention on Apple Silicon, gave Inductor a CuTeDSL code path, and introduced torchcomms for large-cluster training.

PyTorch 2.14는 그 흐름 위에서 바로 이어집니다. CuTeDSL 경로는 에필로그 융합과 낮은 정밀도(low-precision) 지원을 갖춘 완전한 GEMM 백엔드인 NVGEMM으로 성숙했습니다. torchcomms는 nccl2 백엔드로 트리 안(in-tree)에 안착했고, 내결함성 재구성과 단방향 RMA 윈도우가 내결함성을 백엔드의 세부 사항에서 c10d의 개념으로 끌어올렸습니다. Apple Silicon은 어텐션 커널에서 네이티브 선형대수로 나아갔고, 동적 shape은 컴파일, 내보내기, 추적(tracing)을 넘나드는 스펙을 통해 선언적으로 바뀌었습니다.
> PyTorch 2.14 builds directly on those threads. The CuTeDSL path matures into NVGEMM, a full GEMM backend with epilogue fusion and low-precision support. torchcomms lands in-tree as the nccl2 backend, with fault-tolerant reconfiguration and one-sided RMA windows raising fault tolerance from a backend detail to a c10d concept. Apple Silicon moves from attention kernels to native linear algebra, and dynamic shapes become declarative through a spec that travels across compile, export and tracing.

PyTorch 2.14는 성능, 신뢰성, 하드웨어 지원 전반에 걸쳐 의미 있는 개선을 제공합니다. 이번 릴리즈는 행렬 연산에 가장 빠른 커널을 자동으로 선택하는 새로운 GPU 수학 백엔드 NVGEMM을 도입하며, 학습과 추론에서 메모리 사용량을 줄이는 낮은 정밀도 형식 지원도 포함합니다. 여러 대의 머신에 걸쳐 학습하는 팀에게는 재설계된 분산 통신 백엔드(nccl2)가 더 나은 확장성을 제공하고, 새로운 내결함성 기능은 노드 장애가 발생해도 학습 작업을 처음부터 다시 시작하지 않고 복구할 수 있게 해줍니다.
> PyTorch 2.14 delivers meaningful improvements across performance, reliability, and hardware support. The release introduces NVGEMM, a new GPU math backend that automatically selects the fastest kernel for matrix operations — including support for lower-precision formats that reduce memory use during training and inference. For teams training across many machines, a redesigned distributed communication backend (nccl2) offers better scalability, while new fault-tolerance features allow training jobs to recover from node failures without restarting from scratch.

Apple Silicon 사용자는 네이티브 선형대수 루틴(SVD, QR, Cholesky 등)과, Mac GPU에서 오버헤드를 줄이는 손으로 튜닝한 Metal 커널로 폭넓게 이전한 덕을 봅니다. 컴파일러 측면에서는 새로운 제어 흐름 프리미티브(torch.switch, torch.while\_loop)가 효율적으로 컴파일되는 동적 로직을 작성할 때 모델 작성자에게 더 많은 유연성을 주고, 새로운 @dynamic\_spec 데코레이터는 실행 시점에 어떤 텐서 차원이 바뀔 수 있는지 선언하는 단일하고 깔끔한 방법을 제공하여 컴파일, 내보내기, 추적 전반의 작업 흐름을 단순하게 만듭니다.
> Apple Silicon users benefit from native linear-algebra routines (SVD, QR, Cholesky, and more) and a broad migration to hand-tuned Metal kernels that cut overhead on Mac GPUs. On the compiler side, new control-flow primitives (torch.switch, torch.while\_loop) give model authors more flexibility when writing dynamic logic that still compiles efficiently, and a new @dynamic\_spec decorator provides a single, clean way to declare which tensor dimensions can change at runtime — simplifying workflows across compilation, export, and tracing.

플랫폼 지원은 AMD ROCm 7.14, Intel XPU 네이티브 그래프 캡처, NVIDIA의 차세대 Rubin 아키텍처로 확장됩니다. 내부적으로는 컴파일러가 이제 기본적으로 통신과 계산을 중첩(overlap)하고, 작은 GPU 커널을 더 똑똑하게 묶으며, 호출당 오버헤드를 줄입니다. 이 모두가 사용자의 코드 변경 없이 종단 간(end-to-end) 모델 실행을 더 빠르게 만듭니다.
> Platform support expands to AMD ROCm 7.14, Intel XPU native graph capture, and NVIDIA's next-generation Rubin architecture. Under the hood, the compiler now overlaps communication with computation by default, batches small GPU kernels more intelligently, and reduces per-call overhead — all of which add up to faster end-to-end model execution without any code changes from users.

## 성능 개선 / Performance Improvements

### 대규모 MPS 연산의 네이티브 Metal 이전 / Large MPS Op Migration to Native Metal

2.13에서 시작한 이전을 이어, 또 한 무리의 MPS 연산자가 Apple의 MPSGraph 프레임워크에서 손으로 작성한 Metal 컴퓨트 커널로 옮겨 갔습니다. `index_add`, `index_select`, `argmin`, `argmax`, `conv3d`, `median`, `nanmedian`, `linspace`, `arange`, `nan_to_num`, `log_sigmoid`, `sigmoid_backward`, `mish`, GLU가 여기에 포함됩니다.
> Continuing the migration begun in 2.13, a further set of MPS operators moved off Apple's MPSGraph framework onto hand-written Metal compute kernels, including `index_add`, `index_select`, `argmin`, `argmax`, `conv3d`, `median`, `nanmedian`, `linspace`, `arange`, `nan_to_num`, `log_sigmoid`, `sigmoid_backward`, `mish` and GLU.

리덕션은 전체 리덕션, 안쪽 차원(inner-dimension) 리덕션, 스트라이드 및 배치 바깥쪽(outer) 리덕션, 작은 차원과 좁은(narrow) 커널, 그리고 `argmax`와 `argmin`의 split-K 경로를 아우르는 5부짜리 재작성을 거쳤습니다. 마지막 부분에서는 `min`과 `max`를 MPSGraph에서 이전합니다. 입력 업캐스트(up-cast)를 건너뛰고 vec4 로드를 사용하면 MPSGraph 경로에서는 피할 수 없었던 작업이 제거됩니다.
> Reductions received a dedicated five-part rewrite covering full reductions, inner-dimension reductions, strided and batched outer reductions, small-dimension and narrow kernels, and the `argmax` and `argmin` split-K paths. The final part migrates `min` and `max` off MPSGraph. Skipping input up-casts and using vec4 loads removes work the MPSGraph path could not avoid.

네이티브 Metal 경로는 MPSGraph의 연산별 컴파일 비용을 제거하고, 스레드 디스패치와 메모리 접근 패턴에 대한 직접적인 제어권을 PyTorch에 부여하여, Apple Silicon에서 일반적인 학습 및 추론 워크로드 전반에 걸쳐 커널 실행 지연 시간을 줄입니다.
> The native Metal path eliminates MPSGraph's per-op compilation cost and gives PyTorch direct control over thread dispatch and memory access patterns, reducing kernel launch latency across common training and inference workloads on Apple Silicon.

API 안정성: Unstable
> API Unstable

(PR [#191101](https://github.com/pytorch/pytorch/pull/191101), [#191097](https://github.com/pytorch/pytorch/pull/191097), [#191098](https://github.com/pytorch/pytorch/pull/191098), [#191099](https://github.com/pytorch/pytorch/pull/191099), [#191100](https://github.com/pytorch/pytorch/pull/191100) 작성: Irakli Salia, Hugging Face, [#187109](https://github.com/pytorch/pytorch/pull/187109) 및 [#188802](https://github.com/pytorch/pytorch/pull/188802) 작성: Nikita Shulga, Thinking Machines Lab)
> (PR [#191101](https://github.com/pytorch/pytorch/pull/191101), [#191097](https://github.com/pytorch/pytorch/pull/191097), [#191098](https://github.com/pytorch/pytorch/pull/191098), [#191099](https://github.com/pytorch/pytorch/pull/191099) and [#191100](https://github.com/pytorch/pytorch/pull/191100) by Irakli Salia, Hugging Face, [#187109](https://github.com/pytorch/pytorch/pull/187109) and [#188802](https://github.com/pytorch/pytorch/pull/188802) by Nikita Shulga, Thinking Machines Lab)

### MPS 메모리 및 복사 경로 / MPS Memory and Copy Paths

오래 실행되는 디코드(decode) 워크로드는 MPS 캐싱 할당자가 예약한 메모리 사용량을 필요 이상으로 빠르게 키웠습니다. 이제 할당자는 큰 할당을 버킷으로 묶어 예약 메모리를 제한하고, 배치 힙(placement heap)을 사용해 단편화를 줄입니다.
> Long-running decode workloads grew the MPS caching allocator's reserved footprint faster than necessary. The allocator now buckets large allocations to bound reserved memory and uses placement heaps to reduce fragmentation.

호스트와 디바이스 간 전송 경로도 짧아졌습니다. CPU에서 MPS로의 복사는 이벤트 기반 지연 회수(event-deferred reclaim)와 함께 고정(pinned) 버퍼에서 직접 복사(blit)하고, 연속적이면서 같은 dtype인 복사는 그래프 대신 컴퓨트 커널을 사용하며, 원소별(elementwise) 연산은 안쪽이 연속인 슬라이스 뷰에서 벡터화되고, `cat`은 어떤 차원에서든 쓸 수 있는 벡터화된 연속 빠른 경로를 갖게 됐습니다.
> Host and device transfers also got shorter paths. CPU to MPS copies blit directly from pinned buffers with event-deferred reclaim, contiguous same-dtype copies use a compute kernel instead of a graph, elementwise ops are vectorized on inner-contiguous sliced views, and `cat` gains a vectorized contiguous fast path for any dimension.

API 안정성: Unstable
> API Unstable

(PR [#187441](https://github.com/pytorch/pytorch/pull/187441) 및 [#190438](https://github.com/pytorch/pytorch/pull/190438) 작성: Irakli Salia, Hugging Face, [#189512](https://github.com/pytorch/pytorch/pull/189512) 및 [#188613](https://github.com/pytorch/pytorch/pull/188613) 작성: Nikita Shulga, Thinking Machines Lab, [#188483](https://github.com/pytorch/pytorch/pull/188483) 및 [#188200](https://github.com/pytorch/pytorch/pull/188200) 작성: Joona Havukainen, Apple)
> (PR [#187441](https://github.com/pytorch/pytorch/pull/187441) and [#190438](https://github.com/pytorch/pytorch/pull/190438) by Irakli Salia, Hugging Face, [#189512](https://github.com/pytorch/pytorch/pull/189512) and [#188613](https://github.com/pytorch/pytorch/pull/188613) by Nikita Shulga, Thinking Machines Lab, [#188483](https://github.com/pytorch/pytorch/pull/188483) and [#188200](https://github.com/pytorch/pytorch/pull/188200) by Joona Havukainen, Apple)

### MPS에서의 F.linear 디코드 경로 / F.linear Decode Path on MPS

단일 토큰 디코드는 `[B, 1, K]` 활성화를 `F.linear`에 전달하는데, 이 shape이 MPS에서 빠른 경로를 벗어나 수정 사항에 따르면 bf16과 fp16에서 8.5배의 속도 저하를 유발하고 있었습니다. 이제 시퀀스 길이가 1인 경우가 올바르게 라우팅되고, 자기회귀(autoregressive) 디코딩을 지배하는 벡터-행렬 shape을 새로운 GEMV 커널이 뒷받침합니다. 라우팅 수정과 새 GEMV 커널이 함께, 자기회귀 워크로드에서 MPS와 CUDA 사이에 남아 있던 가장 큰 성능 격차 중 하나를 좁힙니다.
> Single-token decode passes a `[B, 1, K]` activation to `F.linear`, a shape that was falling off the fast path on MPS and costing an 8.5x slowdown on bf16 and fp16 according to the fix. The sequence-length-1 case is now routed correctly, and new GEMV kernels back the vector-matrix shapes that dominate autoregressive decoding. Together, the routing fix and new GEMV kernels close one of the largest remaining performance gaps between MPS and CUDA for autoregressive workloads.

API 안정성: Unstable
> API Unstable

(PR [#189855](https://github.com/pytorch/pytorch/pull/189855) 작성: Giovanni Versiglioni, Apple, [#186927](https://github.com/pytorch/pytorch/pull/186927) 작성: Irakli Salia, Hugging Face)
> (PR [#189855](https://github.com/pytorch/pytorch/pull/189855) by Giovanni Versiglioni, Apple, [#186927](https://github.com/pytorch/pytorch/pull/186927) by Irakli Salia, Hugging Face)

### Inductor에서 기본 활성화된 계산-통신 중첩 / Compute and Communication Overlap On By Default in Inductor

집합 통신을 독립적인 계산과 번갈아 배치하여 통신이 임계 경로(critical path)에 남지 않도록 하는 Inductor의 `simple_overlap` 재정렬이, 이제 선택적 활성화가 아니라 기본으로 켜집니다. 중첩이 기본으로 켜지면서, Inductor로 컴파일된 분산 학습 워크로드는 설정을 바꾸지 않아도 자동으로 더 나은 GPU 활용도를 얻습니다.
> Inductor's `simple_overlap` reordering, which interleaves collectives with independent compute so communication is not left on the critical path, is now enabled by default rather than opt-in. By enabling overlap by default, distributed training workloads compiled through Inductor automatically benefit from better GPU utilization without any configuration change.

API 안정성: Unstable
> API Unstable

(PR [#184240](https://github.com/pytorch/pytorch/pull/184240), [#184235](https://github.com/pytorch/pytorch/pull/184235) 작성: Ivan Kobzarev, Meta)
> (PR [#184240](https://github.com/pytorch/pytorch/pull/184240), #[184235](https://github.com/pytorch/pytorch/pull/184235)by Ivan Kobzarev, Meta)

### 학습 그래프에 대한 reorder\_for\_locality 선택적 활성화 / reorder\_for\_locality Opt-In for Training Graphs

Inductor의 post-grad 지역성(locality) 재정렬 패스인 `reorder_for_locality`를, 이제 새로운 reorder\_for\_locality\_in\_training 설정(기본값 off)을 통해 학습 그래프에서도 선택적으로 활성화할 수 있습니다. 이전에는 추론에서만 동작했습니다. 이로써 지역성 최적화가 학습 워크로드까지 확장되어, 기본 동작에 영향을 주지 않으면서 학습 그래프의 캐시 동작을 개선할 수 있는 튜닝 옵션이 생겼습니다.
> `reorder_for_locality`, Inductor's post-grad locality reordering pass, can now be opted into on training graphs via the new reorder\_for\_locality\_in\_training config (default off), where before it only ran on inference. This extends locality optimization to training workloads, giving users a tuning knob to improve cache behavior in training graphs without affecting default behavior.

API 안정성: Unstable
> API Unstable

(PR [#186643](https://github.com/pytorch/pytorch/pull/186643) 작성: @reger-men)
> (PR [#186643](https://github.com/pytorch/pytorch/pull/186643) by @reger-men)

### Inductor의 콤보 커널과 리덕션 / Combo Kernels and Reductions in Inductor

콤보 커널(combo kernel)은 여러 작은 커널을 하나의 실행으로 묶지만, 배치 안에 아주 큰 리덕션이 하나라도 있으면 이전에는 그 리덕션이 커널 전체의 형태를 좌우했습니다. 이제 큰 리덕션은 콤보 분할(combo partitioning)에서 분리되고, 콤보 리덕션은 동적 `RBLOCK` 스케일링을 사용하며, 하위 커널의 본문은 레지스터 압박을 낮추기 위해 인라인되지 않은 디바이스 함수로 방출됩니다. 콤보 커널마다 본문이 공유되고, split-reduction 휴리스틱은 GB200에 맞춰 튜닝됐습니다. 결과적으로 커널 실행 횟수가 줄고 자원 사용이 더 촘촘해지며, 지나치게 큰 리덕션 하나가 융합된 배치 전체에 불이익을 주던 문제가 해소됩니다.
> Combo kernels batch many small kernels into one launch, but a single very large reduction in the batch would previously shape the whole kernel. Large reductions are now split out of combo partitioning, combo reductions get dynamic `RBLOCK` scaling, and sub-kernel bodies are emitted as non-inlined device functions to keep register pressure down. For each combo kernel bodies are shared, and split-reduction heuristics are tuned for GB200. The net effect is fewer kernel launches with tighter resource usage, closing a gap where one oversized reduction could penalize an entire fused batch.

API 안정성: Unstable
> API Unstable

(PR [#186668](https://github.com/pytorch/pytorch/pull/186668), [#186957](https://github.com/pytorch/pytorch/pull/186957), [#190689](https://github.com/pytorch/pytorch/pull/190689) 작성: Karthick Panner Selvam, Meta, [#184323](https://github.com/pytorch/pytorch/pull/184323) 작성: Jason Ansel, Meta, [#188579](https://github.com/pytorch/pytorch/pull/188579) 작성: Liqiang Lu, Nvidia)
> (PR [#186668](https://github.com/pytorch/pytorch/pull/186668), [#186957](https://github.com/pytorch/pytorch/pull/186957) and [#190689](https://github.com/pytorch/pytorch/pull/190689) by Karthick Panner Selvam, Meta, [#184323](https://github.com/pytorch/pytorch/pull/184323) by Jason Ansel, Meta, [#188579](https://github.com/pytorch/pytorch/pull/188579) by Liqiang Lu, Nvidia)

### Dynamo의 호출당 오버헤드 / Dynamo Per-Call Overhead

컴파일된 작은 영역이 많은 모델에서는 그래프 품질보다 고정된 호출당 비용이 더 중요합니다. 이번 릴리즈는 그 비용을 여러 곳에서 줄입니다. `compile_wrapper`는 매 호출마다 발생하던 `DispatchKeySet` pybind 오버헤드를 피하고, `torch._dynamo.disable`은 더 저렴한 경로를 갖게 됐으며, pregraph 프로파일러 마커는 항상 방출되는 대신 프로파일러가 활성화된 경우에만 방출됩니다. 사용되지 않는 함수 입력에 대해서는 가드 생성을 건너뛰고, `invoke_subgraph` 재사용 조회는 dataclass나 namedtuple 같은 pytree 인자에 대해 더 빨라졌습니다. 이런 미세 최적화가 모여 컴파일된 코드로 진입할 때 드는 고정 비용을 낮추어, 컴파일된 작은 영역과 즉시 실행(eager)이 뒤섞인 실제 모델에서도 `torch.compile`을 더 실용적으로 만듭니다.
> For models with many small compiled regions, fixed per-call cost matters more than graph quality. This release trims that cost in several places. `compile_wrapper` avoids `DispatchKeySet` pybind churn on every call, `torch._dynamo.disable` gets a cheaper path, and the pregraph profiler marker is gated on an active profiler instead of always being emitted. Guard creation is skipped for unused function inputs, and `invoke_subgraph` reuse lookup is faster for pytree arguments such as dataclasses and namedtuples. These micro-optimizations collectively lower the tax of entering compiled code, making `torch.compile` more practical for real-world models that mix many small compiled regions with eager execution.

API 안정성: Unstable
> API Unstable

(PR [#190390](https://github.com/pytorch/pytorch/pull/190390), [#190392](https://github.com/pytorch/pytorch/pull/190392), [#190623](https://github.com/pytorch/pytorch/pull/190623) 작성: William Wen, Meta, [#187782](https://github.com/pytorch/pytorch/pull/187782) 및 [#191817](https://github.com/pytorch/pytorch/pull/191817) 작성: Aditya Sanjeev)
> (PR [#190390](https://github.com/pytorch/pytorch/pull/190390), [#190392](https://github.com/pytorch/pytorch/pull/190392) and [#190623](https://github.com/pytorch/pytorch/pull/190623) by William Wen, Meta, [#187782](https://github.com/pytorch/pytorch/pull/187782) and [#191817](https://github.com/pytorch/pytorch/pull/191817) by Aditya Sanjeev)

### 즉시 실행 디스패치와 CPU 커널 / Eager Dispatch and CPU Kernels

즉시 실행(eager) 모드에서 자주 실행되는 경로(hot path) 여럿이 저렴해졌습니다. PyObject 디스패치가 최적화됐고, AOTAutograd는 역방향을 위해 그래프 입력 뷰를 저장할 때 비싼 `Tensor.detach()`를 피하며, Autograd는 프로파일러가 꺼져 있을 때 `at::Tensor`를 복사하지 않고, `addmm`은 C와 D가 서로 다를 때 디바이스 간 복사를 피하며, CPU의 `quantile`과 `nanquantile`은 전체 정렬 대신 부분 선택을 사용합니다.
> Several eager-mode hot paths got cheaper. PyObject dispatch is optimized, AOTAutograd avoids an expensive `Tensor.detach()` when saving graph-input views for backward, autograd stops copying `at::Tensor` when the profiler is off, `addmm` avoids a device-to-device copy when C and D are distinct, and CPU `quantile` and `nanquantile` use partial selection instead of a full sort.

이렇게 표적을 정한 수정들이 즉시 실행 모드의 연산당 비용을 줄여 성능 하한선을 끌어올립니다. 그 결과 선형 계층, Autograd 기록, 통계 집계 같은 흔한 연산이 불필요한 오버헤드를 지지 않게 됩니다. 사용자가 굳이 `torch.compile`을 꺼내 쓰지 않아도 PyTorch의 기본 개발 경험이 빠르게 유지됩니다.
> These targeted fixes reduce the per-operation tax in eager mode, tightening the performance floor so that common operations like linear layers, autograd bookkeeping, and statistical aggregations don't carry unnecessary overhead. They keep PyTorch's default development experience fast without requiring users to reach for `torch.compile`.

API 안정성: Unstable
> API Unstable

(PR [#187949](https://github.com/pytorch/pytorch/pull/187949), [#189759](https://github.com/pytorch/pytorch/pull/189759), [#189582](https://github.com/pytorch/pytorch/pull/189582) 작성: Richard Zou, Meta, [#191706](https://github.com/pytorch/pytorch/pull/191706) 작성: Animesh Jain, Meta, [#188394](https://github.com/pytorch/pytorch/pull/188394) 작성: Kimon N.)
> (PR [#187949](https://github.com/pytorch/pytorch/pull/187949) [#189759](https://github.com/pytorch/pytorch/pull/189759) and [#189582](https://github.com/pytorch/pytorch/pull/189582) by Richard Zou, Meta, [#191706](https://github.com/pytorch/pytorch/pull/191706) by Animesh Jain, Meta, and [#188394](https://github.com/pytorch/pytorch/pull/188394) by Kimon N.)

## 핵심 기능 / Core Features

### torch.linalg.polar와 torch.linalg.matrix\_sqrth / torch.linalg.polar and torch.linalg.matrix\_sqrth

`torch.linalg`에 두 가지가 추가됐습니다. `torch.linalg.polar`는 cuSOLVER의 QDWH 알고리즘으로 극분해(polar decomposition)를 계산하며, CPU, CUDA, MPS에서 역방향 공식을 제공하여 분석용으로만이 아니라 학습 루프 안에서도 쓸 수 있습니다. `torch.linalg.matrix_sqrth`는 대칭 및 에르미트(Hermitian) 양의 정부호 행렬에 대한 행렬 제곱근을 계산하는데, 이전에는 고윳값 분해(eigendecomposition)를 손으로 조합해야 했던 경우입니다.
> Two additions to `torch.linalg`. `torch.linalg.polar` computes the polar decomposition using cuSOLVER's QDWH algorithm, with a backward formula on CPU, CUDA and MPS, which makes it usable inside training loops rather than only for analysis. `torch.linalg.matrix_sqrth` computes the matrix square root for symmetric and Hermitian positive-definite matrices, a case that previously required composing an eigendecomposition by hand.

API 안정성: Unstable
> API Unstable

(PR [#185837](https://github.com/pytorch/pytorch/pull/185837) 작성: Simon Layton, Meta, [#189732](https://github.com/pytorch/pytorch/pull/189732) 작성: Irakli Salia, Hugging Face, [#187987](https://github.com/pytorch/pytorch/pull/187987) 작성: Colin Alberts, Cisco)
> (PR [#185837](https://github.com/pytorch/pytorch/pull/185837) by Simon Layton, Meta, [#189732](https://github.com/pytorch/pytorch/pull/189732) by Irakli Salia, Hugging Face, [#187987](https://github.com/pytorch/pytorch/pull/187987) by Colin Alberts, Cisco)

### Autograd 확장 지점 / Autograd Extension Points

세 가지가 추가되어 Autograd 그래프를 구성하고 들여다보는 방식을 더 세밀하게 제어할 수 있게 됐습니다. `torch.autograd.graph.node_creation_hook`은 Autograd 노드가 생성될 때마다 발동하여, 도구가 나중에 그 맥락을 재구성하는 대신 그래프 구성 시점에 메타데이터를 붙이거나 훅을 등록할 수 있게 합니다. 이를 만든 계기가 된 사례는 역방향 패스의 메모리 사용량을 그 메모리를 만들어낸 순방향 영역에 귀속시키는 것입니다. `ctx.set_output_grad_dtype`은 커스텀 `autograd.Function`이 출력에 들어오는 변화도의 dtype을, 출력 자체의 저장 dtype과 무관하게 선언할 수 있게 해줍니다. 둘이 일치하지 않는 혼합 정밀도 함수를 위한 것입니다. 이제 `cdist`와 `pdist`에 이중 역전파(double backward)가 구현되어, 이전에는 아예 실패하던 `create_graph=True` 사용, 즉 쌍별 거리 계산을 거치는 헤시안(Hessian), 변화도 페널티, 헤시안-벡터 곱이 가능해졌습니다.
> Three additions give more control over how the autograd graph is built and inspected. `torch.autograd.graph.node_creation_hook` fires as each autograd node is created, letting tools attach metadata or register hooks at graph-construction time instead of reconstructing that context afterward — the motivating case is attributing backward-pass memory usage back to the forward region that produced it. `ctx.set_output_grad_dtype` lets a custom `autograd.Function` declare the dtype its output's incoming gradient should be, independent of the output's own storage dtype, for mixed-precision functions where the two don't match. Double backward is now implemented for `cdist` and `pdist`, unblocking `create_graph=True` uses that previously failed outright — Hessians, gradient penalties, and Hessian-vector products through pairwise-distance computations.

API 안정성: Unstable
> API Unstable

(PR [#189284](https://github.com/pytorch/pytorch/pull/189284) 작성: Edward Yang, Meta, [#189634](https://github.com/pytorch/pytorch/pull/189634) 작성: @SongyuanZhao, [#188901](https://github.com/pytorch/pytorch/pull/188901) 작성: Colin Alberts, Cisco)
> (PR [#189284](https://github.com/pytorch/pytorch/pull/189284) by Edward Yang, Meta, [#189634](https://github.com/pytorch/pytorch/pull/189634) by @SongyuanZhao, [#188901](https://github.com/pytorch/pytorch/pull/188901) by Colin Alberts, Cisco)

### torch.switch 고차 연산 / torch.switch Higher-Order Op

`torch.cond`는 두 갈래 분기를 표현하므로 n-way 디스패치는 중첩된 조건문으로 작성해야 했고, 이는 추적된 그래프를 키우고 의도를 흐렸습니다. `torch.switch`는 인덱스에 따라 여러 갈래로 분기하는 새로운 고차 연산(higher-order op)으로, Dynamo에서 리프팅된 인자의 중복 제거를 지원해 공유되는 피연산자가 분기마다 다시 리프팅되지 않습니다. 그 결과 다중 분기가 있는 모델, 특히 `torch.cond` 중첩이 현실적인 장벽이었던 전문가 혼합(mixture-of-experts) 아키텍처를 더 표현력 있고 효율적으로 추적할 수 있습니다.
> `torch.cond` expresses a two-way branch, so an n-way dispatch had to be written as nested conditionals, which grows the traced graph and obscures intent. `torch.switch` is a new higher-order op for multi-way branching on an index, with lifted-argument deduplication in Dynamo so shared operands are not re-lifted per branch. The result is a more expressive and efficient way to trace models with multi-way branching, particularly mixture-of-experts architectures where torch.cond nesting was a practical barrier.

API 안정성: Unstable
> API Unstable

(PR [#182902](https://github.com/pytorch/pytorch/pull/182902) 및 [#188374](https://github.com/pytorch/pytorch/pull/188374) 작성: Thomas Ortner, IBM)
> (PR [#182902](https://github.com/pytorch/pytorch/pull/182902) and [#188374](https://github.com/pytorch/pytorch/pull/188374) by Thomas Ortner, IBM)

### 랭크 3 입력에 대한 SDPA 융합 백엔드 / SDPA Fused Backends for Rank-3 Inputs

이제 스케일링된 내적 어텐션(scaled dot-product attention)이 랭크 3 입력에 대해 math 경로로 폴백하는 대신 융합된 CUDA 백엔드로 디스패치합니다. 따라서 배치 차원이 없거나 이미 평탄화된 텐서를 넘기는 호출자도 reshape 없이 융합 커널을 사용할 수 있습니다. 이 수정은 배치 차원이 없거나 평탄화되었을 때 조용히 빠른 융합 커널을 우회하던 흔한 성능 함정을 없앱니다.
> Scaled dot-product attention now dispatches to the fused CUDA backends for rank-3 inputs instead of falling back to the math path, so callers who pass unbatched or already-flattened tensors get the fused kernels without reshaping. The fix closes a common performance trap where missing or flattened batch dimensions silently bypassed the fast fused kernels.

API 안정성: Unstable
> API Unstable

(PR [#192271](https://github.com/pytorch/pytorch/pull/192271) 작성: Driss Guessous, Meta)
> (PR [#192271](https://github.com/pytorch/pytorch/pull/192271) by Driss Guessous, Meta)

### 복소수 값 텐서에 대한 실험적 torch.compile 지원 / Experimental torch.compile support for complex-valued tensors

복소수 값 텐서를 사용하는 프로그램에 대한 torch.compile 지원입니다. 지원되는 복소수 연산은 컴파일러 백엔드가 최적화할 수 있는 실수 값 계산으로 분해됩니다. 이를 통해 신호 처리, 과학 계산, 복소수 값 신경망 등 더 많은 복소수 워크로드가 컴파일된 실행의 이점을 누릴 수 있습니다. 아직 모든 복소수 연산이 지원되지는 않습니다. [기능 추적 이슈](https://github.com/pytorch/pytorch/issues/194061)와 구현을 참고하세요.
> torch.compile support for programs using complex-valued tensors. Supported complex operations are decomposed into real-valued computations that compiler backends can optimize. This enables more complex-number workloads including signal processing, scientific computing, and complex-valued neural networks to benefit from compiled execution. Not all complex operations are supported yet. See the [feature tracking issue](https://github.com/pytorch/pytorch/issues/194061), implementation

(PR [#167621](https://github.com/pytorch/pytorch/pull/167621), [#169832](https://github.com/pytorch/pytorch/pull/169832), [#172813](https://github.com/pytorch/pytorch/pull/172813) 작성: Hameer Abbasi, OpenTeams)
> (PRs [#167621](https://github.com/pytorch/pytorch/pull/167621) and [#169832](https://github.com/pytorch/pytorch/pull/169832),  [#172813](https://github.com/pytorch/pytorch/pull/172813) by Hameer Abbasi, OpenTeams)

### 그 밖의 작은 API 추가 / Smaller API Additions

이번 릴리즈에는 공개 API에 작은 추가들도 여럿 반영됐습니다.
> A number of smaller public additions landed this release.

- `torch.utils.checkpoint.checkpoint`이 즉시 실행 모드에서 데코레이터와 커링(curried) 호출 방식을 지원합니다([#189411](https://github.com/pytorch/pytorch/pull/189411) 작성: Edward Yang, Meta).
- 읽기 전용 DLPack 내보내기와 `ReadOnlyTensorWrapper`가 추가되어, 소비자에게 변경해서는 안 되는 텐서를 넘길 수 있습니다([#188554](https://github.com/pytorch/pytorch/pull/188554) 작성: Edward Yang, Meta).
- `Generator.philox_state`가 Philox RNG 상태 예약을 Python에 노출합니다([#191019](https://github.com/pytorch/pytorch/pull/191019) 작성: Simon Layton, Meta).
- `torch.accelerator`에 `initial_seed`, `get_rng_state`, `get_rng_state_all`이 추가되어 CUDA 전용 RNG API와의 격차를 일부 좁혔습니다([#186597](https://github.com/pytorch/pytorch/pull/186597) 작성: Guangye Yu, Intel).
- `LBFGS`에 `maximize`가 추가되고, 매개변수 그룹이 비어 있으면 아무 동작도 하지 않습니다([#187309](https://github.com/pytorch/pytorch/pull/187309) 작성: Raj Vijay Firke, Red Hat).
- 2.13에서 도입된 `linear_cross_entropy`가 청크(chunked) 경로에서 확률 타겟을 지원합니다([#187053](https://github.com/pytorch/pytorch/pull/187053) 작성: Pearu Peterson, Quansight).
- `c10::utils::get_env`와 `set_env`가 Python에 노출됩니다([#191015](https://github.com/pytorch/pytorch/pull/191015) 작성: Nikita Shulga, Thinking Machines Lab).

> - `torch.utils.checkpoint.checkpoint` accepts a decorator and curried calling convention in eager ([#189411](https://github.com/pytorch/pytorch/pull/189411) by Edward Yang, Meta).
> - Read-only DLPack export and `ReadOnlyTensorWrapper`, so consumers can be handed a tensor they must not mutate ([#188554](https://github.com/pytorch/pytorch/pull/188554) by Edward Yang, Meta).
> - `Generator.philox_state` exposes Philox RNG state reservation to Python ([#191019](https://github.com/pytorch/pytorch/pull/191019) by Simon Layton, Meta).
> - `torch.accelerator` gains `initial_seed`, `get_rng_state` and `get_rng_state_all`, closing part of the gap with the CUDA-specific RNG APIs ([#186597](https://github.com/pytorch/pytorch/pull/186597) by Guangye Yu, Intel).
> - `LBFGS` gains `maximize` and is a no-op on an empty parameter group ([#187309](https://github.com/pytorch/pytorch/pull/187309) by Raj Vijay Firke, Red Hat).
> - `linear_cross_entropy`, introduced in 2.13, supports probability targets on the chunked path ([#187053](https://github.com/pytorch/pytorch/pull/187053) by Pearu Peterson, Quansight).
> - `c10::utils::get_env` and `set_env` are exposed to Python ([#191015](https://github.com/pytorch/pytorch/pull/191015) by Nikita Shulga, Thinking Machines Lab).

API 안정성: Unstable
> API Unstable

### Python 3.15 지원과 Torchvision ABI 안정성 – 릴리즈 엔지니어링 / Python 3.15 Support and Torchvision ABI Stability – Release Engineering

PyTorch 2.14는 모든 플랫폼에서 자유 스레드(free-threaded, no-GIL) 빌드인 3.15t를 포함해 Python 3.15에 대한 바이너리 지원을 추가합니다. 휠은 x86\_64와 aarch64의 Linux, Windows, Apple Silicon의 macOS에 대해 CPU, CUDA, ROCm, XPU 빌드를 아울러 게시됩니다. 또한 torchvision 0.29.0도 같은 플랫폼 집합에 대해 3.15와 3.15t 휠을 함께 제공합니다.
> PyTorch 2.14 adds binary support for Python 3.15, including the free-threaded (no-GIL) build, 3.15t across all platforms. Wheels are published for Linux on x86\_64 and aarch64, Windows, and macOS on Apple silicon, spanning the CPU, CUDA, ROCm, and XPU builds. Also torchvision 0.29.0 ships matching 3.15 and 3.15t wheels for the same set of platforms.

이제 TorchVision이 torch 2.14에 대해 ABI 안정성을 갖습니다! 즉, torchvision 0.29는 앞으로 나올 torch 2.15, 2.16 등과도 호환됩니다. torch를 업그레이드할 때 TorchVision을 새 버전으로 설치할 필요가 없습니다. 그 결과, TorchVision을 pytorch와 같은 주기로 릴리즈하지 않게 될 수도 있습니다. 다만 TorchVision은 여전히 활발히 유지보수되고 개발됩니다. 릴리즈는 계속 내되, 같은 주기는 아닐 뿐입니다.
> TorchVision is now ABI stable w.r.t. torch 2.14! This means that torchvision 0.29 will be compatible with future versions of torch: 2.15, 2.16, etc. You won't need to install a new version of TorchVision when you upgrade torch. As a result, we might stop releasing TorchVision in sync with pytorch. But TorchVision is still actively maintained and developed: we'll still be pushing releases, just not with the same cadence.

### 설치 / Installation

Python 3.15 및 3.15t 휠은 PyPI에 게시되지 않습니다 — 오직 download.pytorch.org를 통해서만 다음 명령 중 하나로 다운로드할 수 있습니다:
> Python 3.15 and 3.15t wheels are not published to PyPI — they are available to download only via download.pytorch.org, using any of the following commands:

```sh
# CPU
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA (CUDA 버전으로 치환, 예: cu126 / cu130)
pip3 install torch --index-url https://download.pytorch.org/whl/cu130

# ROCm (ROCm 버전으로 치환)
pip3 install torch --index-url https://download.pytorch.org/whl/rocm7.14

# XPU
pip3 install torch --index-url https://download.pytorch.org/whl/xpu
```

동일한 명령을 자유 스레드 인터프리터에서 실행하면 자유 스레드 3.15t 빌드가 설치됩니다.
> The same commands install the free-threaded 3.15t build when run under a free-threaded interpreter.

자유 스레드 빌드에도 같은 내용이 적용됩니다. 3.15t 인터프리터에 설치하면 pip이 cp315t 휠을 자동으로 찾아 설치합니다.
> The same applies to the free-threaded build. Install into a 3.15t interpreter and pip will resolve the cp315t wheels automatically.

### torch.compile은 아직 Python 3.15를 지원하지 않습니다 / torch.compile is not yet supported on Python 3.15

2.14의 Python 3.15 지원은 즉시 실행 전용입니다. Python 3.15에서 torch.compile을 호출하면 조용히 폴백하는 대신 RuntimeError가 발생하므로, 제약이 조용한 성능 손실이 아니라 즉시 드러납니다. 워크로드가 torch.compile에 의존한다면 당분간 Python 3.14 이하에 머무르세요.
> Python 3.15 support in 2.14 is eager-only. Calling torch.compile under Python 3.15 raises a RuntimeError rather than falling back silently, so the limitation surfaces immediately rather than as a silent performance loss. If your workload depends on torch.compile, stay on Python 3.14 or earlier for now.

3.15에 대한 Dynamo 지원은 활발히 개발 중이며, 새 인터프리터를 위한 바이트코드 및 심볼릭 변환 처리는 이미 반영됐습니다. 진행 상황은 [pytorch/pytorch#184352](https://github.com/pytorch/pytorch/issues/184352)에서 추적됩니다.
> Dynamo support for 3.15 is in active development, with bytecode and symbolic-conversion handling already landed for the new interpreter. Progress is tracked in [pytorch/pytorch#184352](https://github.com/pytorch/pytorch/issues/184352).

## 분산 학습 / Distributed Training

### nccl2 백엔드 / nccl2 Backend

torchcomms는 2.13에서 PyTorch Distributed의 CI와 디바이스 메시(device-mesh) 경로에 통합된 통신 백엔드로 등장했습니다. 2.14에서는 그 API가 `USE_C10D_NCCL`로 게이트되는 새로운 `nccl2` c10d 백엔드와 함께 트리 안으로 들어왔고, 재사용 가능한 `NcclApi` 추상화 위에 완전한 `Work` 계약을 구현합니다. 이 백엔드는 즉시 실행 전용이며, 단방향 윈도우, 내결함성, 일시 중단 및 재개 시의 메모리 오프로드 같은 새로운 기능과 크게 정리된 구현을 갖췄습니다. 호환용 `nccl-lazy` 래퍼는 예전의 지연 초기화 동작이 필요한 워크로드를 위해 피어별 P2P 커뮤니케이터를 필요할 때 만들어 줍니다.
> torchcomms arrived in 2.13 as a communications backend integrated into PyTorch Distributed's CI and device-mesh paths. In 2.14 the APIs landed in-tree with a new `nccl2` c10d backend, gated behind `USE_C10D_NCCL`, implementing the full `Work` contract on top of a reusable `NcclApi` abstraction. The backend is eager only with new features such as one-sided windows, fault-tolerance, suspend and resume memory offload, and a greatly cleaned up implementation. A compatibility `nccl-lazy` wrapper builds per-peer P2P communicators on demand for workloads that require the old lazy initialization behavior.

API 안정성: Unstable
> API Unstable

(PR [#188582](https://github.com/pytorch/pytorch/pull/188582), [#189359](https://github.com/pytorch/pytorch/pull/189359), [#190943](https://github.com/pytorch/pytorch/pull/190943), [#191272](https://github.com/pytorch/pytorch/pull/191272) 작성: Tristan Rice, Meta, [#191528](https://github.com/pytorch/pytorch/pull/191528) 및 [#192105](https://github.com/pytorch/pytorch/pull/192105) 작성: Tushar Jain, Meta)
> (PR [#188582](https://github.com/pytorch/pytorch/pull/188582), [#189359](https://github.com/pytorch/pytorch/pull/189359), [#190943](https://github.com/pytorch/pytorch/pull/190943) and [#191272](https://github.com/pytorch/pytorch/pull/191272) by Tristan Rice, Meta, [#191528](https://github.com/pytorch/pytorch/pull/191528) and [#192105](https://github.com/pytorch/pytorch/pull/192105) by Tushar Jain, Meta)

### c10d의 내결함성 집합 통신 / Fault-Tolerant Collectives in c10d

대규모 작업에서 랭크 하나가 실패하면 보통은 프로세스 그룹을 해체하고 다시 시작하는 식으로 복구하는데, 이러면 클러스터 전체의 예열된 상태가 버려집니다. 이제 `Backend`와 `ProcessGroup`이 재구성 인터페이스를 노출하여 그룹을 제자리에서 다시 만들 수 있고, 중단(abort) 훅과 집합 통신 전후 훅이 같은 경로로 연결됩니다. Gloo도 nccl2와 함께 내결함성 지원을 갖추게 됐고, 재구성 API는 문서화되어 있습니다.
> When a rank fails in a large job, the usual recovery is to tear down the process group and restart, which discards warm state across the whole cluster. `Backend` and `ProcessGroup` now expose reconfiguration interfaces so a group can be rebuilt in place, with abort hooks and pre and post collective hooks wired through the same path. Gloo gains fault-tolerance support alongside nccl2, and the reconfigure APIs are documented.

API 안정성: Unstable
> API Unstable

(PR [#186298](https://github.com/pytorch/pytorch/pull/186298), [#186300](https://github.com/pytorch/pytorch/pull/186300), [#187381](https://github.com/pytorch/pytorch/pull/187381), [#191384](https://github.com/pytorch/pytorch/pull/191384) 작성: Tristan Rice, Meta)
> (PR [#186298](https://github.com/pytorch/pytorch/pull/186298), [#186300](https://github.com/pytorch/pytorch/pull/186300), [#187381](https://github.com/pytorch/pytorch/pull/187381) and [#191384](https://github.com/pytorch/pytorch/pull/191384) by Tristan Rice, Meta)

### 단방향(RMA) 윈도우 API / One-Sided (RMA) Window APIs

`Backend`와 `ProcessGroup`에 단방향 윈도우 인터페이스가 추가되어, 기존의 양방향 집합 통신과 함께 원격 메모리 접근(remote-memory-access) 의미론을 제공합니다. 단방향 연산에서는 상대 랭크가 대응하는 호출을 걸지 않아도 그 랭크의 메모리를 읽거나 쓸 수 있어, 임베딩 조회, 가중치 전송, 전문가 라우팅처럼 불규칙한 접근 패턴에 적합합니다. 이로써 nccl2 백엔드를 통해 새로운 ncclGet 및 ncclPut API가 노출됩니다.
> `Backend` and `ProcessGroup` gain one-sided window interfaces, giving remote-memory-access semantics alongside the existing two-sided collectives. One-sided operations let a rank read or write peer memory without the peer posting a matching call, which suits irregular access patterns such as embedding lookups, weight transfer and expert routing. This exposes the new ncclGet and ncclPut APIs via the nccl2 backend.

API 안정성: Unstable
> API Unstable

(PR [#186299](https://github.com/pytorch/pytorch/pull/186299) 및 [#189360](https://github.com/pytorch/pytorch/pull/189360) 작성: Tristan Rice, Meta)
> (PR [#186299](https://github.com/pytorch/pytorch/pull/186299) and [#189360](https://github.com/pytorch/pytorch/pull/189360) by Tristan Rice, Meta)

### 백엔드에 구애받지 않는 Flight Recorder / Backend-Agnostic Flight Recorder

멈춤(hang)과 어긋난 집합 통신을 진단하는 데 쓰이는 집합 통신 추적 버퍼인 Flight Recorder는 NCCL에 묶여 있었습니다. 이제 `FlightRecorderHook`이 `ProcessGroup` 훅을 통해 기록하므로 어떤 백엔드에서도 동작하며, 로그 직렬화도 `DebugMode`를 통해 이식 가능합니다. Gloo나 커스텀 백엔드로 돌아가는 작업을 디버깅할 때 더 이상 추적을 포기하지 않아도 됩니다.
> Flight Recorder, the collective trace buffer used to diagnose hangs and mismatched collectives, was tied to NCCL. `FlightRecorderHook` records through `ProcessGroup` hooks instead, so it works for any backend, and log serialization is portable through `DebugMode`. Debugging a Gloo or custom-backend job no longer means giving up the trace.

API 안정성: Unstable
> API Unstable

(PR [#189363](https://github.com/pytorch/pytorch/pull/189363) 작성: Tristan Rice, Meta, [#185010](https://github.com/pytorch/pytorch/pull/185010) 작성: Jason Ansel, Meta)
> (PR [#189363](https://github.com/pytorch/pytorch/pull/189363) by Tristan Rice, Meta, [#185010](https://github.com/pytorch/pytorch/pull/185010) by Jason Ansel, Meta)

### 플러그인 방식의 분산 백엔드 / Pluggable Distributed Backends

이전에는 통신 백엔드를 추가하려면 c10d에 패치를 넣어야 했습니다. 이제 백엔드는 Python 엔트리 포인트를 통해 등록할 수 있고, 백엔드 문자열은 자동으로 정규화되며, 구현 접근자도 노출됩니다. `PyProcessGroup` 트램폴린(trampoline)을 C++와 동등한 수준으로 끌어올려, 트리 외(out-of-tree) 백엔드가 C++에서든 Python에서든 `batch_isend_irecv`, 병합(coalescing) 매니저, 정리된 \*\_single 변형을 포함한 전체 집합 통신 표면을 구현할 수 있습니다.
> Adding a communications backend previously meant patching c10d. Backends can now register through Python entry points, backend strings are auto-qualified, and implementation accessors are exposed. We've brought the `PyProcessGroup` trampoline to parity with C++, so an out-of-tree backend can implement the full collective surface from either C++ or Python, including `batch_isend_irecv`, the coalescing manager and the cleaned up \*\_single variants.

API 안정성: Unstable
> API Unstable

(PR [#187388](https://github.com/pytorch/pytorch/pull/187388), [#186853](https://github.com/pytorch/pytorch/pull/186853), [#188570](https://github.com/pytorch/pytorch/pull/188570) 작성: Tristan Rice, Meta, [#187494](https://github.com/pytorch/pytorch/pull/187494) 작성: Kapil Sharma, Meta)
> (PR [#187388](https://github.com/pytorch/pytorch/pull/187388), [#186853](https://github.com/pytorch/pytorch/pull/186853) and [#188570](https://github.com/pytorch/pytorch/pull/188570) by Tristan Rice, Meta, [#187494](https://github.com/pytorch/pytorch/pull/187494) by Kapil Sharma, Meta*)*

### torch.distributed API 개선: set\_timeout, 연산별 타임아웃, get\_backend\_impl, 훅, weights\_only=True, \*\_single / torch.distributed API improvements: set\_timeout, per-op timeouts, get\_backend\_impl, hooks, weights\_only=True, \*\_single

더 세밀한 제어를 가능하게 하고 일부 불일치를 정리하는 다양한 개선을 torch.distributed API에 반영했습니다. 안정 API인 `torch.distributed.set_timeout` 메서드를 통해 초기화 이후에도 프로세스 그룹의 집합 통신 타임아웃을 바꿀 수 있습니다. 느린 체크포인트 로드 구간에서는 늘리고, 반대로 줄여서 멈춰 버린 랭크가 기본 대기 시간을 다 채우지 않고 빠르게 실패하도록 할 수 있습니다. 여기에 더해 모든 타임아웃에 걸쳐 연산별 집합 통신도 지원합니다. `torch.distributed.get_backend_impl`을 통해 백엔드별 고급 기능에 접근하기 쉬워졌고, 동작을 커스터마이즈하고 관측 가능성을 확보하기 위한 프로그램적 훅도 추가했습니다. 객체 집합 통신은 이제 torch.load와 동일한 `weights_only=True` 모드를 지원하여 학습 클러스터의 보안을 높일 수 있습니다. 또한 `all_to_all_single`처럼 단일 텐서 변형의 이름을 모두 `_single` 접미사를 공유하도록 정리했습니다.
> We've made a whole host of improvements to the torch.distributed API which allow for more control as well as cleaning up some inconsistencies. You can change a process group's collective timeout after initialization — extending it around a slow checkpoint load, or shortening it so a wedged rank fails fast instead of hanging for the full default window — via the stable `torch.distributed.set_timeout` method as well as we now support per-operation collectives across all timeouts. We've made it easier to access advanced backend specific features via `torch.distributed.get_backend_impl` as well as add programmatic hooks to them to customize behavior and for observability. Object collectives now support the same `weights_only=True` mode as torch.load which can improve your training cluster security. We've also updated the names for all single tensor variants to share the `_single` suffix such as in `all_to_all_single`.

API 안정성: Unstable
> API Unstable

(PR [#187387](https://github.com/pytorch/pytorch/pull/187387) 및 [#187693](https://github.com/pytorch/pytorch/pull/187693) 작성: Tristan Rice, Meta)
> (PR [#187387](https://github.com/pytorch/pytorch/pull/187387) and [#187693](https://github.com/pytorch/pytorch/pull/187693) by Tristan Rice, Meta)

### DTensor 단일 차원 샤딩 전략 / DTensor Single-Dim Sharding Strategies

DTensor의 샤딩(sharding) 규칙은 역사적으로 디바이스 메시 전체를 대상으로 연산자마다 작성되어, 각 규칙이 모든 메시 차원에 걸친 배치(placement)의 모든 조합을 열거해야 했습니다. 작성하는 데 오래 걸리고, 메시 차원이 둘 이상이 되면 미묘하게 틀리기도 쉬웠습니다. 이번 릴리즈는 연산 지원을 단일 차원 전략 함수로 옮기는 작업을 이어갑니다. 이 함수는 하나의 메시 차원이 연산자를 어떻게 샤딩하는지만 기술하고, 이를 메시 전체로 확장하는 일은 프레임워크에 맡깁니다. 여기서는 행렬, 수학, 텐서 연산이 전환되어 직접적인 `register_op_strategy` 등록이 158개에서 114개로 줄었습니다. 합성곱도 윈도우가 마지막 공간 차원을 정확히 타일링할 때(패딩 없음, 팽창(dilation) 1, 스트라이드가 커널 너비와 같음, 차원이 커널 너비 × 메시 크기로 나누어떨어짐) 그 차원에 대한 샤딩을 지원하므로, 해당 합성곱은 복제를 위해 allgather 하는 대신 순방향과 역방향 모두 로컬에서 실행됩니다. 새 규칙은 대체된 기존 규칙보다 엄격해서, 이전에는 우연히 기본 전략과 맞아떨어지던 어노테이션(annotation)이 이제는 재분배가 필요한 것으로 보고됩니다. 그리고 이전된 연산에서는 더 이상 `Partial("product")`가 생성되지 않습니다. 이로써 샤딩 규칙이 등록된 전체 연산 수는 2026년 1월의 585개에서 1239개로 늘었습니다.
> DTensor's sharding rules were historically written per operator against the whole device mesh, so each rule had to enumerate every combination of placements across all mesh dimensions — long to write and easy to get subtly wrong once the mesh had more than one dimension. This release continues moving op coverage to single-dim strategy functions, which describe how one mesh dimension shards an operator and leave the framework to expand that across the full mesh; matrix, math, and tensor ops are converted here, cutting direct `register_op_strategy` registrations from 158 to 114. Convolution also gains sharding on the last spatial dimension when the windows tile it exactly — zero padding, dilation 1, stride equal to kernel width, and a dimension divisible by kernel width times mesh size — so those convolutions run locally in forward and backward instead of allgathering to replicate. The new rules are stricter than the ones they replace, so annotations that previously matched a base strategy by accident will now be reported as needing redistribution, and `Partial("product")` is no longer produced by the migrated ops. This brings the number of total ops with registered sharding rules to 1239, up from 585 in Jan 2026.

API 안정성: Unstable
> API Unstable

(PR [#186667](https://github.com/pytorch/pytorch/pull/186667), [#179203](https://github.com/pytorch/pytorch/pull/179203), [#186754](https://github.com/pytorch/pytorch/pull/186754), [#192147](https://github.com/pytorch/pytorch/pull/192147) 작성: Anshul Sinha, Meta)
> (PR [#186667](https://github.com/pytorch/pytorch/pull/186667), [#179203](https://github.com/pytorch/pytorch/pull/179203), [#186754](https://github.com/pytorch/pytorch/pull/186754), and [#192147](https://github.com/pytorch/pytorch/pull/192147) by Anshul Sinha, Meta)

### 대칭 메모리: NCCL 백엔드 수정과 할당 레이아웃 / Symmetric Memory: NCCL backend fixes and allocation layout

대칭 메모리(symmetric memory)의 NCCL 백엔드에는 실행 시점에만 드러나는 구멍이 있었습니다. `barrier()`가 미구현 오류를 냈고, 시그널 패드(signal pad)가 할당 후에도 0으로 초기화되지 않아 시그널링 프로토콜이 신뢰할 만한 기반을 갖지 못했습니다. 두 가지 모두 수정됐습니다. 이제 barrier는 기존 CUDA barrier 커널을 재사용하고, 패드는 할당 시점에 0으로 초기화됩니다. 또한 시그널 패드가 세 백엔드(CUDA, NCCL, NVSHMEM) 모두에서 대칭 할당의 맨 앞으로 옮겨져, 재활용되거나 크기가 바뀐 할당이 오염된 패드를 물려받을 수 없고, 블록 전체가 아니라 패드만 0으로 초기화되므로 큰 할당이 `alloc()`마다 버퍼 전체를 memset 하는 비용을 더 이상 치르지 않습니다. 시그널 패드를 언제 0으로 초기화하는지는 여러 스트림이 얽힌 상황에서 까다로울 수 있어서, 순서를 확실히 하기 위해 사용자 공간 코드에서 직접 처리하고 싶을 수도 있습니다. CUDA 할당은 드라이버가 지원을 보고하면 GPUDirect RDMA 지원 플래그를 설정합니다. 시그널 패드 슬롯은 같은 할당에 대해 여전히 여러 프로세스 그룹이 공유하므로, 겹치는 그룹에서 동시에 barrier를 걸면 서로 간섭할 수 있습니다.
> Symmetric memory's NCCL backend had gaps that only surfaced at runtime: `barrier()` raised a not-implemented error, and the signal pad was never zeroed after allocation, so the signaling protocol had nothing reliable to build on. Both are fixed — barrier now reuses the existing CUDA barrier kernel, and the pad is zeroed at allocation time. The signal pad also moves to the front of every symmetric allocation across all three backends (CUDA, NCCL, NVSHMEM), so a recycled or resized allocation can't inherit a polluted pad, and only the pad gets zeroed rather than the whole block — large allocations no longer pay a full-buffer memset on every `alloc()`. The timing of when the signal pad is zeroed can be tricky in multiple stream scenarios and users may want to do that in user space code to be sure that they have the sequence right. CUDA allocations now set the GPUDirect RDMA capable flag when the driver reports support. Signal-pad slots are still shared across process groups on the same allocation, so concurrent barriers from overlapping groups can interfere.

API 안정성: Unstable
> API Unstable

(PR [#188051](https://github.com/pytorch/pytorch/pull/188051) 작성: Kapil Sharma, Meta, [#189088](https://github.com/pytorch/pytorch/pull/189088) 작성: Junjie Wang, NVIDIA, [#189941](https://github.com/pytorch/pytorch/pull/189941) 작성: Natalia Gimelshein, Meta)
> (PRs [#188051](https://github.com/pytorch/pytorch/pull/188051) by Kapil Sharma, Meta, [#189088](https://github.com/pytorch/pytorch/pull/189088) by Junjie Wang, NVIDIA, #[189941](https://github.com/pytorch/pytorch/pull/189941) by Natalia Gimelshein, Meta)

### 대칭 메모리: 일반 집합 통신에서 NCCL 대칭 커널 쓰기 / Symmetric Memory: reaching NCCL symmetric kernels from ordinary collectives

대칭 메모리 기반 커널은 NCCL 2.27부터 NVLink 도메인에서 사용할 수 있게 됐습니다. 이 커널은 2026년 1월부터 nightly 빌드의 PyTorch 대칭 메모리에 구현돼 있었지만, 사용자가 쉽게 알아볼 수 있게 문서화되어 있지는 않았습니다. 피드백에 따라 이번 릴리즈에서 문서를 개선했습니다. 이제 문서는 대칭 메모리와 링(ring) 또는 트리(tree) 집합 통신을 모두 다룹니다. `register_mem_pool(pool, symm=True)`로 `torch.cuda.MemPool`을 등록하거나, `set_backend("NCCL")`과 랑데부(rendezvous)를 사용하는 방법, 그리고 적격 조건(`all_gather`는 모든 dtype에서 가능, `all_reduce`와 `reduce_scatter`는 `float64`를 제외한 float dtype에 대해 `SUM`/`AVG`만 가능)과 요구 사항(NCCL 2.27 이상, 단일 직접 NVLink 도메인, `NCCL_WIN_ENABLE`)이 함께 설명돼 있습니다. `NCCL_DEBUG_SUBSYS=TUNING` 아래의 `[Symmetric]` 태그나 프로파일에 나타나는 `ncclSymkDevKernel_*` 이름으로 확인할 수 있습니다.
> Symmetric memory based kernels became available on the NVLink domain in NCCL 2.27. These have been implemented in PyTorch's symmetric memory in nightly builds since Jan 2026 but weren't documented in a way that made it easy for users. Responding to feedback we improved the documentation in this release.  The docs now cover both symmetric memory and ring or tree collectives — registering a `torch.cuda.MemPool` with `register_mem_pool(pool, symm=True)`, or `set_backend("NCCL")` plus rendezvous — along with the eligibility rules (`all_gather` on any dtype; `all_reduce` and `reduce_scatter` only for `SUM`/`AVG` on float dtypes excluding `float64`) and the requirements: NCCL 2.27+, a single direct-NVLink domain, and `NCCL_WIN_ENABLE`. Users can verify via the `[Symmetric]` tag under `NCCL_DEBUG_SUBSYS=TUNING` or `ncclSymkDevKernel_*` names in a profile.

API 안정성: Unstable (PR [#192515](https://github.com/pytorch/pytorch/pull/192515) 작성: Kapil Sharma, Meta)
> API Unstable(PR [#192515](https://github.com/pytorch/pytorch/pull/192515) by Kapil Sharma, Meta)

#### 대칭 메모리: 단방향 `get` / Symmetric Memory: one-sided `get`

다른 랭크에 있는 데이터를 읽으려면 대체로 집합 통신이 필요했습니다. 실제로는 한 랭크만 그 데이터가 필요한데도 모든 랭크가 참여해 동기화해야 했던 것입니다. 이제 대칭 메모리가 `get`을 노출합니다. 이는 피어의 대칭 할당을 로컬 대상 텐서로 직접 읽어 오는 단방향 복사로, 피어의 참여도, 그룹 전체의 동기화도 필요 없습니다. NVSHMEM, NCCL 대칭 메모리, CUDA 백엔드에서 동작하며(XPU와 rocSHMEM은 아직 지원되지 않습니다), 원본이 랑데부된 대칭 할당이면서 dtype과 원소 개수가 대상과 일치해야 합니다. 진행 중인 단방향 DTensor 작업의 기반이 되는 프리미티브이며, 모든 랭크에 걸친 all-gather보다 한 피어에서 끌어오는 편이 나은 알고리즘이라면 어디든 바로 쓸 수 있습니다.
> Reading data that lives on another rank has generally meant a collective: every rank participates and synchronizes, even when only one rank actually needs the data. Symmetric memory now exposes `get`, a one-sided copy that reads a peer's symmetric allocation directly into a local destination tensor, with no participation from the peer and no group-wide synchronization. It works on the NVSHMEM, NCCL symmetric-memory, and CUDA backends — XPU and rocSHMEM aren't supported yet — and requires the source to be a rendezvoused symmetric allocation matching the destination in dtype and element count. It's the primitive underneath the in-progress one-sided DTensor work, and it's directly usable for any algorithm where pulling from one peer beats an all-gather across all of them.

API 안정성: Unstable (PR [#182378](https://github.com/pytorch/pytorch/pull/182378) 작성: Benjamin Brock, Intel)
> API Unstable(PR [#182378](https://github.com/pytorch/pytorch/pull/182378) by Benjamin Brock, Intel)

### TokenSwitch

전문가 혼합(mixture-of-experts) 학습은 스텝 시간의 상당 부분을 각 토큰을 그 토큰이 고른 전문가가 있는 랭크로 보내고 전문가의 출력을 다시 가져오는 데 씁니다. 그리고 팀들은 보통 이 과정을 역방향 패스까지 포함해 벤더 커널 라이브러리 위에 직접 짜 넣습니다. `TokenSwitch`는 여기에 `create_routing()`, `dispatch()`, `combine()`이라는 인터페이스를 씌우고, NCCL의 전문가 병렬(expert-parallel) 커널 위에 만들어진 `TokenSwitchNCCL`을 첫 번째 백엔드로 제공합니다. `out=` 인자 없이 호출하면 dispatch와 combine이 미분 가능한 텐서를 반환하므로, MoE 계층을 Autograd가 추적하는 평범한 Python으로 작성할 수 있습니다. `out=`을 넘기면 버퍼 재사용 경로를 유지하는 대신 Autograd를 포기하게 됩니다. 아직 초기 코드입니다. 모듈이 비공개이고 NCCL 2.30 고정 버전(pin)에 맞춰 `USE_NCCL_EP=1`로 빌드해야 하므로, 지금으로서는 NVIDIA 전용이고 기본 휠로는 쓸 수 없습니다.
> Mixture-of-experts training spends much of its step time sending each token to the ranks holding its chosen experts and bringing the expert outputs back, and teams generally wire that up themselves against a vendor kernel library, backward pass included. `TokenSwitch` puts an interface around it — `create_routing()`, `dispatch()`, `combine()` — with `TokenSwitchNCCL` as the first backend, built on NCCL's expert-parallel kernels. Called without an `out=` argument, dispatch and combine return differentiable tensors, so an MoE layer can be written as ordinary autograd-tracked Python; passing `out=` keeps the buffer-reuse path but gives up autograd. It is early code: the module is private and needs a build with `USE_NCCL_EP=1` against the NCCL 2.30 pin, so today it is NVIDIA-only and out of reach of a stock wheel.

API 안정성: Unstable
> API Unstable

(PR [#178712](https://github.com/pytorch/pytorch/pull/178712) 및 [#181314](https://github.com/pytorch/pytorch/pull/181314) 작성: Ke Wen, NVIDIA)
> (PR [#178712](https://github.com/pytorch/pytorch/pull/178712) and [#181314](https://github.com/pytorch/pytorch/pull/181314) by Ke Wen, NVIDIA)

### 한 랭크에서만 컴파일하기 / Compile-on-One-Rank

분산 작업에서는 모든 랭크가 같은 모델을 각자 컴파일하므로, 수 분이 걸리는 컴파일 한 번을 학습 시작 전에 N번 치르게 됩니다. Compile-on-one-rank는 컴파일된 산출물 하나를 어디서든 재사용할 수 있게 만듭니다. `make_fx`는 더 이상 추적 중인 랭크의 디바이스를 팩토리와 캐스트 연산에 박아 넣지 않고, Inductor의 코드 생성과 Triton 런처는 로드 시점에 디바이스를 해석합니다. 그래서 생성된 소스가 랭크 전체에서 동일하고, `cuda:0`에서 컴파일된 커널이 `cuda:3`에서 로드되어 실행됩니다. `DeviceMesh.get_group()` 역시 torchbind `ProcessGroup`을 박아 넣는 대신 그래프 안에서 메시로부터 그룹을 가져오는데, 예전에는 이 때문에 그래프를 직렬화할 수 없었습니다. 그리고 이 플래그가 켜지면 `dist.all_reduce` 같은 레거시 집합 통신이 함수형(functional) 형태로 추적됩니다. torchtitan의 실험적 [`graph_trainer`](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/graph_trainer)가 의도한 모습을 보여줍니다. 프로세스 하나가 미리 컴파일해서 산출물 하나를 쓰면, 모든 랭크가 시작할 때 이를 로드하며, 산출물을 만들기 위해 N-GPU 작업을 돌릴 필요가 없습니다. 다만 이 모드는 프로그램당 가속기 디바이스가 하나라고 가정하며 두 번째 디바이스를 건드리는 그래프는 거부하고, `torch.compiler.config.compile_on_one_rank` 뒤에서 선택적 활성화로 남아 있습니다.
> Every rank in a distributed job compiles the same model independently, so a single multi-minute compile is paid N times over before training starts. Compile-on-one-rank makes one compiled artifact reusable everywhere: `make_fx` no longer bakes the tracing rank's device into factory and cast ops, and Inductor's codegen and Triton launcher resolve the device at load time, so the generated source is identical across ranks and a kernel compiled on `cuda:0` loads and runs on `cuda:3`. `DeviceMesh.get_group()` likewise fetches the group from the mesh in-graph instead of baking in a torchbind `ProcessGroup`, which used to leave the graph unserializable, and under the flag legacy collectives like `dist.all_reduce` trace as their functional forms. torchtitan's experimental [`graph_trainer`](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/graph_trainer) shows the intended shape: a single process compiles ahead of time and writes one artifact that every rank loads at startup, no N-GPU job required to produce it — though the mode assumes a single accelerator device per program and refuses a graph that touches a second, and stays opt-in behind `torch.compiler.config.compile_on_one_rank`.

API 안정성: Unstable
> API Unstable

(PR [#187869](https://github.com/pytorch/pytorch/pull/187869), [#186892](https://github.com/pytorch/pytorch/pull/186892), [#187870](https://github.com/pytorch/pytorch/pull/187870), [#188215](https://github.com/pytorch/pytorch/pull/188215) 작성: Aaron Orenstein, University of Alberta)
> (PR [#187869](https://github.com/pytorch/pytorch/pull/187869), [#186892](https://github.com/pytorch/pytorch/pull/186892), [#187870](https://github.com/pytorch/pytorch/pull/187870) and [#188215](https://github.com/pytorch/pytorch/pull/188215) by Aaron Orenstein, University of Alberta)

## 컴파일 및 내보내기 / Compilation and Export

### @dynamic\_spec을 통한 선언적 동적 shape / Declarative Dynamic Shapes with @dynamic\_spec

어떤 입력 차원이 변하는지 PyTorch에 알려주는 방법은 진입점마다 달랐습니다. `torch.export`에는 `dynamic_shapes` 딕셔너리, `torch.compile`에는 거친 `dynamic=` 플래그, `make_fx`에는 전역 추적 모드가 있었고, 어느 쪽이든 선언이 정작 대상 모델에서 멀리 떨어진 호출 지점에 놓였습니다. 이번 릴리즈는 `torch.fx.experimental.dynamic_spec` 아래에 `ShapesSpec` API를 추가합니다. 차원의 이름을 한 번 짓고(`ShapeVar("batch", min=2, max=128)`), 여러 입력에 걸쳐 재사용하고, `batch * 2` 같은 파생 차원을 만들고, `batch % 2 == 0` 같은 가정을 붙일 수 있으며, 이제 세 진입점 모두가 동일한 `dynamic_shapes=` 키워드로 이를 받습니다. `@dynamic_spec` 데코레이터는 그 스펙을 함수나 모듈의 `forward`에 직접 붙이므로, `torch.compile`, strict 또는 non-strict `torch.export.export`, `make_fx(tracing_mode="fake")`가 호출 지점에서 아무것도 넘기지 않아도 이를 집어 갑니다. 이렇게 선언된 차원은 값이 확정되지 않은(unbacked) 심볼이 되므로, 컴파일러가 마침 추적한 배치 크기에 조용히 특수화할 수 없습니다. 대신 shape에 의존하는 분기가 가드와 재컴파일이 아니라 데이터 의존 오류로 드러나는 것이 그 대가입니다. 이 API는 실험적이며 아직 바뀌고 있습니다. `make_fx` 지원은 `tracing_mode="fake"`로 제한되고, 스펙을 `prefer_deferred_runtime_asserts_over_guards=True`와 함께 쓰거나 데코레이터를 호출 지점의 `dynamic_shapes=` 인자와 함께 쓰면 오류가 발생합니다.
> Telling PyTorch which input dimensions vary has meant a different mechanism per entry point — a `dynamic_shapes` dict for `torch.export`, a coarse `dynamic=` flag for `torch.compile`, a global tracing mode for `make_fx` — and in each case the declaration sits at the call site, far from the model it describes. This release adds a `ShapesSpec` API under `torch.fx.experimental.dynamic_spec`: you name a dimension once (`ShapeVar("batch", min=2, max=128)`), reuse it across inputs, build derived dims like `batch * 2`, and attach assumptions such as `batch % 2 == 0`, with all three entry points now taking it under the same `dynamic_shapes=` keyword. A `@dynamic_spec` decorator attaches that spec directly to a function or a module's `forward`, so `torch.compile`, strict or non-strict `torch.export.export`, and `make_fx(tracing_mode="fake")` all pick it up with nothing passed at the call site. Dimensions declared this way become unbacked symbols, so the compiler cannot quietly specialize on the batch size it happened to trace — the trade-off is that shape-dependent branching now surfaces as a data-dependent error rather than a guard and a recompile. The API is experimental and still moving: `make_fx` support is limited to `tracing_mode="fake"`, and combining a spec with `prefer_deferred_runtime_asserts_over_guards=True`, or a decorator with a call-site `dynamic_shapes=` argument, raises an error.

API 안정성: Unstable (PR [#187639](https://github.com/pytorch/pytorch/pull/187639), [#185982](https://github.com/pytorch/pytorch/pull/185982), [#187602](https://github.com/pytorch/pytorch/pull/187602), [#186751](https://github.com/pytorch/pytorch/pull/186751), [#187010](https://github.com/pytorch/pytorch/pull/187010) 작성: Laith Sakka, Meta)
> API Unstable(PR [#187639](https://github.com/pytorch/pytorch/pull/187639), [#185982](https://github.com/pytorch/pytorch/pull/185982), [#187602](https://github.com/pytorch/pytorch/pull/187602), [#186751](https://github.com/pytorch/pytorch/pull/186751) and [#187010](https://github.com/pytorch/pytorch/pull/187010) by Laith Sakka, Meta)

### AOTInductor 외부 상수와 무복사 가중치 공유 / AOTInductor External Constants and Zero-Copy Weight Sharing

같은 가중치를 공유하는 여러 AOTInductor 모델을 서빙하려면 예전에는 모델 컨테이너마다 GPU에 자기 사본을 할당하고 로드해야 했습니다. 새로운 C API인 `AOTInductorModelContainerCreateWithExternalConstants`는 컨테이너를 만들 때 호출자가 가중치 텐서를 넘겨줄 수 있게 합니다. AOTI는 상수 로딩을 아예 건너뛰고 호출자의 메모리를 사용하므로, 사본 하나로 여러 모델을 뒷받침하거나 CUDA IPC를 통해 여러 프로세스가 공유할 수 있습니다. 소유권은 호출자가 계속 갖는데, 이는 그 텐서들이 컨테이너보다 오래 살아 있어야 한다는 뜻입니다. 그리고 이 API는 C ABI를 통해서만 제공되며 아직 Python 진입점은 없습니다. 기존 코드 경로는 그대로입니다. 새 생성자는 외부 상수를 명시적으로 넘겼을 때만 동작합니다. 같은 기반 모델의 여러 변형을 서빙하는 서버 군집(fleet)이라면, 모델별 가중치 메모리가 공유된 할당 하나로 바뀝니다.
> Serving several AOTInductor models that share the same weights used to mean every model container allocating and loading its own copy on the GPU. A new C API, `AOTInductorModelContainerCreateWithExternalConstants`, lets the caller hand in weight tensors at container creation; AOTI skips constant loading entirely and uses the caller's memory instead, so one copy can back several models or be shared across processes via CUDA IPC. The caller retains ownership, which means those tensors have to outlive the container, and the API is available through the C ABI only, with no Python entry point yet. Existing code paths are untouched — the new constructor only engages when external constants are explicitly supplied. For a fleet serving many variants of the same base model, this turns per-model weight memory into a single shared allocation.

API 안정성: Unstable
> API Unstable

(PR [#188643](https://github.com/pytorch/pytorch/pull/188643) 작성: @iuliur-meta)
> (PR [#188643](https://github.com/pytorch/pytorch/pull/188643) by @iuliur-meta)

### AOTInductor 컴파일 / AOTInductor Compilation

`triton.autotune_at_compile_time=False`로 모델을 패키징하면 예전에는 코드 생성 전체를 두 번 돌렸습니다. 컴파일하고, 실행해서 커널 메타데이터를 모으고, 상태를 초기화한 뒤, 패키징을 위해 다시 컴파일하는 식이었습니다. 이제 그 경로는 JIT과 AOTI 래퍼 본문을 한 번의 코드 생성 패스에서 방출하고, 실제 입력으로 JIT 본문을 한 번 실행해 Triton 커널 설정을 포착한 뒤 패키징된 소스에 이를 박아 넣습니다. 새 경로에서는 `torch.cond`와 `torch.while_loop`가 지원됩니다. 별개로, `cpp_wrapper`가 이제 명시적인 사용자 스트림과 이벤트를 방출할 수 있습니다. 현재는 CUDA 전용이며 CUDA 그래프와 함께 쓸 수는 없습니다. 지연 오토튜닝 흐름에서 두 번째 코드 생성 패스가 사라졌고, 추적된 다중 스트림 코드가 AOTI 패키지까지 살아남습니다.
> Packaging a model with `triton.autotune_at_compile_time=False` used to run the whole codegen twice: compile, run to collect kernel metadata, reset state, then recompile for packaging. That path now emits the JIT and AOTI wrapper bodies in a single codegen pass, running the JIT body once with real inputs to capture the Triton kernel configuration that the packaged source then embeds. `torch.cond` and `torch.while_loop` are supported on the new path. Separately, `cpp_wrapper` can now emit explicit user streams and events, currently CUDA-only and not usable alongside CUDA graphs. The second codegen pass is gone from the lazy-autotune flow, and traced multi-stream code now survives into an AOTI package.

API 안정성: Unstable
> API Unstable

(PR [#184735](https://github.com/pytorch/pytorch/pull/184735) 및 [#184736](https://github.com/pytorch/pytorch/pull/184736) 작성: @desertfire, [#182971](https://github.com/pytorch/pytorch/pull/182971) 작성: Brian Bustamante)
> (PR [#184735](https://github.com/pytorch/pytorch/pull/184735) and [#184736](https://github.com/pytorch/pytorch/pull/184736) by @desertfire, [#182971](https://github.com/pytorch/pytorch/pull/182971) by Brian Bustamante)

### AOTInductor 상수 로딩 / AOTInductor Constant Loading

모델의 가중치를 로드할 때는 호스트 메모리에서 GPU로 복사하는데, 페이지 가능한(pageable) 메모리에서 동기적으로 복사하면 디바이스 전체 동기화가 강제되어 다른 스트림에서 이미 돌고 있는 추론이 멈춥니다. `AOTInductorSetUsePinnedAsyncConstantsCopy`는 상수 로딩과 갱신을 대신 고정(pinned) 스테이징 버퍼를 통해 라우팅하여 호스트 복사와 디바이스 전송을 중첩시키며, 버퍼 크기를 정하는 동반 호출과 폴백용 `AOTI_COPY_USE_PINNED_ASYNC` 환경 변수를 함께 제공합니다. 기본적으로는 꺼져 있고, 모델이나 컨테이너를 만들기 전에 활성화해야 합니다. `.so`를 로드한 뒤 모델이 서빙 준비를 마치기까지의 구간은 예전에는 아무것도 보이지 않았기에, 이제 `AOTI_LOG_LOADING`을 설정하면 복사 시간과 고정 풀(pinned-pool) 진단 정보를 담은 `[AOTI_LOAD]` 마커가 방출됩니다. 운영 트래픽을 받는 중에 모델을 갈아 끼우는 서버라면, 이 둘을 합쳐 로드 중에도 GPU의 나머지를 계속 바쁘게 유지하고 느린 로드를 다시 빌드하지 않고도 진단할 수 있습니다.
> Loading a model's weights copies them from host memory to the GPU, and a synchronous copy out of pageable memory forces a device-wide synchronization that stalls inference already running on other streams. `AOTInductorSetUsePinnedAsyncConstantsCopy` routes constant loading and updates through pinned staging buffers instead, overlapping host copies with device transfers, with a companion call to size the buffers and the `AOTI_COPY_USE_PINNED_ASYNC` environment variable as a fallback. It's off by default and has to be enabled before the model or container is created. The stretch between loading the `.so` and having a model ready to serve was previously silent, so setting `AOTI_LOG_LOADING` now emits `[AOTI_LOAD]` markers with copy timing and pinned-pool diagnostics. For a server swapping models in and out under live traffic, that combination keeps the rest of the GPU busy during a load and makes a slow one diagnosable without a rebuild.

API 안정성: Unstable
> API Unstable

(PR [#186258](https://github.com/pytorch/pytorch/pull/186258) 및 [#186309](https://github.com/pytorch/pytorch/pull/186309) 작성: @joshuuuasu)
> (PR [#186258](https://github.com/pytorch/pytorch/pull/186258) and [#186309](https://github.com/pytorch/pytorch/pull/186309) by @joshuuuasu)

### Helion 백엔드 통합 / Helion Backend Integration

빠른 GPU 커널을 손으로 작성한다는 것은 타일 크기, 루프 순서, 메모리 접근 패턴을 고르고, 새로운 shape과 새로운 GPU마다 그 전부를 다시 튜닝한다는 뜻입니다. Helion은 그 수준을 한 단계 끌어올립니다. 알고리즘을 Python으로 작성하면 Helion이 스케줄 공간을 탐색해 Triton을 방출해 줍니다. PyTorch 2.14는 2.13에서 도입한 네이티브 DSL 레지스트리에 Helion을 세 번째 항목으로 등록하므로, Triton과 CuTeDSL로 작성된 커널이 그랬듯 Helion으로 작성된 커널도 ATen 연산을 재정의할 수 있고, `torch.backends.python_native.helion`으로 제어됩니다. 등록하려면 `helion` 패키지와 그 저수준화(lowering) 백엔드가 필요하며, ROCm 빌드에서는 사용할 수 없습니다. 이번 릴리즈에서 Helion으로 라우팅되는 연산자는 없습니다. 이는 이후 릴리즈에서 나올 Helion 기반 커널 재정의를 위한 토대입니다.
> Writing a fast GPU kernel by hand means choosing tile sizes, loop order, and memory access patterns, then re-tuning all of it for every new shape and every new GPU. Helion raises that a level: you write the algorithm in Python, and Helion searches the schedule space and emits Triton for you. PyTorch 2.14 registers Helion as a third entry in the native DSL registry introduced in 2.13, so Helion-authored kernels can override ATen operations the same way Triton and CuTeDSL ones already can, controlled through `torch.backends.python_native.helion`. Registration requires the `helion` package plus its lowering backend, and is unavailable on ROCm builds. No operators are routed through Helion in this release; this is the foundation for Helion-backed kernel overrides landing in subsequent ones.

API 안정성: Unstable
> API Unstable

(PR [#190636](https://github.com/pytorch/pytorch/pull/190636) 작성: Karthick Panner Selvam, Meta)
> (PR [#190636](https://github.com/pytorch/pytorch/pull/190636) by Karthick Panner Selvam, Meta)

## 플랫폼 기능 및 업데이트 / Platform Features and Updates

### CUDA

#### Inductor를 위한 CuTeDSL GEMM 백엔드, NVGEMM / NVGEMM, a CuTeDSL GEMM Backend for Inductor

PyTorch 2.13은 TorchInductor를 위한 NVGEMM CuTeDSL 백엔드를 도입했고, 이번 릴리즈에서는 지원을 에필로그 융합까지 넓히게 되어 기쁩니다. 이전 버전은 독립적인 커널을 방출할 수 있었지만, 그 뒤에 오는 것(바이어스 덧셈, 활성화, 재스케일링)은 별도 커널에 남아 결과를 메모리에서 다시 읽어야 했습니다. 이번 릴리즈는 NVIDIA의 공식 `cutlass.operators` API인 NVGEMM을 활용해, `mm`, `addmm`, `scaled_mm`에 대해 Triton, ATen과 경쟁하는 후보를 생성합니다. 이 커널들은 Triton 템플릿이 그러하듯 에필로그를 융합합니다. addmm의 바이어스 덧셈, 연쇄된 점별(pointwise) 연산, GEMM 결과에 대한 리덕션이 대상이며, 커널이 리덕션된 값과 전체 출력 행렬을 함께 반환하는 경우도 포함됩니다. 융합은 낮은 정밀도 경로까지 닿아서, 스케일링된 GEMM 이후의 점별 작업이 커널 안으로 접혀 들어가고 NVFP4의 실행 시점 전역 스케일도 별도의 곱셈이 아니라 에필로그 안에서 적용됩니다. 융합된 커널은 디스크에도 캐시되므로, 지금까지 매번 처음부터 다시 컴파일하던 프로세스가 대신 이를 재사용합니다. 사용하려면 `max_autotune` 아래의 `max_autotune_gemm_backends`에 `NVGEMM`을 추가하세요. `nvidia-cutlass-dsl` 4.6.0이 필요하고, NVFP4 경로에는 Blackwell이 필요하며, 백엔드가 표현할 수 없는 에필로그는 Triton으로 폴백하므로 그런 경우에도 기존 융합은 유지됩니다. 이 백엔드에 계속 투자하고 있으며, 오토튜닝 시간과 성능에 대한 추가 개선을 현재 구현 중입니다.
> PyTorch 2.13 introduced the NVGEMM CuTeDSL backend for TorchInductor and in this release we are excited to expand support to epilogue fusion — the previous version could emit a standalone kernel, and whatever followed it (a bias add, an activation, a rescale) stayed in a separate kernel that re-read the result from memory. This release now utilizes NVGEMM, NVIDIA's official `cutlass.operators` API, generating candidates that compete with Triton and ATen for `mm`, `addmm`, and `scaled_mm`. Its kernels fuse epilogues the way Triton templates do: addmm's bias add, chained pointwise ops, and reductions over the GEMM result, including cases where the kernel returns both the reduced value and the full output matrix. Fusion now reaches the low-precision paths as well, so pointwise work after a scaled GEMM folds into the kernel, and NVFP4's runtime global scale is applied inside the epilogue rather than as a separate multiply. Fused kernels are also cached to disk, so a process that recompiles them from scratch today reuses them instead. Enable it by adding `NVGEMM` to `max_autotune_gemm_backends` under `max_autotune`; it needs `nvidia-cutlass-dsl` 4.6.0, the NVFP4 paths require Blackwell, and epilogues the backend can't express fall back to Triton so those cases keep their existing fusion. We are continuing to invest in this backend, with further improvements to autotuning time and performance currently being implemented.

API 안정성: Unstable
> API Unstable

(PR [#186183](https://github.com/pytorch/pytorch/pull/186183), [#187013](https://github.com/pytorch/pytorch/pull/187013), [#189772](https://github.com/pytorch/pytorch/pull/189772), [#189774](https://github.com/pytorch/pytorch/pull/189774), [#189805](https://github.com/pytorch/pytorch/pull/189805), [#190808](https://github.com/pytorch/pytorch/pull/190808), [#190823](https://github.com/pytorch/pytorch/pull/190823) 작성: Michael Lazos, Meta)
> (PRs [#186183](https://github.com/pytorch/pytorch/pull/186183), [#187013](https://github.com/pytorch/pytorch/pull/187013), [#189772](https://github.com/pytorch/pytorch/pull/189772), [#189774](https://github.com/pytorch/pytorch/pull/189774), [#189805](https://github.com/pytorch/pytorch/pull/189805), [#190808](https://github.com/pytorch/pytorch/pull/190808) and [#190823](https://github.com/pytorch/pytorch/pull/190823) by Michael Lazos, Meta)

#### CUDA 그래프 생명주기 훅 / CUDA Graph Lifecycle Hooks

프로파일러나 메모리 추적기처럼 CUDA 그래프를 바깥에서 지켜보려는 도구는 그래프별 훅만 등록할 수 있었는데, 그래프를 그 도구가 아니라 Inductor나 NCCL이 만들었다면 아무 도움이 되지 않았습니다. 이번 릴리즈는 프로세스 안의 모든 그래프에 대해 발동하는 모듈 수준 훅(캡처 시작과 종료, 재생 시작과 종료, 인스턴스화, 소멸)과 함께, 그래프별 재생 시작/종료 훅, 그리고 어느 형태로도 존재하지 않던 캡처 시작 훅을 추가합니다. `CUDAGraph`에는 정리 작업이나 객체 수명을 그래프 자신의 수명에 묶는 `register_destroy_callback`과 `retain_object`도 추가됐습니다. 콜백이 그래프가 여전히 참조하는 메모리를 해제한다면 `synchronize_before_release=True`를 넘기세요. 해체가 비동기적이라 재생이 진행 중인 상태에서 해제하면 해제 후 사용(use-after-free)이 되기 때문입니다. 이제 관측 도구는 그래프 코드가 그 도구를 전혀 알지 못해도 그래프의 전체 생명주기를 따라갈 수 있고, 아무것도 등록하지 않으면 비용도 들지 않습니다.
> Tools that want to watch CUDA graphs from the outside, like a profiler or a memory tracker, could only register per-graph hooks, which is no help when the graph was constructed by Inductor or NCCL rather than by the tool. This release adds module-level hooks that fire for every graph in the process (capture start and end, replay start and end, instantiate, destroy), along with per-graph replay start/end hooks and a capture-start hook that previously existed in neither form. `CUDAGraph` also gains `register_destroy_callback` and `retain_object` for tying cleanup or object lifetime to the graph's own; pass `synchronize_before_release=True` if the callback frees memory the graph still references, since teardown is asynchronous and releasing under an in-flight replay is a use-after-free. Observability tooling can now follow a graph's full lifecycle without the graph code carrying any knowledge of the tool, and registering nothing costs nothing.

API 안정성: Unstable
> API Unstable

(PR [#190582](https://github.com/pytorch/pytorch/pull/190582) 및 [#190602](https://github.com/pytorch/pytorch/pull/190602) 작성: Natalia Gimelshein, [#191299](https://github.com/pytorch/pytorch/pull/191299) 및 [#192162](https://github.com/pytorch/pytorch/pull/192162) 작성: @dolpm)
> (PR [#190582](https://github.com/pytorch/pytorch/pull/190582) and [#190602](https://github.com/pytorch/pytorch/pull/190602) by Natalia Gimelshein, [#191299](https://github.com/pytorch/pytorch/pull/191299) and [#192162](https://github.com/pytorch/pytorch/pull/192162) by @dolpm)

#### 하나의 CUDA 그래프 안에서 여러 메모리 풀 사용 / Multiple Memory Pools in a Single CUDA Graph

`CUDAGraph` 캡처는 예전에 정확히 하나의 메모리 풀에 묶여 있었고, 그래서 별도 풀에서 와야 하는 할당(대칭 메모리가 이 문제를 촉발한 사례입니다)은 캡처된 영역에 참여할 수 없었습니다. 이제 캡처가 `torch.cuda.use_mem_pool()`로 보조 풀에 진입할 수 있고 그래프는 그 풀들을 모두 유지합니다. `g.pool()`은 `torch.cuda.graph()`에 넘긴 주 풀을 반환하고, `g.pools()`는 캡처 중에 진입한 보조 풀까지 포함한 전체 집합을 반환합니다. 이로써 대칭 메모리 버퍼가 CUDA 그래프 안에서 살아남을 수 있어, 전용 풀을 통해 할당하는 분산 워크로드의 그래프 캡처가 가능해집니다. 기존 제약 하나는 그대로 남습니다. `use_mem_pool`은 스레드 ID로 할당을 라우팅하므로, 다중 스레드 Autograd에서 그 안에서 `.backward()`를 호출해도 역방향 할당은 그 풀로 가지 않습니다. 그렇게 하려면 Autograd를 같은 스레드에서 실행하세요.
> A `CUDAGraph` capture previously bound to exactly one memory pool, which meant allocations that need to come from a separate pool — symmetric memory being the case that forced the issue — couldn't participate in a captured region. A capture can now enter side pools with `torch.cuda.use_mem_pool()` and the graph retains all of them: `g.pool()` returns the primary pool passed to `torch.cuda.graph()`, and `g.pools()` returns the full set including any side pools entered during capture. This lets symmetric-memory buffers survive inside a CUDA graph, which unblocks graph capture for distributed workloads that allocate through a dedicated pool. One existing limitation carries over: `use_mem_pool` routes allocations by thread ID, so calling `.backward()` inside it with multithreaded autograd won't send the backward allocations to the pool — run autograd on the same thread if you need that.

API 안정성: Unstable
> API Unstable

(PR [#187929](https://github.com/pytorch/pytorch/pull/187929) 작성: @Aidyn-A)
> (PR [#187929](https://github.com/pytorch/pytorch/pull/187929) by @Aidyn-A)

#### torch.while\_loop의 CUDA 그래프 캡처 / CUDA Graph Capture for torch.while\_loop

데이터에 따라 반복 횟수가 달라지는 루프는 워크로드를 CUDA 그래프로 온전히 캡처하지 못하게 만드는 전형적인 이유 중 하나였습니다. 몇 번 반복할지 정하려고 디바이스에서 호스트로 복사를 강제했고, 하필 그대로 두고 싶은 지점에서 캡처가 끊겼습니다. 이제 `torch.while_loop`를 CUDA의 `while` 조건 노드를 사용해 CUDA 그래프에 캡처할 수 있습니다. 조건은 노드가 추가되기 전에 부모 스트림에서 평가되고 본문이 실행될 때마다 끝에서 다시 평가되므로, 캡처된 그래프 하나가 재생될 때 실행 시점에 결정된 횟수만큼 반복합니다. 이것만으로 처리량이 늘지는 않습니다. 핵심은 루프가 더 이상 그래프 캡처 밖으로 내몰지 않는다는 것이며, 그래서 가변 길이 인덱스 텐서에 대한 리덕션이나 가변 개수의 패킹된 시퀀스에 적용되는 손실 같은 경우가 그래프 하나 안에 머무를 수 있습니다. 고정된 최대 반복 횟수와 텐서만 허용되는 이월(carried) 입력 등 기존 `while_loop` 제약은 그대로 적용됩니다.
> Data-dependent loop counts have been one of the standard reasons a workload can't be fully CUDA-graph captured, forcing a device-to-host copy to decide how many iterations to run and breaking the capture at exactly the point you'd rather keep it intact. `torch.while_loop` can now be captured into a CUDA graph using CUDA's `while` conditional nodes: the condition is evaluated on the parent stream before the node is added and re-evaluated at the end of each body execution, so a single captured graph runs a runtime-determined number of iterations on replay. This isn't a throughput win on its own — the point is that the loop no longer forces you out of graph capture, so cases like a reduction over a variable-length index tensor, or a loss applied to a variable number of packed sequences, stay inside one graph. The usual `while_loop` constraints still apply, including a fixed maximum trip count and tensor-only carried inputs.

API 안정성: Unstable
> API Unstable

(PR [#186055](https://github.com/pytorch/pytorch/pull/186055) 작성: Daniel Galvez, NVIDIA)
> (PR [#186055](https://github.com/pytorch/pytorch/pull/186055) by Daniel Galvez, NVIDIA)

#### CUDA 그래프를 위한 커널 어노테이션 / Kernel Annotations for CUDA Graphs

`torch.cuda.graph_annotations`는 CUDA 그래프 캡처와 함께 쓰이던 커널 어노테이션 API를 공개합니다. `mark_kernels`로 GPU 작업에 이름표를 붙이면, 나중에 프로파일러 추적을 내보낼 때 익명의 커널 실행이 아니라 이름이 붙은 채로 나타납니다. 예전에는 `mark_kernels` 범위 안에 어휘적으로(lexically) 포함된 순방향 패스 커널에서만 동작했고, Autograd가 실제로 실행될 때 뒤늦게 캡처되는 역방향 커널에는 이름표가 붙지 않았습니다. 이제 역방향 커널도 앞서 소개한 `node_creation_hook` 메커니즘을 사용해 자동으로 어노테이션되며, 이중 역전파와 체크포인트 재계산을 거치더라도 그 커널을 만든 순방향 범위로 귀속됩니다. 역방향 귀속을 직접 처리하려는 호출자를 위해 `backward=False`로 끌 수 있습니다.
> `torch.cuda.graph_annotations` makes the kernel-annotation API used with CUDA graph capture public: `mark_kernels` lets you tag GPU work with a name so it shows up labeled when you later export a profiler trace, rather than as an anonymous kernel launch. Previously this only worked for forward-pass kernels captured lexically inside the `mark_kernels` scope — backward kernels, captured later when autograd actually runs, were never tagged. Backward kernels are now annotated automatically, using the `node_creation_hook` mechanism from above to attribute them back to whichever forward scope created them, including through double-backward and checkpoint recomputation. This is opt-out via `backward=False` for callers that want to do their own backward attribution.

API 안정성: Unstable
> API Unstable

(PR [#189417](https://github.com/pytorch/pytorch/pull/189417) 및 [#191563](https://github.com/pytorch/pytorch/pull/191563) 작성: Edward Yang, Meta)
> (PR [#189417](https://github.com/pytorch/pytorch/pull/189417) and [#191563](https://github.com/pytorch/pytorch/pull/191563) by Edward Yang, Meta)

#### 사후 메모리 스냅샷 어노테이션 / Post-Facto Memory Snapshot Annotations

메모리 스냅샷은 이미 할당에 메타데이터를 붙일 수 있게 해주지만, 붙일 수 있는 시점은 할당이 생성되는 순간뿐이었습니다. 어떤 정보, 예를 들어 텐서가 결국 Autograd 그래프에 붙들려 있게 되었는지 같은 것은 역방향 테이프에 담긴 뒤에야 알 수 있습니다. `torch.cuda.memory._annotate_tensor(tensor, metadata)`는 살아 있는 할당에 나중에 메타데이터를 붙일 수 있게 하며, 별도의 타임스탬프 이벤트로 기록되므로 할당 시점에 기록된 내용을 덮어쓰지 않습니다. 뷰와 오프셋 텐서는 자동으로 할당의 기준 주소로 해석되고, 메모리 스냅샷 시각화 도구는 이제 이 어노테이션을 타임라인의 할당 옆에 함께 보여줍니다.
> Memory snapshots already let you attach metadata to an allocation, but only at the moment it's created — some information, like whether a tensor ended up retained by the autograd graph, is only known later, after it's been packed into the backward tape. `torch.cuda.memory._annotate_tensor(tensor, metadata)` lets you attach metadata to a live allocation after the fact, recorded as a separate timestamped event so it doesn't clobber whatever was recorded at alloc time. Views and offset tensors resolve to the allocation's base address automatically, and the memory snapshot visualizer now shows these annotations alongside the allocation in the timeline.

API 안정성: Unstable
> API Unstable

(PR [#190575](https://github.com/pytorch/pytorch/pull/190575) 작성: Edward Yang, Meta)
> (PR [#190575](https://github.com/pytorch/pytorch/pull/190575) by Edward Yang, Meta)

#### CUDA에서의 TunableOp / TunableOp on CUDA

TunableOp는 입력 shape마다 사용 가능한 GEMM 구현들을 실행 시점에 프로파일링해 가장 빠른 것을 캐시하지만, CUDA 빌드에서는 고를 수 있는 후보가 cuBLAS 기본값 하나뿐이었습니다. 이제 cuBLASLt 휴리스틱 후보도 등록되며, 후보 개수는 `PYTORCH_TUNABLEOP_CUBLASLT_REQUESTED_ALGO_COUNT`나 `torch.cuda.tunable.set_cublaslt_requested_algo_count()`로 정합니다. 이는 전반적인 속도 향상 수단이라기보다, 기본 휴리스틱이 잘 다루지 못하는 shape을 구제하는 도구입니다. H100에서 평균은 대체로 평평하고, 개별 shape은 0.66배에서 1.57배까지 분포합니다. 패딩된 선행 차원(leading dimension)이 m/n/k 중 하나와 같은 경우에 `tune_gemm_in_file`이 조용히 엉뚱한 shape을 튜닝하던 문제도 오프라인 튜닝에서 수정됐습니다.
> TunableOp profiles the available GEMM implementations for each input shape at runtime and caches the fastest, but on CUDA builds it previously had only one candidate to pick from — the cuBLAS default. It now registers cuBLASLt heuristic candidates too, with the candidate count set by `PYTORCH_TUNABLEOP_CUBLASLT_REQUESTED_ALGO_COUNT` or `torch.cuda.tunable.set_cublaslt_requested_algo_count()`. It's a tool for rescuing shapes the stock heuristic handles badly rather than a general speedup: on H100 the mean is roughly flat, with individual shapes ranging from 0.66x to 1.57x. Offline tuning is also fixed for the case where a padded leading dimension equals one of m/n/k, which previously made `tune_gemm_in_file` silently tune the wrong shape.

API 안정성: Unstable
> API Unstable

(PR [#186270](https://github.com/pytorch/pytorch/pull/186270) 작성: Grayson Derossi, NVIDIA, [#189355](https://github.com/pytorch/pytorch/pull/189355) 작성: Aditya Srichandan, AMD)
> (PR [#186270](https://github.com/pytorch/pytorch/pull/186270) by Grayson Derossi, NVIDIA  and [#189355](https://github.com/pytorch/pytorch/pull/189355) by Aditya Srichandan, AMD)

#### 그룹 GEMM 백엔드로서의 cuBLASLt / cuBLASLt as a grouped GEMM backend

그룹 GEMM(grouped GEMM)은 서로 다른 shape의 행렬곱이 한꺼번에 발행되는 MoE 계층을 떠받칩니다. cuBLASLt가 CUTLASS 및 폴백과 함께 백엔드로 합류했습니다. CUDA 13.2 이상의 Blackwell과 CUDA 13.3 이상의 Hopper에서 fp16에 대해 기본값이 되고, 같은 조합에서 bf16에 대해서는 `torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm = True`로 선택 활성화할 수 있습니다. 이 구분은 측정 결과를 반영한 것입니다. 크기가 불규칙한(ragged) MoE 형태의 그룹에서는 앞서지만, 균일한 그룹에서는 대체로 CUTLASS bf16 커널에 뒤집니다. `torch.compile` 및 CUDA 그래프와 함께 동작하며, 직접 챙겨야 할 것은 행렬과 선행 차원의 16바이트 정렬 하나뿐입니다.
> Grouped GEMM drives MoE layers, where many differently-shaped matmuls are issued together. cuBLASLt joins CUTLASS and the fallback as a backend: default for fp16 on Blackwell with CUDA 13.2+ and Hopper with CUDA 13.3+, opt-in for bf16 on the same combinations via `torch.backends.cuda.matmul.prefer_cublaslt_grouped_gemm = True`. That split reflects the measurements — it wins on ragged MoE-style groups but generally trails the CUTLASS bf16 kernel on uniform ones. It works with `torch.compile` and CUDA Graphs; the one thing you must handle yourself is 16-byte alignment on matrices and leading dimensions.

API 안정성: Unstable
> API Unstable

(PR [#177037](https://github.com/pytorch/pytorch/pull/177037) 작성: Grayson Derossi, NVIDIA)
> (PR [#177037](https://github.com/pytorch/pytorch/pull/177037) by Grayson Derossi, NVIDIA)

### ROCm

#### 그룹 GEMM, CK 템플릿, Origami (ROCm GEMM) / Grouped GEMM, CK Templates and Origami (ROCm GEMM)

AMD GPU의 전문가 혼합 모델은 그동안 Inductor의 Triton 컴파일 그룹 GEMM을 쓰지 못했습니다. 이 기능이 NVIDIA SM90+ 하드웨어로 제한돼 있어, ROCm은 대신 hipBLASLt/rocBLAS 호출을 for 루프로 도는 더 느린 경로로 폴백했습니다. 이번 릴리즈는 그 Triton 저수준화(lowering)를 스케일링된(FP8) 변형까지 포함해 ROCm에 가져오고, Composable Kernel GEMM 템플릿이 사전 컴파일뿐 아니라 JIT `cpp_wrapper` 컴파일에서도 동작하게 합니다. 여기에 더해 AMD의 분석적 타일 크기 선택기인 Origami가 ROCm max-autotune에서 기본으로 켜지므로, Inductor가 전체 오토튜닝 탐색(sweep) 비용을 치르는 대신 지연 시간 모델로부터 거의 최적에 가까운 GEMM 구성을 고를 수 있습니다. 이 모두는 ROCm에 한정되며 max-autotune 경로에만 영향을 줍니다. NVIDIA 사용자에게는 아무 변화도 없습니다.
> Mixture-of-experts models on AMD GPUs used to miss out on Inductor's Triton-compiled grouped GEMM, which was previously limited to NVIDIA SM90+ hardware — ROCm fell back to a slower for-loop over hipBLASLt/rocBLAS calls instead. This release brings that Triton lowering to ROCm, including the scaled (FP8) variant, and also lets Composable Kernel GEMM templates work under JIT cpp\_wrapper compilation, not just ahead-of-time compilation. On top of that, Origami — AMD's analytical tile-size selector — is now on by default for ROCm max-autotune, so Inductor can pick near-optimal GEMM configs from a latency model instead of paying for a full autotuning sweep. All of this is ROCm-specific and only affects max-autotune paths; NVIDIA users won't see any change.

API 안정성: Unstable
> API Unstable

(PR [#188600](https://github.com/pytorch/pytorch/pull/188600) 및 [#188742](https://github.com/pytorch/pytorch/pull/188742) 작성: Nichols A. Romero, AMD, [#185505](https://github.com/pytorch/pytorch/pull/185505) 작성: Bin Bao, Meta, [#186644](https://github.com/pytorch/pytorch/pull/186644) 작성: Umesh Chand, AMD)
> (PR [#188600](https://github.com/pytorch/pytorch/pull/188600) and [#188742](https://github.com/pytorch/pytorch/pull/188742) by Nichols A. Romero, AMD, [#185505](https://github.com/pytorch/pytorch/pull/185505) by Bin Bao, Meta, [#186644](https://github.com/pytorch/pytorch/pull/186644) by Umesh Chand, AMD)

#### RDNA3를 위한 FlexAttention 타일 설정 / FlexAttention Tile Configs for RDNA3

AMD의 RDNA3 GPU(MI 시리즈 데이터센터 라인이 아니라 Radeon 워크스테이션 및 소비자용 카드)에서 FlexAttention은 해당 아키텍처에 맞게 튜닝되지 않은 타일 크기를 쓰고 있어서, 짧은 시퀀스에서 중간 길이 시퀀스까지의 성능을 놓치고 있었습니다. 이번 릴리즈는 RDNA3에 대해 직접 프로파일링한 시퀀스 길이 인식 타일 설정을 추가하여, Inductor가 고정된 기본값 하나가 아니라 실제 시퀀스 길이에 맞는 타일 크기를 고르도록 합니다. 그 결과 수백 토큰 초반대 구간에서 지연 시간이 대략 2~8배 낮아졌고, 아주 짧은 시퀀스에서는 성능 저하가 없습니다. 다른 AMD 또는 NVIDIA 하드웨어에서 FlexAttention을 실행 중이라면 이 변경은 해당되지 않습니다.
> FlexAttention on AMD's RDNA3 GPUs (Radeon workstation and consumer cards, not the MI-series datacenter line) was using tile sizes that weren't tuned for the architecture, leaving performance on the table for short-to-medium sequence lengths. This release adds sequence-length-aware tile configs specifically profiled for RDNA3, so Inductor picks a tile size matched to the actual sequence length instead of one fixed default. The result is roughly 2-8x lower latency in the low-hundreds-of-tokens range, with no regression on very short sequences. If you're running FlexAttention on other AMD or NVIDIA hardware, this change doesn't apply to you.

API 안정성: Unstable
> API Unstable

(PR [#177840](https://github.com/pytorch/pytorch/pull/177840) 작성: Robert Esclapez, AMD)
> (PR [#177840](https://github.com/pytorch/pytorch/pull/177840) by Robert Esclapez, AMD)

### MPS (Apple Silicon)

#### 네이티브 선형대수 / Native Linear Algebra

MPS 선형대수는 역사적으로 Apple의 MPSGraph 프리미티브에 기대거나, 기초적인 연산을 넘어서는 것이면 아예 CPU로 폴백해 왔습니다. 그래서 CPU와 MPS를 오가는 왕복이 수치 계산 코드에서 흔한 속도 저하의 원인이었습니다. 이번 릴리즈는 그런 구멍 몇 개를 네이티브 Metal 커널로 대체합니다. SVD, `eigh`, `lstsq`가 이제 float32와 complex64에 대해 Jacobi 방식 커널을 통해 네이티브로 실행되며(Metal에는 double 타입이 없으므로 float64는 CPU로 폴백하고, GPU 실행 오버헤드가 아깝지 않은 작은 행렬이나 작은 배치도 마찬가지입니다), 이에 의존하는 `matrix_rank`, `pinv`, `cond`, 노름 계산도 함께 살아납니다. 다만 에르미트가 아닌 `eig`/`eigvals`는 나중으로 미뤄졌습니다. Cholesky는 `matmul2d` 기반 후행 업데이트(trailing update)를 사용하는 더 빠른 패널 분해 알고리즘을 갖게 됐고(크기에 따라 대략 1.2~2.8배 빠릅니다), 이전에는 dtype 가드가 없어 조용히 잘못된 결과를 낼 수 있었던 복소수 dtype에 대한 정확성 수정도 포함됐습니다. `lu_factor`와 `lu_solve`는 Apple의 `MPSMatrixDecompositionLU`에서 손으로 작성한 Metal 커널로 옮겨 갔고, 연산 수준의 속도 향상은 상당할 수 있습니다. 제출자가 측정한 바로는 작은 배치 행렬에서 100배 이상, 더 큰 단일 행렬에서 2~9배였습니다. 마무리로 `geqrf`가 추가되고 `linalg_qr`은 CPU와 CUDA가 쓰는 디바이스에 구애받지 않는 코드 경로를 공유하도록 리팩터링됐으며, `matrix_exp`와 `linalg.polar`(역방향 패스 포함)도 이제 MPS에서 사용할 수 있습니다. 다만 `matrix_exp`는 대략 512×512를 넘어서야 CPU를 앞섭니다.
> MPS linear algebra has historically leaned on Apple's MPSGraph primitives or fallen back to CPU entirely for anything beyond the basics, which made mixed CPU/MPS round-trips a common source of slowdown in numerical code. This release replaces several of those gaps with native Metal kernels. SVD, `eigh`, and `lstsq` now run natively via Jacobi-style kernels for float32 and complex64 (falling back to CPU for float64, since Metal has no double type, and for small matrices/batches where the GPU launch overhead isn't worth it) — and this also lights up `matrix_rank`, `pinv`, `cond`, and norm computations that depend on them, though non-Hermitian `eig`/`eigvals` are left for later. Cholesky gets a faster panel-factorization algorithm with a `matmul2d`-based trailing update (roughly 1.2–2.8x faster depending on size) plus a correctness fix for complex dtypes, which previously had no dtype guard and could silently produce wrong results. `lu_factor` and `lu_solve` move from Apple's `MPSMatrixDecompositionLU` to hand-written Metal kernels, and the op level speedups can be substantial — the submitter measured >100x on small batched matrices and 2–9x on larger single matrices. Rounding things out, `geqrf` is added and `linalg_qr` is refactored to share the same device-agnostic code path CPU and CUDA use, and `matrix_exp` and `linalg.polar` (including its backward pass) are now available on MPS as well — though `matrix_exp` only overtakes CPU above roughly 512×512.

API 안정성: Unstable
> API Unstable

(PR [#185954](https://github.com/pytorch/pytorch/pull/185954) 작성: Darko SImonovski, [#187022](https://github.com/pytorch/pytorch/pull/187022) 및 [#191836](https://github.com/pytorch/pytorch/pull/191836) 작성: Irakli Salia, Hugging Face, [#189192](https://github.com/pytorch/pytorch/pull/189192) 작성: Kurt Mohler, OpenTeams, [#187038](https://github.com/pytorch/pytorch/pull/187038), [#189200](https://github.com/pytorch/pytorch/pull/189200), [#188954](https://github.com/pytorch/pytorch/pull/188954), [#189701](https://github.com/pytorch/pytorch/pull/189701), [#189732](https://github.com/pytorch/pytorch/pull/189732) 작성: Irakli Salia, Hugging Face)
> (PR [#185954](https://github.com/pytorch/pytorch/pull/185954) by Darko SImonovski, [#187022](https://github.com/pytorch/pytorch/pull/187022) and [#191836](https://github.com/pytorch/pytorch/pull/191836) by Irakli Salia, Hugging Face, [#189192](https://github.com/pytorch/pytorch/pull/189192) by Kurt Mohler, OpenTeams,  [#187038](https://github.com/pytorch/pytorch/pull/187038), [#189200](https://github.com/pytorch/pytorch/pull/189200), [#188954](https://github.com/pytorch/pytorch/pull/188954), [#189701](https://github.com/pytorch/pytorch/pull/189701), and [#189732](https://github.com/pytorch/pytorch/pull/189732) by Irakli Salia, Hugging Face)

#### FlexAttention 개선 / FlexAttention Improvements

2.13에서 FlexAttention이 MPS에 도입된 데 이어, 이번 릴리즈는 사람들이 실제 모델에 쓰기 시작하면서 드러난 여러 구멍을 메웁니다. KV 배치 브로드캐스팅은 key/value 텐서를 정확히 일치시키는 대신 쿼리 배치 전반에서 공유할 수 있게 해줍니다. 이는 공유 KV 캐시에 대해 여러 시퀀스를 서빙해야 하는 페이지드 어텐션(paged attention)의 전제 조건입니다. 이제 `flex_attention`은 주 결과와 함께 log-sum-exp 및 최대 점수 보조 출력을 반환할 수 있는데, 출력 자체를 넘어 어텐션 가중치를 소비하는 모든 것(커스텀 손실, 분석, 그리고 언젠가 나올 MPS 네이티브 역방향 지원)이 이를 필요로 합니다. 그리고 `score_mod`/`mask_mod` 함수가 이제 동적 shape 값(SymInt)을 직접 캡처할 수 있어, 실행 시점 크기로 만든 마스크(예: 시퀀스 길이에 따라 달라지는 컷오프)가 `torch.compile(dynamic=True)`를 쓸 때 shape이 바뀔 때마다 재컴파일을 강제하지 않습니다. 후속 최적화는 값이 들어갈 때 캡처된 SymInt를 32비트 정수로 좁힙니다. 커널 안에서 64비트 산술은 측정 가능한 수준으로 느리기 때문입니다.
> Building on FlexAttention's arrival on MPS in 2.13, this release rounds out several gaps that showed up once people started using it for real models. KV batch broadcasting lets key/value tensors be shared across the query batch instead of requiring an exact match — a prerequisite for paged attention, which needs this to serve multiple sequences against a shared KV cache. `flex_attention` can now return the log-sum-exp and max-score auxiliary outputs alongside the main result, needed by anything that consumes attention weights beyond the output itself (custom losses, analysis, and eventually MPS-native backward support). And `score_mod`/`mask_mod` functions can now capture dynamic shape values (SymInts) directly, so masks built from runtime sizes — like a sequence-length-dependent cutoff — no longer force a recompile every time the shape changes under `torch.compile(dynamic=True)`. A follow-up optimization narrows those captured SymInts to 32-bit integers when the value fits, since 64-bit arithmetic in the kernel is measurably slower.

API 안정성: Unstable
> API Unstable

(PR [#187722](https://github.com/pytorch/pytorch/pull/187722), [#187768](https://github.com/pytorch/pytorch/pull/187768), [#188362](https://github.com/pytorch/pytorch/pull/188362), [#188403](https://github.com/pytorch/pytorch/pull/188403), [#188663](https://github.com/pytorch/pytorch/pull/188663) 작성: Irakli Salia, Hugging Face)
> (PR [#187722](https://github.com/pytorch/pytorch/pull/187722), [#187768](https://github.com/pytorch/pytorch/pull/187768), [#188362](https://github.com/pytorch/pytorch/pull/188362), and [#188403](https://github.com/pytorch/pytorch/pull/188403) , [#188663](https://github.com/pytorch/pytorch/pull/188663) by Irakli Salia, Hugging Face)

#### MPS 프리필 어텐션 가속 / MPS Prefill Attention Acceleration

macOS 26.2에서 새로 도입된 Apple의 Metal Performance Primitives(MPP)는 그동안 Metal 커널에서 쓸 수 없었던, 어텐션 계열 워크로드를 위한 더 낮은 수준의 구성 요소를 노출합니다. 이번 릴리즈는 이를 활용해 MPS를 위한 두 번째 프리필 어텐션 커널을 도입합니다. MLX가 M5 칩에서 취한 접근을 이식하되 이전 세대 Apple Silicon까지 확장한 것으로, 헤드 차원이 64, 96, 128, 256이고 쿼리 길이가 8보다 큰 fp16/bf16 입력을 대상으로 합니다(macOS 26.2 이상, 다른 shape과 dtype은 기존 simdgroup 행렬 커널을 계속 사용합니다). 성과는 상당합니다. 작성자 본인의 벤치마크에서는 헤드 차원과 시퀀스 길이 전반에 걸쳐 이전 커널 대비 대략 2~4배의 속도 향상을 보였고, 헤드 차원이 작고 시퀀스가 길수록 이득이 컸습니다. Apple Silicon에서 어텐션 비중이 큰 모델을 돌리는 사람이라면 코드 변경 없이 의미 있는 프리필 속도 향상을 얻습니다. shape과 dtype이 조건을 만족하면 MPS가 알아서 더 빠른 커널을 고릅니다. 이 커널의 성능 이점은 Apple M5 하드웨어에서 가장 잘 확인되는데, 바탕이 되는 레인별(per-lane) 데이터 레이아웃이 그 하드웨어를 기준으로 검증됐기 때문입니다. 이전 세대 Apple Silicon에서 어떻게 동작하는지는 이번 릴리즈에서 독립적으로 검증되지 않았으며 위 수치와 다를 수 있습니다.
> Apple's Metal Performance Primitives (MPP), new in macOS 26.2, expose lower-level building blocks for attention-style workloads that weren't previously available to Metal kernels. This release takes advantage of them with a second prefill attention kernel for MPS, porting the approach MLX uses on M5 chips but extending it to older Apple silicon generations as well, for fp16/bf16 inputs with head dims of 64, 96, 128, or 256 and query length greater than 8 (macOS 26.2+; other shapes and dtypes keep using the existing simdgroup-matrix kernel). The payoff is substantial — the author's own benchmarks show roughly 2–4x speedups over the previous kernel across head dims and sequence lengths, with the largest gains at smaller head dims and longer sequences. For anyone running attention-heavy models on Apple silicon, this is a meaningful prefill speedup with no code changes required — MPS just picks the faster kernel automatically when the shape and dtype qualify. This kernel's performance benefits are best confirmed on Apple M5 hardware, which is what the underlying per-lane data layout was validated against; behavior on earlier Apple silicon generations has not been independently verified for this release and may not reflect the numbers above.

API 안정성: Unstable
> API Unstable

(PR [#182256](https://github.com/pytorch/pytorch/pull/182256) 작성: Irakli Salia, Hugging Face)
> (PR [#182256](https://github.com/pytorch/pytorch/pull/182256) by Irakli Salia, Hugging Face)

#### CTC 손실에 대한 MPS 가속 / MPS acceleration for CTC Loss

음성 인식이나 OCR처럼 정렬이 필요 없는 시퀀스 모델을 떠받치는 손실인 `ctc_loss`가 처음으로 MPS에서 순방향과 역방향 패스를 모두 지원합니다. 이 연산 하나 때문에 Mac 사용자가 CPU로 폴백해야 했던 구멍을 메운 것입니다. 구현은 가변 길이(패딩된) 배치의 올바른 처리를 포함해 CUDA 커널과 동일한 로그 도메인 접근을 따릅니다. 이제 Apple Silicon에서 CTC 기반 모델을 종단 간으로 학습할 때 손실 계산을 위해 CPU로 내려갈 필요가 없습니다.
> `ctc_loss` — the loss behind alignment-free sequence models like speech recognition and OCR — gets both forward and backward passes on MPS for the first time, closing a gap that previously forced Mac users to fall back to CPU for this one op. The implementation follows the same log-domain approach as the CUDA kernel, including correct handling of variable-length (padded) batches. Training CTC-based models end-to-end on Apple silicon no longer requires dropping into CPU for the loss computation.

API 안정성: Unstable
> API Unstable

(PR [#187716](https://github.com/pytorch/pytorch/pull/187716) 및 [#188187](https://github.com/pytorch/pytorch/pull/188187) 작성: Kurt Mohler, OpenTeams)
> (PR [#187716](https://github.com/pytorch/pytorch/pull/187716) and [#188187](https://github.com/pytorch/pytorch/pull/188187) by Kurt Mohler, OpenTeams)

### XPU (Intel GPU) / XPU (Intel GPUs)

#### 향상된 XPU 그래프 성능 / Enhanced XPU Graph Performance

XPU Graph의 그래프 캡처와 재생 오버헤드를 줄여 실행 효율을 개선했고, Intel® Arc™ B 시리즈 및 그 이후 Intel GPU에서 그래프 기반 학습 및 추론 워크로드에 더 나은 성능을 제공합니다.
> Reduced graph capture and replay overhead in XPU Graph, improving execution efficiency and delivering better performance for graph-based training and inference workloads on Intel® Arc™ B-Series and newer Intel GPUs.

API 안정성: Unstable
> API Unstable

(PR [#188874](https://github.com/pytorch/pytorch/pull/188874) 작성: Jing Ma, Intel)
> (PR [#188874](https://github.com/pytorch/pytorch/pull/188874) by Jing Ma, Intel)

#### scaled\_mm에 대한 MXFP8 및 MXFP4 지원 / MXFP8 and MXFP4 Support for scaled\_mm

scaled\_mm에 MXFP8과 MXFP4 지원을 추가하여, 차세대 Intel GPU를 위한 소프트웨어 준비를 앞당기고 개발자가 새롭게 등장하는 낮은 정밀도 연산 형식에 맞춰 AI 워크로드를 준비할 수 있게 돕습니다.
> Added MXFP8 and MXFP4 support for scaled\_mm, enabling early software readiness for next-generation Intel GPUs and helping developers prepare AI workloads for emerging low-precision computation formats.

API 안정성: Unstable
> API Unstable

(PR [#181726](https://github.com/pytorch/pytorch/pull/181726), [#181727](https://github.com/pytorch/pytorch/pull/181727), [#187315](https://github.com/pytorch/pytorch/pull/187315) 작성: Carson Wang, Intel)
> (PR [#181726](https://github.com/pytorch/pytorch/pull/181726), [#181727](https://github.com/pytorch/pytorch/pull/181727), and [#187315](https://github.com/pytorch/pytorch/pull/187315) by Carson Wang, Intel)

#### 분산 AI 워크로드를 위한 대칭 메모리 / Symmetric Memory for Distributed AI Workloads

스케일업 배포를 위해 XPU 대칭 메모리 백엔드를 활성화하여, Intel GPU에서 비동기 텐서 병렬화(Async Tensor Parallelism, Async TP)를 가능하게 하고 더 확장성 있는 분산 AI 워크로드의 토대를 제공합니다.
> Enabled the XPU Symmetric Memory backend for scale-up deployments, unlocking Async Tensor Parallelism (Async TP) on Intel GPUs and providing the foundation for more scalable distributed AI workloads.

API 안정성: Unstable
> API Unstable

(PR [#185102](https://github.com/pytorch/pytorch/pull/185102) 작성: Cherry Zhang, Intel)
> (PR[#185102](https://github.com/pytorch/pytorch/pull/185102) by Cherry Zhang, Intel)

#### 프로세스별 세밀한 Intel GPU 메모리 추적 / Fine-grained Per-Process Intel GPU Memory Tracking

torch.xpu.list\_gpu\_processes()를 추가하여, Intel GPU 메모리 사용량을 프로세스 단위로 상세히 추적하고 보고할 수 있습니다.
> Added torch.xpu.list\_gpu\_processes(), enabling detailed tracking and reporting of Intel GPU memory usage on a per-process basis.

API 안정성: Unstable
> API Unstable

(PR [#185192](https://github.com/pytorch/pytorch/pull/185192) 작성: Guangye Yu, Intel)
> (PR [#185192](https://github.com/pytorch/pytorch/pull/185192) by Guangye Yu, Intel)

#### WSL2 지원 확대 / Expanded WSL2 Support

Windows Subsystem for Linux 2(WSL2)에서 실행되는 Ubuntu 24.04 및 Ubuntu 26.04 지원을 추가하여, 개발자가 Windows 환경에서 Intel GPU용 AI 워크로드를 더 쉽게 빌드하고 실행할 수 있게 했습니다.
> Added support for Ubuntu 24.04 and Ubuntu 26.04 running under Windows Subsystem for Linux 2 (WSL2), making it easier for developers to build and run AI workloads on Intel GPUs from Windows environments.

### C++ ABI

#### 확장된 torch::stable 표면 / Expanded torch::stable Surface

`torch::stable`에 정의된 API 부분집합을 사용하는 C++ 애플리케이션은 릴리즈 간 ABI 호환성에 의존할 수 있으며, 이번 릴리즈는 그 표면을 넓힙니다. (C++ 애플리케이션은 정기적으로 다시 빌드하거나 libtorch 버전을 고정할 의향이 있다면 `libtorch.so`의 전체 API 표면을 사용할 수도 있습니다.) 2.13의 `torch::stable::Generator`에 이어, 이번 릴리즈는 `PyObject`에서 `torch::stable::Tensor`로의 변환과 `Tensor::has_storage`, 그리고 `bitwise_and`, `bitwise_or`, `left_shift`, `right_shift`, `permute`, `view_dtype`, `index_select`, `floor_divide`, `is_pinned`의 안정 오버로드를 추가합니다. 더 많은 유틸리티가 헤더 전용인 `torch::headeronly`로 옮겨졌고(`fastAtomicAdd`와 `isinf`/`isnan` 포함), 확장 작성자는 `libtorch`에 링크하지 않고도 이들을 쓸 수 있습니다.
> C++ applications that use the subset of APIs defined in `torch::stable` can rely on ABI compatibility across releases, and this release expands that surface. (C++ applications can also use the full API surface of `libtorch.so` if they're willing to rebuild regularly or pin their libtorch version.) Following `torch::stable::Generator` in 2.13, this release adds `PyObject`-to-`torch::stable::Tensor` conversion and `Tensor::has_storage`, plus stable overloads for `bitwise_and`, `bitwise_or`, `left_shift`, `right_shift`, `permute`, `view_dtype`, `index_select`, `floor_divide`, and `is_pinned`. More utilities have also moved into the header-only `torch::headeronly` (including `fastAtomicAdd` and `isinf`/`isnan`), so extension authors can use them without linking against `libtorch` at all.

C++ 인터페이스는 API 안정성 Unstable, C 인터페이스는 API 안정 및 ABI 안정입니다.
> C++ interface API Unstable while C interface is API Stable and ABI Stable.

(PR [#183323](https://github.com/pytorch/pytorch/pull/183323) 작성: Paweł Gadziński, NVIDIA, [#189877](https://github.com/pytorch/pytorch/pull/189877), [#191973](https://github.com/pytorch/pytorch/pull/191973), [#193604](https://github.com/pytorch/pytorch/pull/193604) 작성: Jane Xu, Meta, [#192083](https://github.com/pytorch/pytorch/pull/192083) 및 [#192097](https://github.com/pytorch/pytorch/pull/192097) 작성: Chris Leonard, Red Hat)
> (PR [#183323](https://github.com/pytorch/pytorch/pull/183323) by Paweł Gadziński, NVIDIA, [#189877](https://github.com/pytorch/pytorch/pull/189877), [#191973](https://github.com/pytorch/pytorch/pull/191973) and [#193604](https://github.com/pytorch/pytorch/pull/193604) by Jane Xu, Meta, [#192083](https://github.com/pytorch/pytorch/pull/192083) and [#192097](https://github.com/pytorch/pytorch/pull/192097) by Chris Leonard, Red Hat)

## 프로파일링 및 디버깅 / Profiling and Debugging

#### 고정 CPU 메모리에 대한 메모리 스냅샷 / Memory Snapshots for Pinned CPU Memory

메모리 스냅샷은 한동안 디바이스 할당을 다뤄 왔지만, 호스트에서 디바이스로의 전송을 준비하는 데 쓰이는 고정(page-locked) 호스트 메모리는 다루지 않았습니다. 그래서 그 메모리가 예상치 못하게 늘어나도, 이미 손에 익은 그 도구로는 볼 방법이 없었습니다. 고정 버퍼는 놓치기도 쉽습니다. CUDA 그래프는 캡처된 복사에 대해 고정된 호스트 주소를 요구하므로, 보통 한 번 할당해 프로세스 수명 내내 붙들고 있기 때문입니다. 이제 `torch.cuda.memory._record_memory_history()`에 `record_host=True`를 넘기면 고정 할당도 함께 포착되어, 기존 디바이스 데이터 옆에 새로운 `host_segments`와 `host_traces` 키로 드러납니다. 이로써 호스트와 디바이스 메모리를 하나의 일관된 관점에서 보며 누수나 예상치 못한 증가를 추적할 수 있습니다. `memory_viz` 시각화 도구는 아직 호스트 데이터를 렌더링하지 않으며, PyTorch의 할당자 바깥에서 원시 `cudaHostRegister` 호출로 만든 할당은 포착되지 않습니다.
> Memory snapshots have covered device allocations for a while, but not the pinned (page-locked) host memory used to stage host-to-device transfers — so if that memory grew unexpectedly, there was no way to see it in the same tool you'd already reach for. Pinned buffers are also easy to lose track of: they're typically allocated once and held for the life of a process, since CUDA graphs require a fixed host address for any captured copy. Passing `record_host=True` to `torch.cuda.memory._record_memory_history()` now captures pinned allocations too, surfaced as new `host_segments` and `host_traces` keys alongside the existing device data. This gives host and device memory a single, consistent view for tracking down leaks or unexpected growth. The `memory_viz` visualizer does not yet render host data, and allocations made via a raw `cudaHostRegister` call outside PyTorch's allocator are not captured.

API 안정성: Unstable
> API Unstable

(PR [#182407](https://github.com/pytorch/pytorch/pull/182407) 작성: Edward Yang, Meta)
> (PR [#182407](https://github.com/pytorch/pytorch/pull/182407) by Edward Yang, Meta)

## 지원 중단 및 하위 호환성 변경 사항 / Deprecations and Backwards-Incompatible Changes

- **TorchScript 지원 중단 경고가 이제 보이게** 되었고, TorchScript는 import 경로 밖으로 빠집니다. 지원이 중단된 `isIntegral` 오버로드는 제거됐습니다. [#189914](https://github.com/pytorch/pytorch/pull/189914)와 [#187115](https://github.com/pytorch/pytorch/pull/187115)를 참고하세요.
- Python 함수 이벤트가 기본적으로 프로파일러의 `key_averages()`에서 제외되며, 이는 프로파일러 출력에서 눈에 보이는 변화입니다. [#188631](https://github.com/pytorch/pytorch/pull/188631)을 참고하세요.
- 프로파일러에서 지원이 중단된 `use_cuda` 옵션이 제거되고 `with_modules`가 지원 중단됐으며, 패턴 매처, `BasicEvaluation`, `profiler_metrics`, `profiler_measure_per_kernel`이 제거됐습니다. [#192543](https://github.com/pytorch/pytorch/pull/192543), [#192808](https://github.com/pytorch/pytorch/pull/192808), [#187362](https://github.com/pytorch/pytorch/pull/187362), [#187439](https://github.com/pytorch/pytorch/pull/187439), [#187204](https://github.com/pytorch/pytorch/pull/187204)를 참고하세요.
- Dynamo TVM 백엔드의 Relay 경로가 `FutureWarning` 지원 중단 절차를 거쳐 제거됐습니다. relax 프론트엔드를 사용하세요. [#189639](https://github.com/pytorch/pytorch/pull/189639)와 [#190766](https://github.com/pytorch/pytorch/pull/190766)을 참고하세요.
- 분산에서는 `_set_pg_timeout`이 `torch.distributed.set_timeout`으로 대체되고, `setSequenceNumberForGroup`은 지원 중단된 no-op이 되며, 제어 집합 통신(control collectives) 구현이 제거되고, compile-on-one-rank의 `torch.distributed` 별칭은 `torch.compiler.config`로 대체됩니다. [#187387](https://github.com/pytorch/pytorch/pull/187387), [#188611](https://github.com/pytorch/pytorch/pull/188611), [#188617](https://github.com/pytorch/pytorch/pull/188617), [#187869](https://github.com/pytorch/pytorch/pull/187869)를 참고하세요.
- CUDA 그린 컨텍스트(green context)의 `set`과 `pop`이 지원 중단되고, 그린 컨텍스트는 CUDA Python 바인딩으로 옮겨 갔습니다. [#188419](https://github.com/pytorch/pytorch/pull/188419)와 [#185527](https://github.com/pytorch/pytorch/pull/185527)을 참고하세요.
- 희소(sparse) 텐서는 `weights_only`로 로드할 때 일관성 검증을 거칩니다. [#184750](https://github.com/pytorch/pytorch/pull/184750)을 참고하세요.
- `linear_cross_entropy`에서 `balanced` 정확도 정책이 제거됐습니다. [#188283](https://github.com/pytorch/pytorch/pull/188283)을 참고하세요.

> - **TorchScript deprecation warnings are now visible** and TorchScript is kept out of import paths. Deprecated `isIntegral` overloads are removed. See [#189914](https://github.com/pytorch/pytorch/pull/189914) and [#187115](https://github.com/pytorch/pytorch/pull/187115).
> - Python function events are excluded from profiler `key_averages()` by default, a visible change in profiler output. See [#188631](https://github.com/pytorch/pytorch/pull/188631).
> - In the profiler, the deprecated `use_cuda` option is removed and `with_modules` is deprecated; the pattern matcher, `BasicEvaluation`, `profiler_metrics` and `profiler_measure_per_kernel` are removed. See [#192543](https://github.com/pytorch/pytorch/pull/192543), [#192808](https://github.com/pytorch/pytorch/pull/192808), [#187362](https://github.com/pytorch/pytorch/pull/187362), [#187439](https://github.com/pytorch/pytorch/pull/187439) and [#187204](https://github.com/pytorch/pytorch/pull/187204).
> - The Dynamo TVM backend's Relay path is removed after a `FutureWarning` deprecation; use the relax frontend. See [#189639](https://github.com/pytorch/pytorch/pull/189639) and [#190766](https://github.com/pytorch/pytorch/pull/190766).
> - In distributed, `_set_pg_timeout` gives way to `torch.distributed.set_timeout`, `setSequenceNumberForGroup` becomes a deprecated no-op, the control collectives implementation is removed, and the compile-on-one-rank `torch.distributed` alias gives way to `torch.compiler.config`. See [#187387](https://github.com/pytorch/pytorch/pull/187387), [#188611](https://github.com/pytorch/pytorch/pull/188611), [#188617](https://github.com/pytorch/pytorch/pull/188617) and [#187869](https://github.com/pytorch/pytorch/pull/187869).
> - CUDA green context `set` and `pop` are deprecated, and green contexts moved to the CUDA Python bindings. See [#188419](https://github.com/pytorch/pytorch/pull/188419) and [#185527](https://github.com/pytorch/pytorch/pull/185527).
> - Sparse tensors are validated for consistency when loaded with `weights_only`. See [#184750](https://github.com/pytorch/pytorch/pull/184750).
> - The `balanced` accuracy policy is removed from `linear_cross_entropy`. See [#188283](https://github.com/pytorch/pytorch/pull/188283).

## 기능 외 업데이트 / Non-Feature Updates

| **구성 요소 / Component** | **2.13** | **2.14** |
| --- | --- | --- |
| CUDA | 12.6, 13.0, 13.2 | 12.6, 13.0, 13.2 |
| 기본 휠 / Default wheel | CUDA 13.0 | CUDA 13.0, 변경 없음 |
| ROCm | 7.1, 7.2 | 7.2, 7.14. 7.1 제외, 7.14는 TheRock 사용 |
| Python | 3.10 ~ 3.15, 3.14t와 3.15t 포함 | 변경 없음 |
| C++ 표준 / C++ standard | C++20 | 변경 없음 |

- 빌드 시스템이 setuptools에서 scikit-build-core로 이전했고, Windows와 macOS 휠 빌드가 Python 파이프라인으로 리팩터링됐습니다. [#180247](https://github.com/pytorch/pytorch/pull/180247), [#184407](https://github.com/pytorch/pytorch/pull/184407), [#187944](https://github.com/pytorch/pytorch/pull/187944)를 참고하세요.
- ROCm 7.14 휠은 RPATH 기반 라이브러리 해석과 함께 TheRock pip SDK로 빌드되고, manywheel은 4GB가 넘는 휠에서 발생하던 잘못된 ZIP64를 고치기 위해 auditwheel로 재패키징되며, 도구는 `rocm_smi`에서 `amd_smi`로 이전합니다. [#190276](https://github.com/pytorch/pytorch/pull/190276), [#189903](https://github.com/pytorch/pytorch/pull/189903), [#190014](https://github.com/pytorch/pytorch/pull/190014)를 참고하세요.
- cuDNN이 conv 엔진 5를 다시 활성화한 9.24로, oneDNN이 3.12.3으로, XPU 지원 패키지가 2026.1로 올라갑니다. [#189483](https://github.com/pytorch/pytorch/pull/189483), [#188785](https://github.com/pytorch/pytorch/pull/188785), [#189593](https://github.com/pytorch/pytorch/pull/189593)을 참고하세요.
- C++20이 최소 표준으로 유지되며, 헤더 가드 강제 적용이 완료됐습니다. [#178150](https://github.com/pytorch/pytorch/pull/178150)을 참고하세요.
- 새로운 CI 및 플랫폼 지원으로 네이티브 linux-riscv64 빌드 이미지, B200 벤치마크 워크플로우, P2P IPC 테스트를 위한 전용 H100 패브릭 러너, Intel BMG 클라이언트 스모크 테스트가 추가됐습니다. [#190887](https://github.com/pytorch/pytorch/pull/190887), [#192659](https://github.com/pytorch/pytorch/pull/192659), [#191280](https://github.com/pytorch/pytorch/pull/191280), [#187421](https://github.com/pytorch/pytorch/pull/187421)을 참고하세요.
- Inductor가 튜닝된 벡터화 원소별 커널로 Rubin(sm\_107)을 타겟팅합니다. [#190654](https://github.com/pytorch/pytorch/pull/190654)와 [#190546](https://github.com/pytorch/pytorch/pull/190546)을 참고하세요.

> - The build system migrated from setuptools to scikit-build-core, and Windows and macOS wheel builds are refactored into Python pipelines. See [#180247](https://github.com/pytorch/pytorch/pull/180247), [#184407](https://github.com/pytorch/pytorch/pull/184407) and [#187944](https://github.com/pytorch/pytorch/pull/187944).
> - ROCm 7.14 wheels are built from the TheRock pip SDK with RPATH-based library resolution, manywheels are repackaged with auditwheel to fix invalid ZIP64 on wheels over 4 GB, and tooling migrates from `rocm_smi` to `amd_smi`. See [#190276](https://github.com/pytorch/pytorch/pull/190276), [#189903](https://github.com/pytorch/pytorch/pull/189903) and [#190014](https://github.com/pytorch/pytorch/pull/190014).
> - cuDNN advances to 9.24 with conv engine 5 re-enabled, oneDNN to 3.12.3, and the XPU support package to 2026.1. See [#189483](https://github.com/pytorch/pytorch/pull/189483), [#188785](https://github.com/pytorch/pytorch/pull/188785) and [#189593](https://github.com/pytorch/pytorch/pull/189593).
> - C++20 remains the minimum standard, and header-guard enforcement completes. See [#178150](https://github.com/pytorch/pytorch/pull/178150).
> - New CI and platform coverage includes a native linux-riscv64 build image, a B200 benchmark workflow, a dedicated H100 fabric runner for P2P IPC tests, and Intel BMG client smoke tests. See [#190887](https://github.com/pytorch/pytorch/pull/190887), [#192659](https://github.com/pytorch/pytorch/pull/192659), [#191280](https://github.com/pytorch/pytorch/pull/191280) and [#187421](https://github.com/pytorch/pytorch/pull/187421).
> - Inductor targets Rubin (sm\_107) with tuned vectorized elementwise kernels. See [#190654](https://github.com/pytorch/pytorch/pull/190654) and [#190546](https://github.com/pytorch/pytorch/pull/190546).
