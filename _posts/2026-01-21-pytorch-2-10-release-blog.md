---
layout: blog_detail
title: "PyTorch 2.10 출시 공지"
author: PyTorch Foundation
category: ["pytorch.org", "translation"]
org_title: "PyTorch 2.10 Release Blog"
org_link: https://pytorch.org/blog/pytorch-2-10-release-blog/
---

PyTorch® 2.10([릴리즈 노트](https://github.com/pytorch/pytorch/releases/tag/v2.10.0))의 출시를 발표하게 되어 기쁩니다! 이번 릴리즈는 성능 및 수치 디버깅(numerical debugging)에 대한 다양한 개선 사항을 제공합니다. 성능은 2.0에서 도입된 PyTorch 컴파일러 스택의 기능을 기반으로, 2.x 릴리즈 시리즈 전반에 걸쳐 PyTorch의 핵심 목표였습니다. 결정론성(determinism) 및 수치 디버깅은 더 많은 모델들이 사후 학습(post-training)에 분산 강화 학습 워크플로우를 사용하면서 더욱 중요해졌습니다.
> We are excited to announce the release of PyTorch® 2.10 ([release notes](https://github.com/pytorch/pytorch/releases/tag/v2.10.0))! This release features a number of improvements for performance and numerical debugging. Performance has been a focus for PyTorch throughout the 2.x release series, building on the capabilities of the PyTorch compiler stack introduced in 2.0. Determinism and Numerical Debugging have become more important as more models are being post-trained using distributed reinforcement learning workflows.

PyTorch 2.10 릴리즈의 주요 변경 사항은 다음과 같습니다:
> The PyTorch 2.10 release features the following changes:

- `torch.compile()`에서 **Python 3.14** 지원. Python 3.14t(프리스레드 빌드)도 실험적으로 지원됩니다.
- torchinductor의 **콤보 커널(combo-kernels)** 수평 퓨전을 통한 커널 실행 오버헤드 감소
- 비정형(ragged) 및 패킹된 시퀀스를 지원하는 새로운 **`varlen_attn()`** 연산
- **DnXgeev**를 활용한 효율적인 고유값 분해(eigenvalue decomposition)
- **Intel GPU** 성능 최적화 및 기능 개선
- `torch.compile()`의 **use_deterministic_mode** 준수
- 디스패치된 호출 추적 및 수치 발산 디버깅을 위한 **DebugMode** – 미묘한 수치 버그를 더 쉽게 추적할 수 있습니다.

> - **Python 3.14** support for *torch.compile()*. Python 3.14t (freethreaded build) is experimentally supported as well.
> - Reduced kernel launch overhead with **combo-kernels** horizontal fusion in torchinductor
> - A new **varlen_attn()** op providing support for ragged and packed sequences
> - Efficient eigenvalue decompositions with **DnXgeev**
> - **Intel GPU** performance optimizations and feature enhancements
> - *torch.compile()* now respects **use_deterministic_mode**
> - **DebugMode** for tracking dispatched calls and debugging numerical divergence – This makes it simpler to track down subtle numerical bugs.

이번 릴리즈는 PyTorch 2.9 이후 536명의 기여자로부터 4,160회의 커밋으로 구성되었습니다. 헌신적인 커뮤니티의 기여에 진심으로 감사드립니다. 언제나 그렇듯, 새로운 기능들을 사용해보시고 2.10을 개선하는 데 도움이 되도록 문제를 보고해 주시기 바랍니다. PyTorch 2 시리즈를 시작하는 방법에 대한 자세한 정보는 [시작하기](https://pytorch.org/get-started/pytorch-2.0/) 페이지에서 확인할 수 있습니다.
> This release is composed of 4160 commits from 536 contributors since PyTorch 2.9. We want to sincerely thank our dedicated community for your contributions. As always, we encourage you to try these out and report any issues as we improve 2.10. More information about how to get started with the PyTorch 2-series can be found at our [Getting Started](https://pytorch.org/get-started/pytorch-2.0/) page.

1월 28일 수요일에 Andrey Talaman, Nikita Shulga, Shangdi Yu가 2.10의 새로운 사항을 소개하는 짧은 라이브 세션을 진행합니다. 릴리즈 주기 변경, TorchScript 지원 중단, `torch.compile`의 Python 3.14 지원, DebugMode와 `tlparse` 등을 다루며, 이어서 실시간 Q&A가 진행됩니다. [참석 등록하기.](https://streamyard.com/watch/2PZFMH8puHwa)
> On Wednesday, January 28, Andrey Talaman, Nikita Shulga, and Shangdi Yu will host a short live session to walk through what's new in 2.10, including updates to the release cadence, TorchScript deprecation, `torch.compile` support for Python 3.14, DebugMode and `tlparse`, and more, followed by a live Q&A. [Register to attend.](https://streamyard.com/watch/2PZFMH8puHwa)


## 성능 관련 API-UNSTABLE 기능 / Performance Related API-UNSTABLE Features

### 콤보 커널: torchinductor에서의 수평 퓨전을 통한 커널 실행 오버헤드 감소 / Reduced kernel launch overhead with Combo-kernels horizontal fusion in torchinductor

콤보 커널(Combo Kernels)은 데이터 의존성이 없는 여러 독립적인 연산을 하나의 통합 GPU 커널로 결합하는 수평 퓨전(horizontal fusion) 최적화입니다. 순차적 연산을 퓨전하는 수직 퓨전(vertical fusion, 생산자-소비자)과 달리, 콤보 커널은 병렬 연산을 퓨전합니다. [RFC에서 예시를 확인하세요.](https://github.com/pytorch/pytorch/issues/170268)
> Combo Kernels is a horizontal fusion optimization that combines multiple independent operations with no data dependencies into a single unified GPU kernel. Unlike vertical fusion (producer-consumer), which fuses sequential operations, combo kernels fuse parallel operations. See example [here in the RFC](https://github.com/pytorch/pytorch/issues/170268).

![콤보 커널 다이어그램 / Combo kernels diagram](/assets/blog/2026-01-21-pytorch-2-10-release-blog/1-3.png){:style="width:100%"}

### varlen_attn() – 가변 길이 어텐션 / varlen_attn() – Variable length attention

`varlen_attn()`이라는 비정형(ragged) 및 패킹된 시퀀스를 위한 새로운 `torch.nn.attention` 연산이 제공됩니다. 이 API는 순전파(forward) + 역전파(backward)를 지원하며 `torch.compile`과 호환됩니다. 현재 FA2를 백엔드로 사용하며, cuDNN 및 FA4 지원이 계획되어 있습니다. NVIDIA CUDA A100 GPU 이상에서 사용 가능하며, BF16 및 FP16 데이터 타입을 지원합니다. 자세한 내용은 [API 문서](https://docs.pytorch.org/docs/main/nn.attention.varlen.html#module-torch.nn.attention.varlen) 및 [튜토리얼](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html)을 참고하세요.
> A new torch.nn.attention op is provided for ragged / packed sequences called varlen_attn(). This API supports forward + backward, and is torch.compile-able. It's currently being supported by FA2 with plans to add cuDNN and FA4 support. This op is available on NVIDIA CUDA with an A100 GPU or newer, and supports BF16 and FP16 dtypes. To learn more, please see the [API doc](https://docs.pytorch.org/docs/main/nn.attention.varlen.html#module-torch.nn.attention.varlen) and [Tutorial](https://docs.pytorch.org/tutorials/intermediate/variable_length_attention_tutorial.html)

### DnXgeev를 활용한 효율적인 고유값 분해 / Efficient eigenvalue decompositions with DnXgeev

PyTorch linalg가 NVIDIA GPU에서 고효율 일반 고유값 분해(general eigenvalue decomposition)를 제공하기 위해 cuSOLVER의 DnXgeev를 사용하도록 확장되었습니다.
> PyTorch linalg has been extended to be able to use cuSOLVER's DnXgeev to provide highly efficient general eigenvalue decomposition on NVIDIA GPUs.

### Intel GPU 성능 최적화 및 기능 개선 / Intel GPU performance optimizations and feature enhancements

이번 릴리즈에서는 다음과 같은 주요 개선을 통해 Intel GPU 아키텍처의 기능 향상과 성능 최적화를 도입합니다:
> This latest release introduces feature enhancement and performance optimizations for Intel GPU architectures through the following key enhancements:

- Windows 및 Linux에서 최신 [Intel® Core™ Ultra Series 3(Intel® Arc™ Graphics 탑재)](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html)로 Intel GPU 지원 확대
- Intel GPU에서의 FP8 지원: 주요 기본 연산자(타입 승격, 형상 연산자 등)와 Tensor 및 채널 단위 스케일링 팩터를 사용한 스케일드 행렬 곱셈 구현
- Intel GPU에서의 Aten 연산자 MatMul 복소수 데이터 타입 지원 구현
- Windows에서 사용자가 새로운 커스텀 연산자를 구현할 수 있도록 [PyTorch CPP Extension API의 SYCL 지원](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html#cpp-custom-ops-tutorial-sycl) 확대
- Intel GPU 단위 테스트 커버리지 확대

> - Expand the Intel GPUs support to latest [Intel® Core™ Ultra Series 3 with Intel® Arc™ Graphics](https://www.intel.com/content/www/us/en/products/details/processors/core-ultra.html) on both Windows and Linux.
> - Implement FP8 support on Intel GPUs by adding commonly used basic operators (e.g., type promotion and shape operators) and scaled matrix multiplication using tensor- and channel-wise scaling factors.
> - Implement Aten operator MatMul complex data type support on Intel GPUs.
> - Extend [SYCL support in PyTorch CPP Extension API](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html#cpp-custom-ops-tutorial-sycl) to allow users to implement new custom operators in Windows.
> - Broaden Intel GPU UT coverage.


## 수치 디버깅 관련 API-UNSTABLE 기능 / Numerical Debugging Related API-UNSTABLE Features

### torch.compile()의 use_deterministic_mode 준수 / *torch.compile()* now respects use_deterministic_mode

실행 간(run-to-run) 결정론성은 학습 실행의 디버깅을 용이하게 하며, 대규모 환경에서 운영하는 사용자와 `torch.compile`을 사용하는 코드를 안정적으로 테스트하려는 사용자들에게 가장 많이 요청되었던 기능 중 하나입니다. 이제 `torch.use_deterministic_algorithms(True)`로 이 기능을 활성화할 수 있으며, `torch.compile`을 두 번 호출해도 정확히 동일한 연산을 수행하도록 보장합니다.
> Run-to-run determinism makes it easier to debug training runs and has been one of the most requested features for users running at scale and for users who want to reliably test their code that uses torch.compile. This can now be switched on using *torch.use_deterministic_algorithms(True)* and makes sure that two invocations of torch.compile will perform the same operations exactly.

```python
torch.use_deterministic_algorithms(True)
```

### 디스패치 호출 추적 및 수치 발산 디버깅을 위한 DebugMode / DebugMode for tracking dispatched calls and debugging numerical divergence

DebugMode는 프로파일링 스타일의 런타임 덤프를 제공하는 커스텀 TorchDispatchMode입니다. 수치 동치(numerical equivalence)의 중요성이 높아짐에 따라, Tensor 해싱을 통해 발산을 더 쉽게 격리할 수 있도록 개선되었습니다. Tensor 해싱을 사용하면 동일한 입력으로 모델의 두 버전을 실행하여 모든 Tensor가 동일한 해시를 가지는지 확인할 수 있습니다. 해시가 일치하다가 발산하기 시작하는 지점이 종종 다르게 동작하는 연산입니다.
> DebugMode is a custom TorchDispatchMode that provides profiling-style runtime dumps. With the growing importance of numerical equivalence, we've enhanced it so you can more easily isolate divergences with tensor hashing. With Tensor hashing, you can run two versions of a model with the same input, and you should see that all the tensors have the same hash. Where they go from matching to diverging is often the op that is behaving differently.

주요 기능:
> Key capabilities:

- **런타임 로깅** – 디스패치된 연산과 TorchInductor로 컴파일된 Triton 커널을 기록합니다.
- **Tensor 해싱** – 입력/출력에 결정론적 해시를 첨부하여 미묘한 수치 오류가 도입되는 지점을 더 쉽게 확인할 수 있습니다.
- **디스패치 훅(Dispatch hooks)** – 호출에 주석을 달기 위한 커스텀 훅 등록이 가능합니다.

> - Runtime logging – Records dispatched operations and TorchInductor compiled Triton kernels.
> - Tensor hashing – Attaches deterministic hashes to inputs/outputs to make it easier to see where subtle numerical errors are introduced.
> - Dispatch hooks – Allows registration of custom hooks to annotate calls

자세한 내용은 [여기 튜토리얼](https://docs.pytorch.org/tutorials/recipes/debug_mode_tutorial.html)에서 확인할 수 있습니다.
> More details in the tutorial [here](https://docs.pytorch.org/tutorials/recipes/debug_mode_tutorial.html).


## 비-기능 업데이트 / Non-Feature Updates

### TorchScript 지원 중단 / Torchscript is now Deprecated

TorchScript는 2.10에서 지원 중단(deprecated)되며, 대신 [torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html)를 사용해야 합니다. 자세한 내용은 PTC에서의 [발표 영상](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903)을 참고하세요.
> Torchscript is deprecated in 2.10, and [torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html) should be used instead. For more details, see [this talk](https://youtu.be/X2YbbDmCsOI?si=8s6Ue3BKIa_FYUne&t=903) from PTC.

### 컴파일러 이슈에 대한 명확한 버그 리포트를 위한 tlparse 및 TORCH_TRACE / tlparse & TORCH_TRACE can be used to submit clearer bug reports on compiler issues

PyTorch 개발자들이 독립적인 재현이 어려운 복잡한 이슈를 만났을 때, `tlparse` 결과는 GitHub에 쉽게 업로드하여 공유할 수 있는 로그 형식을 제공하는 효과적인 방법입니다. PyTorch 외부 개발자들도 이로부터 유용한 정보를 추출할 수 있으며, PyTorch 개발자에게 버그를 보고할 때 tlparse 로그 아티팩트를 첨부하는 것을 권장합니다. 자세한 내용은 [GitHub](https://github.com/meta-pytorch/tlparse) 및 [튜토리얼](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.observability.html)에서 확인할 수 있습니다.
> When PyTorch developers encounter complex issues for which standalone reproduction is difficult, tlparse results are a viable solution with a log format that is easy to upload and share on GitHub. Non-PyTorch developers can still extract useful information from it, and we encourage attaching tlparse log artifacts when reporting bugs to PyTorch developers. More details here on [Github](https://github.com/meta-pytorch/tlparse) & this [tutorial](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.observability.html).

### 2026년 릴리즈 주기 / 2026 Release Cadence

2026년부터 릴리즈 주기가 분기별에서 2개월에 1회로 빨라집니다. 공개된 릴리즈 [일정](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence)을 참고하세요.
> For 2026, the expected release cadence will be increased to 1 per 2 months, from quarterly. See the published release [schedule.](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-cadence)
