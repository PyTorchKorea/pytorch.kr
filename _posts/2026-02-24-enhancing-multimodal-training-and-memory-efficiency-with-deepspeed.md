---
layout: blog_detail
title: "DeepSpeed로 멀티모달 학습과 메모리 효율성 향상하기"
author: Masahiro Tanaka (Anyscale), Olatunji Ruwase (Snowflake)
category: ["pytorch.org", "translation"]
org_title: "Enhancing Multimodal Training and Memory Efficiency with DeepSpeed"
org_link: https://pytorch.org/blog/enhancing-multimodal-training-and-memory-efficiency-with-deepspeed/
---

## 개요 / Overview

이 블로그에서는 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)의 두 가지 중요한 업데이트를 소개합니다: (1) 멀티모달, 다중 구성요소 모델의 효율적인 학습을 가능하게 하는 PyTorch 동일 backward API(비스칼라(non-scalar) backward 호출 포함)와, (2) 최대 메모리 사용량을 크게 줄이는 저정밀도(low-precision) 모델 학습입니다.
> This blog walks through two crucial [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) updates: (1) a PyTorch-identical backward API that enables efficient training of multimodal, multi-component models (including non-scalar backward calls), and (2) low-precision model training that significantly reduces peak memory, especially.

비전 인코더와 LLM을 결합하는 것과 같은 멀티모달 워크로드의 경우, 학습 루프가 복잡하고 다중 구성요소로 이루어질 수 있습니다. 첫 번째 업데이트는 이러한 루프 작성을 간단하게 만드는 PyTorch 동일 backward API를 도입하여, DeepSpeed가 다양한 성능 최적화를 투명하게 관리하면서도 간단하고 깔끔한 코드로 정교한 병렬처리 방식을 구현할 수 있게 합니다. 한 가지 예로, 이 API의 유연성을 통해 [분리 하이브리드 병렬처리(disaggregated hybrid parallelism)](https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid)가 가능해졌으며, 멀티모달 AI 모델 학습에서 30%의 속도 향상을 달성하는 동시에 DeepSpeed를 사용한 모델 개발이 "일반 PyTorch(vanilla PyTorch)"에 더 가깝게 느껴지도록 했습니다.
> For multimodal workloads, like combining a vision encoder with an LLM, training loops can become complex and multi-component. The first update introduces a PyTorch-identical backward API that makes writing such loops straightforward, enabling sophisticated parallelism schemes with simple, clean code, while DeepSpeed transparently manages various performance optimizations. As one example, the flexibility of the API enabled [disaggregated hybrid parallelism](https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid), achieving a 30% speedup for multimodal AI model training while making model development with DeepSpeed feel closer to "vanilla PyTorch".

한편, LLM 미세 조정의 경우, 모든 모델 상태(매개변수, 변화도, 옵티마이저 상태)를 BF16이나 FP16과 같은 저정밀도로 유지하는 새로운 옵션이 메모리 사용량을 대폭 줄여, 제한된 하드웨어에서 더 큰 모델을 학습할 수 있게 합니다. 저정밀도 학습은 지도 미세 조정(SFT), 강화 학습(RL), 멀티모달 학습 등 다양한 응용 분야에서 매우 유용합니다. 실험 결과, 수치 안정성을 유지하면서 최대 메모리 사용량을 40% 줄일 수 있었습니다([벤치마킹 스크립트](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/bf16_master_weight)). 수치 안정성은 `torch.autocast`와의 통합을 통해 달성되며, 이를 통해 모델의 품질이 유지됩니다.
> Meanwhile, for LLM fine-tuning, a new option to keep all model states (parameters, gradients, and optimizer states) in lower-precision, such as BF16 or FP16, drastically reduces the memory footprint, allowing researchers to train larger models on more constrained hardware. Low-precision training is highly beneficial across a wide range of applications, including supervised fine-tuning (SFT), reinforcement learning (RL), and multimodal training. Our experiment showed 40% peak memory reduction while keeping numerical stability ([benchmarking script](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/bf16_master_weight)). The numerical stability is achieved through integration with torch.autocast, which ensures the quality of the model is maintained.

이 블로그의 나머지 부분에서는 이러한 업데이트가 최첨단 학습 워크로드 개발을 어떻게 직접적으로 촉진하는지 자세히 설명합니다.
> The remainder of this blog will elaborate on how these updates directly facilitate the development of cutting-edge training workloads.

## 1. PyTorch 동일 backward API / PyTorch-identical backward API

DeepSpeed는 이제 모든 최적화를 유지하면서 PyTorch의 네이티브 `backward()` 구문을 지원합니다. 기존에 DeepSpeed의 학습 루프는 엔진의 backward API에 의존했습니다:
> DeepSpeed now supports PyTorch's native `backward()` syntax while preserving all its optimizations. Traditionally, DeepSpeed's training loop relied on the engine's backward API:

```python
loss = model_engine(batch)
model_engine.backward(loss)
model_engine.step()
```

엔진의 `backward` API는 기존의 사전 학습 및 미세 조정 파이프라인에는 충분했습니다. 그러나 최근의 복잡한 학습 파이프라인은 더 많은 유연성을 필요로 합니다. 두 가지 주요 제한 사항이 있었습니다:
> The engine's `backward` API was sufficient for traditional pretraining and fine-tuning pipelines. However, recent complex training pipelines require more flexibility. There were two major limitations:

1. 스칼라 손실만 허용했습니다.
> 1. It only accepted a scalar loss.

2. 일반적인 PyTorch의 `loss.backward()` 방식 대신 `model_engine.backward(loss)`를 호출해야 했습니다.
> 2. You had to call `model_engine.backward(loss)`, rather than using the usual PyTorch `loss.backward()` style.

이러한 제약으로 인해 사용자는 일반 PyTorch에서 허용하는 패턴을 간단히 구현할 수 없었습니다. 몇 가지 예시는 다음과 같습니다:
> Due to these constraints, users could not simply implement patterns that vanilla PyTorch allows. Here are some examples:

```python
# 1. 여러 모델과 손실을 결합
output1 = model1(batch1)
output2 = model2(batch2)
loss = criterion(output1, output2)
loss.backward()

# 2. 주 모델과 별도로 손실 함수를 정의
output = model(batch)
loss = loss_fn(output)
loss.backward()

# 3. 커스텀 변화도로 비스칼라 Tensor에 대해 backward 호출
output = model(batch)
output.backward(grad)
```

DeepSpeed 엔진은 내부 API를 사용하여 이러한 사용 사례를 처리할 수 있었지만, 상당한 코드 변경이 필요했고 버그가 쉽게 발생할 수 있었습니다. PyTorch 동일 backward API가 추가됨에 따라, 이제 ZeRO와 오프로딩을 포함한 DeepSpeed의 강력한 최적화를 유지하면서 네이티브 PyTorch와 동일한 코드를 사용할 수 있습니다.
> DeepSpeed Engine was able to handle these use cases using internal APIs; however, that required significant code changes and could easily introduce bugs. With the addition of PyTorch-identical backward API, we can now use the same code as native PyTorch while keeping DeepSpeed's powerful optimizations, including ZeRO and offloading.

PyTorch 동일 backward API의 한 가지 사용 사례는 [Ray](https://github.com/ray-project/ray)를 사용한 멀티모달 모델의 [분리 하이브리드 병렬처리](https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid)입니다. 이 학습 파이프라인에서는 두 개의 Ray Actor 그룹이 비전 인코더와 LLM을 각각 처리합니다. Backward 패스에서 LLM이 비전 인코더에 변화도를 전달하고, 비전 인코더는 해당 변화도로 backward 함수를 호출합니다. 그러나 변화도가 비스칼라 Tensor이기 때문에, 이러한 사용 사례는 DeepSpeed API에서 공식적으로 지원되지 않았습니다. 분리 하이브리드 병렬처리는 backward API의 유연성과 DeepSpeed의 최적화, 그리고 [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)(고효율 시퀀스 병렬처리)를 결합하여 학습에서 30%의 속도 향상을 달성함을 보여줍니다.
> One example use case for the PyTorch-identical backward API is [disaggregated hybrid parallelism](https://www.anyscale.com/blog/30-faster-multimodal-ai-training-with-ray-and-disaggregated-hybrid) for multimodal models using [Ray](https://github.com/ray-project/ray). In this training pipeline, two Ray Actor groups handle the vision encoder and the LLM separately. On a backward pass, the LLM passes a gradient to the vision encoder, and the vision encoder calls the backward function with that gradient. However, because the gradient is a non-scalar tensor, such a use case wasn't officially supported by DeepSpeed APIs. The disaggregated hybrid parallelism demonstrates that the flexibility of the backward API combined with DeepSpeed's optimization and [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509) (highly efficient sequence parallelism), achieves 30% speedup in training.

아래는 서로 다른 Actor에서 실행되는 두 모델의 의사 코드입니다. 서로 다른 프로세스에서 실행되므로, Ray Actor 통신을 통해 변화도를 전달합니다. 여기서 볼 수 있듯이, 비전 임베딩의 변화도는 비스칼라 Tensor입니다. 이 코드는 PyTorch API와 동일하지만, 설정에 따라 다양한 DeepSpeed 최적화를 활성화합니다.
> Below is the pseudo-code for the two models running on different actors. Since they run in different processes, we pass gradients via Ray actor communication. As seen here, the gradient of the vision embedding is a non-scalar tensor. Although this code is identical to the PyTorch API, it will activate various DeepSpeed optimizations based on your configuration.

```python
# LLM Actor에서 실행
def text_backward_step(self):
  # ...
  self.loss.backward()
  return self.vision_embeddings.grad.detach().clone()

# Vision Actor에서 실행
def vision_backward_step(self, vision_embedding_grad):
  self.vision_output.backward(gradient=vision_embedding_grad)
```

전체 학습 파이프라인은 [리포지토리](https://github.com/ray-project/multimodal-training)에서 확인할 수 있습니다.
> Check out the [repository](https://github.com/ray-project/multimodal-training) for the complete training pipeline.

## 2. 메모리 효율적인 저정밀도 모델 상태 / Memory-efficient low-precision model states

이제 모든 모델 상태(매개변수, 변화도, 옵티마이저 상태)를 BF16 또는 FP16으로 유지하여 메모리 소비를 크게 줄일 수 있습니다.
> You can now keep all model states (parameters, gradients, and optimizer states) in BF16 or FP16, significantly reducing memory consumption.

기존 DeepSpeed의 혼합 정밀도 방식은 FP32 마스터 매개변수, 변화도, 옵티마이저 상태를 유지했는데, 이는 기술적으로 더 안전하지만 메모리를 많이 사용합니다. DeepSpeed는 설정을 통해 `torch.autocast`를 지원해왔지만(API 문서 참조), FP32 상태 생성을 우회하는 옵션이 없어 제한된 하드웨어에서 대규모 모델의 학습 가능성이 제한되었습니다. 실제로 많은 학습 워크로드는 FP32 상태 없이도 안정적으로 수렴합니다.
> Traditionally, DeepSpeed's mixed precision keeps FP32 master parameters, gradients, and optimizer states, which is technically safer but memory-intensive. While DeepSpeed has supported `torch.autocast` via configuration (see the API documentation), the lack of an option to bypass creating FP32 states limited the trainability of large models on constrained hardware. In practice, many training workloads converge stably without FP32 states.

저정밀도 모델 상태 옵션을 사용하면, FP32 상태 생성을 쉽게 건너뛰고 저정밀도 옵션을 `torch.autocast` 지원과 결합할 수 있습니다(설정 세부 사항은 문서와 예제를 참조하세요). 이 조합은 수렴성을 희생하지 않으면서 메모리 효율성을 크게 향상시킵니다.
> With the low-precision model states option, you can easily skip creating FP32 states and combine the low-precision option with `torch.autocast` support (see the document and example for configuration details). This combination drastically improves memory efficiency without sacrificing convergence.

```json
{
  ...
  "zero_optimization": {
    "stage": 3,
    ...
  },
  "bf16": {
    "enabled": true,
    "bf16_master_weights_and_grads": true,
    "bf16_optimizer_states": true
  },
  "torch_autocast": {
    "enabled": true,
    "dtype": "bfloat16"
  }
}
```

[예제 스크립트](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/bf16_master_weight)를 통해 상당한 메모리 절감을 확인할 수 있습니다:
> Our [example script](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/training/bf16_master_weight) demonstrates the significant memory savings:

| 설정 / Configuration | 할당 메모리 / Allocated Memory | 최대 메모리 / Peak Memory | 평균 스텝 시간 / Avg Step Time |
|---|---|---|---|
| 기준선 (fp32 마스터) / Baseline (fp32 master) | 25.74 GB | 31.38 GB | 0.6016s |
| BF16 저정밀도 (마스터 + 옵티마이저 상태) / BF16 low-precision (master + opt states) | **16.17 GB** | **18.93 GB** | 0.6427s |

실험(7B 모델, ZeRO3, 4 GPU)에서 **최대 메모리 사용량 40% 감소**를 달성했습니다. BF16 저정밀도 학습이 수치 안정성을 유지하는지 검증하기 위해, Wikitext-103 데이터셋에서 1000 스텝 동안 학습했습니다:
> The experiment (7B model, ZeRO3, 4GPUs) demonstrated **40% reduction in peak memory**. To verify that BF16 low-precision training maintains numerical stability, we trained for 1000 steps on the Wikitext-103 dataset:

![손실 곡선 비교 / Loss curve comparison](/assets/blog/2026-02-24-enhancing-multimodal-training-and-memory-efficiency-with-deepspeed/loss-curve-comparison.png){:style="width:100%"}
*손실 곡선 비교 / Loss curve comparison*

| 설정 / Configuration | 최종 손실 / Final Loss | 평균 손실 / Mean Loss |
|---|---|---|
| 기준선 (fp32 마스터) / Baseline (fp32 master) | 3.09 | 2.78 |
| BF16 저정밀도 / BF16 Low-Precision | 3.12 | 2.90 |

## 관련 테스트 / Related Tests

이러한 새로운 API는 CI에서 지속적으로 테스트되고 있으며, 테스트에서 다양한 사용 사례 패턴을 확인할 수 있습니다.
> We continuously test these new APIs in our CI, and you can see various use-case patterns in the tests.

- [PyTorch 호환 backward API / PyTorch-compatible backward API](https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/zero/test_zero_user_backward.py)
- [저정밀도 마스터 매개변수/변화도/옵티마이저 상태 / Low-precision master params/grads/optimizer states](https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/half_precision/test_bf16.py)
- [`torch.autocast`와의 결합 / Combination with torch.autocast](https://github.com/deepspeedai/DeepSpeed/tree/master/tests/unit/v1/half_precision/test_with_autocast.py)

## 마무리 / Closing Thoughts

이번 DeepSpeed 업데이트는 핵심적인 발전을 제공합니다:
> This DeepSpeed update delivers key advancements:

- **복잡한 멀티모달 워크로드 지원**: 새로운 PyTorch 동일 backward API는 멀티모달 모델에 필요한 정교한 다중 구성요소 학습 루프를 간단하고 깔끔한 코드로 구현할 수 있게 합니다. 한 가지 예로, PyTorch 동일 backward API를 통해 분리 하이브리드 병렬처리에서 30%의 속도 향상을 달성했습니다.
> - **Enabling Complex Multimodal Workloads**: The new PyTorch-identical backward API enables sophisticated multi-component training loops, such as those required for multimodal models, with simple, clean code. As one example, the PyTorch-identical backward API has enabled a 30% speedup for disaggregated hybrid parallelism.

- **더 큰 모델로의 확장**: 저정밀도 모델 상태와 `torch.autocast`를 결합하면 수렴성을 희생하지 않으면서 최대 메모리 사용량을 최대 40%까지 줄여, 동일한 하드웨어에서 더 큰 모델을 학습할 수 있습니다.
> - **Scaling to Larger Models:** Low-precision model states combined with `torch.autocast` reduce peak memory by up to 40% without sacrificing convergence, allowing you to train larger models with the same hardware.

이 블로그에서 설명한 새로운 API와 기능을 여러분의 학습 환경에서 어떻게 활용하시는지 기대됩니다. 사용해보시면서 [GitHub](https://github.com/deepspeedai/DeepSpeed)에 피드백과 이슈를 남겨주시기 바랍니다.
> We are excited to see how you use the new APIs and features described in this blog post in your own training setups, and we welcome feedback and issues on [GitHub](https://github.com/deepspeedai/DeepSpeed) as you try them out.
