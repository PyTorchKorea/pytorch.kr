---
layout: blog_detail
title: "ExecuTorch와 Arm을 사용하여 PyTorch 모델을 마이크로 엣지에 배포하기"
author: Dominica Abena Oforiwaa Amanfo
category: ["pytorch.org", "translation"]
org_title: "Deploying PyTorch Models to the Micro-Edge with ExecuTorch and Arm"
org_link: https://pytorch.org/blog/deploying-pytorch-models-to-the-micro-edge-with-executorch-and-arm/
---

AI의 세계는 클라우드를 넘어 손바닥 안에 들어오는 장치로까지 확장되고 있습니다. 메모리가 킬로바이트 단위인 이러한 초소형 시스템에서 PyTorch 모델을 실행하려면 완전히 새로운 사고방식이 필요합니다. 바로 이 지점에서 엣지 추론을 위한 경량 런타임인 ExecuTorch가 익숙한 PyTorch 워크플로우와 저전력 Arm 기반 마이크로컨트롤러 사이의 간극을 연결해주며, 양자화(quantization)와 그래프 컴파일(graph compilation) 같은 최적화를 통해 모델을 엣지에서 실행할 수 있을 만큼 효율적으로 만들어줍니다.
> The world of AI is expanding beyond the cloud, reaching devices that fit in the palm of your hand. Running PyTorch models on these tiny systems, where memory is measured in kilobytes, requires a new way of thinking. That's where ExecuTorch, the lightweight runtime for edge inference, bridges the gap between familiar PyTorch workflows and low-power Arm-based microcontrollers, using optimizations such as quantization and graph compilation to make models efficient enough for the edge.

최근 [Arm Corstone-320 플랫폼](https://www.arm.com/products/silicon-ip-subsystems/corstone-320)에서 PyTorch와 ExecuTorch를 사용하여 간단한 가위바위보(Tiny Rock-Paper-Scissors, RPS) 데모를 만들었습니다. 목표는 PyTorch로 학습한 소형 합성곱 신경망(CNN)을 Arm Ethos-U NPU가 탑재된 시뮬레이션 Arm 마이크로컨트롤러([Arm FVP(Fixed Virtual Platform)](https://www.arm.com/products/development-tools/simulation/fixed-virtual-platforms) 사용)에 배포하는 것이었습니다. 이 여정이 어떻게 진행되었는지, 그리고 엣지에서 개발하는 사람들에게 왜 중요한지 소개합니다.
> I recently built a Tiny Rock-Paper-Scissors (RPS) demo using PyTorch and ExecuTorch on the [Arm Corstone-320 platform](https://www.arm.com/products/silicon-ip-subsystems/corstone-320). The goal: take a small Convolutional Neural Network (CNN) trained in PyTorch and deploy it all the way to a simulated Arm microcontroller with an Arm Ethos-U NPU (via the [Arm Fixed Virtual Platform (FVP)](https://www.arm.com/products/development-tools/simulation/fixed-virtual-platforms)). Here's what that journey looks like, and why it matters for anyone building at the edge.

## 엣지에서 PyTorch를 사용하는 이유 / Why PyTorch at the Edge?

PyTorch는 모델 실험을 빠르고 직관적으로 할 수 있게 해주지만, 동적 그래프의 유연성에서 임베디드 하드웨어의 엄격한 제약 조건으로 전환하는 것은 쉽지 않습니다. 대부분의 마이크로컨트롤러는 RAM이 1 MB 미만이고 운영체제가 없기 때문에, 기존의 Python 추론은 사용할 수 없습니다.
> PyTorch makes model experimentation fast and intuitive, but moving from the flexibility of dynamic graphs to the rigid constraints of embedded hardware isn't trivial. Most microcontrollers have less than 1 MB of RAM and no operating system, so traditional Python inference is off the table.

ExecuTorch는 PyTorch 모델을 최소한의 연산, 전력, 메모리로 동작하는 장치에서 실행할 수 있는 컴팩트하고 이식 가능한 형식(`.pte`)으로 컴파일하여 이 문제를 해결합니다. 이 과정에서 가중치와 활성화 값은 부동소수점에서 더 낮은 정밀도의 정수 형식(일반적으로 int8)으로 양자화되어, 모델 정확도를 유지하면서 메모리 사용량과 연산 비용을 획기적으로 줄여줍니다. 또한 연산 그래프를 평탄화(flatten)하고 융합(fuse)하며 최적화하여 불필요한 연산을 제거하고 엣지에서 원활한 실행이 가능하도록 합니다. ExecuTorch는 PyTorch 생태계를 가장 작은 Arm Cortex-M 및 Ethos-U 기반 시스템까지 확장합니다.
> ExecuTorch solves this by compiling PyTorch models into a compact, portable format (`.pte`) that runs on devices with minimal compute, power, and memory. During this process, weights and activations are quantized from floating-point to lower-precision integer formats (typically int8), dramatically reducing both memory footprint and compute costs while maintaining model accuracy. The computation graph is also flattened, fused, and optimized, removing redundant operations and enabling smooth execution at the edge. It extends the PyTorch ecosystem all the way down to the smallest Arm Cortex-M and Ethos-U-based systems.

## PyTorch에서 마이크로 엣지까지 / From PyTorch to the Micro-Edge

좋은 소식이 있습니다. 엔드투엔드(end-to-end) TinyML 엣지 AI 파이프라인을 안내하는 상세한 학습 경로를 구축해 두었습니다.
> The great news is, I have built a detailed learning path to guide you through an end-to-end TinyML EdgeAI pipeline.

### 가위바위보 게임 / The Tiny RPS Game

이 과정의 핵심은 가위바위보 게임입니다. TinyML에 대해 배울 수 있는 재미있고 접근하기 쉬운 방법이면서, PyTorch 워크플로우가 스케일업 못지않게 스케일다운도 쉽게 할 수 있다는 것을 보여줍니다. 최소한이지만 완전한 AI 워크플로우로 다음과 같은 과정을 포함합니다:
> The course's centerpiece is the Tiny RPS game. It's a fun and approachable way to learn about TinyML, while showing that PyTorch workflows can scale down just as easily as they scale up. It is a minimal but complete AI workflow which:

- 자체 데이터셋을 생성합니다.
- PyTorch에서 CNN을 학습합니다.
- ExecuTorch를 통해 내보냅니다.
- FVP에 배포하며, 물리적 하드웨어가 필요 없습니다.
- x86 Linux 호스트 머신이나 Ubuntu 22.04 이상을 실행하는 VM만 있으면 됩니다.

> - Generates its own dataset.
> - Trains a CNN in PyTorch.
> - Exports it via ExecuTorch.
> - Deploys it to the FVP, no need for physical hardware.
> - All you need is an x86 Linux host machine or VM running Ubuntu 22.04 or later.

### 파이프라인 / The Pipeline

**1. PyTorch에서 모델 학습 / Model Training in PyTorch**

"바위", "보", "가위"의 합성 이미지를 분류하기 위한 소형 CNN을 정의하고 학습합니다. 각 클래스는 데이터 변동을 시뮬레이션하기 위해 첫 글자("R", "P", "S")를 노이즈가 있는 28×28 그레이스케일 이미지로 렌더링합니다. (자세한 스크립트는 [학습 경로](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/training-inference-pytorch/)를 참고하세요)
> We define and train a compact CNN to classify synthetic images of "rock," "paper," and "scissors." Each class is rendered as a noisy 28×28 grayscale image of its first letter ("R", "P", or "S") to simulate data variation. ([See Learning Path](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/training-inference-pytorch/) for detailed script)

```python
import torch
import torch.nn as nn

class TinyRPS(nn.Module):
    """
    Simple ConvNet:
    [B,1,28,28] -> Conv3x3(16) -> ReLU -> Conv3x3(32) -> ReLU
      -> MaxPool2d(2) -> Conv3x3(64) -> ReLU -> MaxPool2d(2)
      -> flatten -> Linear(128) -> ReLU -> Linear(3)
    """

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x
```

이 아키텍처는 컴팩트하고 Ethos 친화적이어서 마이크로 엣지 배포에 이상적입니다. 학습에는 Adam 옵티마이저를 사용하며, 소규모 합성 데이터셋으로 몇 에폭만에 95% 이상의 검증 정확도를 달성합니다.
> This architecture is compact and Ethos-friendly, ideal for deployment to the micro-edge. Training uses Adam with a small synthetic dataset and achieves over 95% validation accuracy after a few epochs.

**2. ExecuTorch로 내보내기 / Exporting to ExecuTorch**

학습이 완료되면 모델을 ExecuTorch `.pte` 프로그램으로 내보냅니다. 이 형식은 소형 임베디드 런타임에서 Python 없이 실행할 수 있도록 최적화되어 있습니다. (자세한 스크립트는 [학습 경로](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/training-inference-pytorch/)를 참고하세요)
> Once trained, the model is exported to an ExecuTorch `.pte` program. This format is optimized for execution without Python on devices running tiny embedded runtimes. ([See Learning Path](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/training-inference-pytorch/) for detailed script)

```python
from executorch import exir
from torch.export import export

def export_to_pte(model: nn.Module, out_path: str, img_size: int) -> None:
    model.eval()
    example = torch.zeros(
        1, 1, img_size, img_size,
        dtype=torch.float32
    )

    # PyTorch의 exporter로 내보내기
    exported = export(model, (example,))
    edge = exir.to_edge(exported)
    prog = edge.to_executorch()

    with open(out_path, "wb") as f:
        f.write(prog.buffer)

    print(f"[export] wrote {out_path}")
```

이 단계에서는 PyTorch 연산 그래프를 마이크로컨트롤러에서 최소한의 오버헤드로 실행할 수 있는 정적이고 메모리 효율적인 그래프로 변환합니다.
> This step effectively converts your PyTorch computation graph into a static, memory-efficient graph that can run on microcontrollers with minimal overhead.

**3. Arm Corstone-320 FVP에 배포 / Deployment on Arm Corstone-320 FVP**

`.pte` 파일은 Cortex-M CPU와 Ethos-U microNPU가 결합된 소프트웨어 시뮬레이션인 Arm Corstone-320 FVP에 배포됩니다. 이를 통해 개발자는 실제 하드웨어에 플래싱하기 전에 로컬에서 모델을 실행하고 검증할 수 있습니다. 가위바위보 게임은 터미널에서 대화형으로 플레이할 수 있어, 실시간 온디바이스(on-device) 추론을 직접 체험할 수 있습니다.
> The `.pte` file is deployed on the Arm Corstone-320 FVP, a software simulation of a Cortex-M CPU paired with an Ethos-U microNPU. This allows developers to run and validate their model locally before flashing it to real hardware. The RPS game lets you play interactively in the terminal, demonstrating real-time on-device inference.

![FVP에서 실행되는 가위바위보 게임 / RPS game running on FVP](/assets/blog/2026-03-05-deploying-pytorch-models-to-the-micro-edge-with-executorch-and-arm/1.png){:style="width:100%"}

## 배운 교훈 / Lessons Learned

이 데모를 만들면서 PyTorch의 유연성이 데이터 센터에서 멈출 필요가 없다는 것을 알게 되었습니다. ExecuTorch를 사용하면 익숙한 PyTorch 워크플로우를 IoT 센서, 웨어러블, 임베디드 장치에 그대로 가져갈 수 있으며, 어디에서든 프라이버시를 보호하면서 저전력 AI를 구현할 수 있습니다.
> Working on this demo revealed that PyTorch's flexibility doesn't have to stop at the data center. ExecuTorch makes it possible to bring the same familiar PyTorch workflow to IoT sensors, wearables, and embedded devices, enabling privacy-preserving, low-power AI anywhere.

엣지 AI는 크기는 작지만, 잠재력은 무한합니다.
> Edge AI may be small in size, but it's huge in potential.

## 직접 해보기 / Try It Yourself

학습 경로: [Edge AI with PyTorch & ExecuTorch – Tiny RPS on Arm](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/training-inference-pytorch/)
대상: PyTorch 기본 경험이 있는 ML 개발자 및 임베디드 엔지니어
사전 요구사항: Introduction to TinyML on Arm
> Learning Path: [Edge AI with PyTorch & ExecuTorch – Tiny RPS on Arm](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/training-inference-pytorch/)
> Target Audience: ML developers and embedded engineers with basic PyTorch experience.
> Prerequisite: Introduction to TinyML on Arm

## 감사의 글 / Acknowledgements

이 학습 경로는 협업의 결실이며, 이 과정을 완성하는 데 도움을 준 팀에 특별한 감사를 드립니다. Annie Tallund, Zingo Andersen, George Gekov, Gemma Paris, Adrian Lundell, Madeline Underwood, Mary Bennion, Fredrik Knutsson의 소중한 기여에 감사드립니다.
> This learning path was a collaborative effort, and I owe a special thanks to the team that helped bring this course to life, including the valuable contributions of Annie Tallund, Zingo Andersen, George Gekov, Gemma Paris, Adrian Lundell, Madeline Underwood, Mary Bennion, and Fredrik Knutsson.

[저전력, 자원 제한 장치에서 Arm Ethos-U NPU를 사용한 엣지 AI 개발 관련 다른 도움말 살펴보기](https://developer.arm.com/edge-ai)
> [Explore other help for Edge AI development on low-power, resource-constrained devices using Arm Ethos-U NPUs](https://developer.arm.com/edge-ai)
