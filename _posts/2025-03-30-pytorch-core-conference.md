---
layout: blog_detail
title: "PyTorch Core Maintainer와 함께한 기술 교류의 장 - 파이토치 한국 사용자 모임 리뷰"
author: 김지호
---

지난 3월 말, 파이토치 한국 사용자 모임은 특별한 연사분들을 모시고 파이토치 코어와 생태계 전반에 대한 깊이 있는 이야기를 나누는 시간을 가졌습니다. 행사 규모가 이전보다 두 배 이상 커진 만큼, 더 많은 분들과 함께 기술을 공유하고 소통할 수 있었습니다. 멋진 행사 장소를 후원해주신 [구름](https://goorm.co/)에도 이 자리를 빌려 감사 인사를 전합니다.😄

![컨퍼런스 종료 후 단체 사진](/assets/images/pytorch_core_conference/total.png)

이번 후기는 컨퍼런스에 직접 참석하지 못하신 분들께도 의미 있는 내용을 전하고, 현장을 함께했던 분들께는 그날의 열기와 인사이트를 다시 떠올릴 수 있도록 정리한 글입니다. 컨퍼런스에서는 PyTorch Core, AI 가속기, 추론 최적화, 대규모 언어 모델 개발까지 다양한 분야의 전문가들이 함께했습니다. 그 중심이 되었던 세션들을 아래에 간략히 소개합니다.

## 1️⃣ 이제응 | PyTorch Foundation

리눅스 재단 소속의 파이토치 재단은 오픈소스 생태계의 핵심 기술 발전을 이끌고 있습니다. 파이토치의 성장 배경과 함께, 전 세계에서 진행 중인 수많은 프로젝트, 그리고 연 20% 이상의 생태계 성장률 등을 통해 파이토치의 탄탄한 기반을 확인할 수 있었습니다. 재단의 운영 방식, 멤버사 참여, 향후 행사 계획 등 실무자들이 궁금해할 정보도 함께 다뤄졌습니다.

![Session 1: 이제응 - PyTorch Foundation](/assets/images/pytorch_core_conference/session_1.png)

## 2️⃣ Alban Desmaison | PyTorch Roadmap

PyTorch의 디자인 철학 그리고 Meta의 파이토치 컨트리뷰션 로드맵([링크](https://dev-discuss.pytorch.org/t/meta-pytorch-team-2025-h1-roadmaps/2794))을 공유합니다. PyTorch가 제공하는 Eager 모드와 Compiled 모드를 비교하면서, 특히 디바이스 Eager 백엔드를 구성하는 핵심 요소에 대해 자세히 설명하였습니다. 기술적인 내용을 깊이 있게 다루었을 뿐 아니라, 메모리 프로파일러, 향상된 커스텀 연산 지원, Pinned Memory 개선 등 실용적인 기능들도 함께 소개되었습니다.

![Session 2: Alban Desmaison - PyTorch Roadmap](/assets/images/pytorch_core_conference/session_2.png)

## 3️⃣ 김홍석 | PyTorch on Rebellions AI Accelerators: Status

Rebellions은 PyTorch 2.0의 구조적 진화에 발맞춰 자체 NPU 아키텍처에 최적화된 런타임 통합을 진행 중입니다. 이번 세션에서는 개발 중인 칩의 성능과 확장성, 그리고 PyTorch 런타임과의 통합을 위한 아키텍처 설계 및 Eager Mode 지원 관련 기술적 고민들을 공유했습니다. 금년도 릴리즈를 목표로 한 개발 계획도 함께 소개되었습니다.

![Session 3: 김홍석 - PyTorch on Rebellions AI Accelerators: Status](/assets/images/pytorch_core_conference/session_3.png)

## 4️⃣ 조규진 | Backend.AI: 모든 AI 가속기를 위한 통합 플랫폼

Backend.AI는 다양한 종류의 AI 가속기를 추상화하고, 통합된 워크플로우를 제공하는 플랫폼입니다. AI 가속기 아키텍처가 다양해지면서, 높은 이식성과 인프라 통합의 중요성도 함께 커지고 있습니다. 이번 세션에서는 NPU 스케줄링, 리소스 할당, 모니터링 등 AI 개발과 운영 전반을 아우르는 기능들이 소개되었으며, 현재는 NVIDIA, Intel, Tenstorrent, Rebellions 등 다양한 AI 가속기를 지원하고 있음을 확인할 수 있었습니다.

![Session 4: 조규진 - Backend.AI: 모든 AI 가속기를 위한 통합 플랫폼](/assets/images/pytorch_core_conference/session_4.png)

## 5️⃣ 김태호 | NetsPresso를 이용한 다양한 모델을 다양한 칩셋에 최적화 및 배포하는 방법

이 세션에서는 AI 모델을 산업에 적용하는 과정에서 추론 단계의 중요성과 현실적인 어려움을 짚었습니다. 다양한 최신 SOTA 모델들이 빠르게 등장하는 가운데, 원클릭으로 디바이스별 모델 실행 가능 여부를 확인할 수 있는 환경이 이상적인 방향으로 제시되었습니다. Netspresso는 PyTorch와 호환이 가능한 정적 그래프 표현 방안을 고민하고 있으며, 모델 개발, 최적화, 테스트 전반을 효율적으로 지원하고 있습니다.

![Session 5: 김태호 - NetsPresso를 이용한 다양한 모델을 다양한 칩셋에 최적화 및 배포하는 방법](/assets/images/pytorch_core_conference/session_5.png)

## 6️⃣ 이정엽 | The Journey to Reproduce Deepseek-R1

대규모 언어 모델 Deepseek을 직접 재현하는 과정에서 총 201번의 실험을 거치며 얻은 시행착오와 인사이트를 생생히 공유한 세션이었습니다. 한국어 기반 학습 과정에서 발생한 문제, tokenizer 수정, 미세 조정 전략 등 실질적인 경험이 중심이었고, 이를 기반으로 한 개선 방향과 다음 단계까지 이어지는 흐름이 인상적이었습니다.

![Session 6: 이정엽 - The Journey to Reproduce Deepseek-R1](/assets/images/pytorch_core_conference/session_6.png)

## 7️⃣ 김솔 | A journey from TCP architecture to production-level LLMs

Tensor Contraction을 하드웨어 수준에서 지원하는 TCP 아키텍처를 중심으로, 대규모 모델을 프로덕션 수준에서 운용하기 위한 통합 최적화 전략이 소개되었습니다. 하드웨어 추상화 계층(HAL)을 기반으로 한 실행 최적화, PyTorch 생태계와의 bottom-up 방식 통합 작업 등, 소프트웨어와 하드웨어를 함께 고려한 복합적인 접근이 돋보였습니다.

![Session 7: 김솔 - A journey from TCP architecture to production-level LLMs](/assets/images/pytorch_core_conference/session_7.png)

## 💡 Q&A 세션 💡

발표 세션 종료 후 이어진 패널 토크에서는 참석자들의 적극적인 질문과 연사들의 통찰력 있는 답변이 어우러지며, 마지막까지 깊이 있는 논의가 이어졌습니다. 국내 개발자들의 파이토치에 대한 뜨거운 관심과 커뮤니티의 에너지를 더욱 생생하게 느낄 수 있었습니다.

![패널 토크 및 네트워킹의 순간들](/assets/images/pytorch_core_conference/qna.png)

## 마치며

파이토치 한국 사용자 모임은 2022년 10월 첫 오프라인 컨퍼런스를 시작으로, 벌써 다섯 번째 기술 컨퍼런스를 운영하고 있습니다. 매번 행사를 거듭할수록 파이토치 생태계의 넓이와 깊이를 새롭게 실감하고 있습니다. 사용자, 기여자, 생태계 관리자 등 다양한 시각에서 펼쳐지는 이야기는 계속해서 확장되고 있으며, 앞으로도 이를 함께 나누는 자리를 꾸준히 이어갈 계획입니다.

다음 컨퍼런스에서도 다양한 주제의 발표로 찾아뵙겠습니다. 그때 또 만나요! 🙌