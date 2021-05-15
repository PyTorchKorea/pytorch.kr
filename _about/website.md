---
title: 파이토치 한국 사용자 모임 소개
order: 2
snippet: >
  ```python
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(backend='gloo')
    model = DistributedDataParallel(model)
  ```

---

파이토치 한국 사용자 모임은 2018년 중순, 학습을 목적으로 PyTorch 튜토리얼 문서를 한국어로 번역하면서 시작하였습니다.<br />

Facebook, Inc와 관련없는 개인들이 시작한 비공식 / 비영리 사이트로, 한국어를 사용하시는 많은 분들께 PyTorch를 소개하고 함께 배우며 성장하는 것을 목표로 하고 있습니다.<br />

이 홈페이지 또는 튜토리얼에 개선이 필요한 부분을 발견하셨다면 <a href="https://github.com/9bow/PyTorchKR" target="_blank">홈페이지 저장소</a> 또는 <a href="https://github.com/9bow/PyTorch-tutorials-kr" target="_blank">튜토리얼 저장소</a>에 이슈 또는 PR을 남겨주세요.
