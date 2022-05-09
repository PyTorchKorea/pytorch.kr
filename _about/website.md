---
title: 파이토치 한국 사용자 모임 소개
order: 2
snippet: >
  ```python
    import torch

    class MyModule(torch.nn.Module):
      def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

      def forward(self, input):
        if input.sum() > 0:
          output = self.weight.mv(input)
        else:
          output = self.weight + input
        return output

      my_script_module = torch.jit.script(MyModule(3, 4))
      my_script_module.save("my_script_module.pt")
  ```

---

파이토치 한국 사용자 모임은 2018년 중순 학습 목적으로 PyTorch 튜토리얼 문서를 한국어로 번역하면서 시작하였습니다. <br />

PyTorch를 학습하고 사용하는 한국 사용자들이 시작한 사용자 커뮤니티로, 한국어를 사용하시는 많은 분들께 PyTorch를 소개하고 함께 배우며 성장하는 것을 목표로 합니다.<br />

PyTorch를 사용하며 얻은 유용한 정보를 공유하고 싶으시거나 다른 사용자와 소통하고 싶으시다면 <a href="{{ site.external_urls.site_community }}">커뮤니티 공간</a>에 방문해주세요!
