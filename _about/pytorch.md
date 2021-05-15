---
title: 파이토치(PyTorch) 소개
order: 1
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

PyTorch(파이토치)는 연구용 프로토타입부터 상용 제품까지 빠르게 만들 수 있는 오픈 소스 머신러닝 프레임워크입니다.<br /><br />

PyTorch는 사용자 친화적인 프론트엔드(front-end)와 분산 학습, 다양한 도구와 라이브러리를 통해 빠르고 유연한 실험 및 효과적인 상용화를 가능하게 합니다. <br /><br />

PyTorch에 대한 더 자세한 소개는 <a href="https://pytorch.org/" target="_blank">공식 홈페이지(영어)</a> 및 <a href="https://github.com/pytorch/pytorch" target="_blank">공식 저장소(영어)</a>에서 확인하실 수 있습니다.
