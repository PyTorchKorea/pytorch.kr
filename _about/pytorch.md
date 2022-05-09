---
title: 파이토치(PyTorch) 소개
order: 1
snippet: >
  ```python
    # import torch
    import torch

    # load model
    model = torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)

    #inference
    img = torch.randn(1, 3, 640, 384)
    features, regression, classification, anchors, segmentation = model(img)
  ```

---

PyTorch(파이토치)는 FAIR(Facebook AI Research)에서 만든 연구용 프로토타입부터 상용 제품까지 빠르게 만들 수 있는 오픈 소스 머신러닝 프레임워크입니다.<br />

PyTorch는 사용자 친화적인 프론트엔드(front-end)와 분산 학습, 다양한 도구와 라이브러리를 통해 빠르고 유연한 실험 및 효과적인 상용화를 가능하게 합니다. <br />

PyTorch에 대한 더 자세한 소개는 <a href="{{ site.external_urls.org_www }}">공식 홈페이지</a>에서 확인하실 수 있습니다.
