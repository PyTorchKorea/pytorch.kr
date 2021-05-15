---
title: Distributed Training
order: 3
snippet: >
  ```python
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(backend='gloo')
    model = DistributedDataParallel(model)
  ```

summary-home: torch.distributed 백엔드로 연구 및 상용에서 확장 가능한 분산 학습 및 성능 최적화할 수 있습니다.
featured-home: true

---

Optimize performance in both research and production by taking advantage of native support for asynchronous execution of collective operations and peer-to-peer communication that is accessible from Python and C++.
