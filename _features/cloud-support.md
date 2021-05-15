---
title: Cloud Support
order: 8
snippet: >
  ```sh
    export IMAGE_FAMILY="pytorch-latest-cpu"
    export ZONE="us-west1-b"
    export INSTANCE_NAME="my-instance"

    gcloud compute instances create $INSTANCE_NAME \
      --zone=$ZONE \
      --image-family=$IMAGE_FAMILY \
      --image-project=deeplearning-platform-release
  ```

summary-home: PyTorch는 주요 클라우드 플랫폼에서 쉽게 개발하고 간편하게 확장(scaling)할 수 있습니다.
featured-home: true

---

PyTorch is well supported on major cloud platforms, providing frictionless development and easy scaling through prebuilt images, large scale training on GPUs, ability to run models in a production scale environment, and more.
