---
layout: get_started
title: 이전 버전의 PyTorch
permalink: /get-started/previous-versions/
background-class: get-started-background
body-class: get-started
order: 4
published: true
redirect_from: /previous-versions.html
---

## 이전 버전의 PyTorch 설치하기

[최신 버전](/get-started/locally)을 설치하시기를 권해드리지만,
편의를 위해 아래와 같이 이전 버전의 설치 파일과 방법을 제공하고 있습니다.

## 1.0.0 이상 버전 설치하기

### v2.6.0

#### Wheel

##### OSX

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

##### Linux and Windows

```
# ROCM 6.1 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.1
# ROCM 6.2.4 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# CPU only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

### v2.5.1

#### Conda

##### OSX

```
# conda
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

##### Linux and Windows

```
# ROCM 6.1 (Linux only)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.1
# ROCM 6.2 (Linux only)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2
# CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

### v2.5.0

#### Conda

##### OSX

```
# conda
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
```

##### Linux and Windows

```
# ROCM 6.1 (Linux only)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.1
# ROCM 6.2 (Linux only)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2
# CUDA 11.8
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
```

### v2.4.1
#### Conda
##### OSX
```
# conda
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 -c pytorch
```
#####  Linux and Windows
```
# CUDA 11.8
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly -c pytorch
```
#### Wheel
##### OSX
```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
```
##### Linux and Windows
```
# ROCM 6.1 (Linux only)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/rocm6.1
# CUDA 11.8
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

### v2.4.0

#### Conda

##### OSX

```
# conda
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

##### Linux and Windows

```
# ROCM 6.1 (Linux only)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1
# CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# CUDA 12.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### v2.3.1

#### Conda

##### OSX

```
# conda
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

##### Linux and Windows

```
# ROCM 6.0 (Linux only)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0
# CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

### v2.3.0

#### Conda

##### OSX

```
# conda
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0
```

##### Linux and Windows

```
# ROCM 6.0 (Linux only)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
# CUDA 11.8
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
```

### v2.2.2

#### Conda

##### OSX

```
# conda
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
```

##### Linux and Windows

```
# ROCM 5.7 (Linux only)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7
# CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

### v2.2.1

#### Conda

##### OSX

```
# conda
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
```

##### Linux and Windows

```
# ROCM 5.7 (Linux only)
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/rocm5.7
# CUDA 11.8
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
```

### v2.2.0

#### Conda

##### OSX

```
# conda
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
```

##### Linux and Windows

```
# ROCM 5.6 (Linux only)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
# CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
```

### v2.1.2

#### Conda

##### OSX

```
# conda
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```

##### Linux and Windows

```
# ROCM 5.6 (Linux only)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/rocm5.6
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

### v2.1.1

#### Conda

##### OSX

```
# conda
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
```

##### Linux and Windows

```
# ROCM 5.6 (Linux only)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/rocm5.6
# CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
```

### v2.1.0

#### Conda

##### OSX

```
# conda
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 12.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

##### Linux and Windows

```
# ROCM 5.6 (Linux only)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/rocm5.6
# CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### v2.0.1

#### Conda

##### OSX

```
# conda
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.7
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
# CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

##### Linux and Windows

```
# ROCM 5.4.2 (Linux only)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/rocm5.4.2
# CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# CPU only
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### v2.0.0

#### Conda

##### OSX

```
# conda
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.7
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# CUDA 11.8
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

##### Linux and Windows

```
# ROCM 5.4.2 (Linux only)
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/rocm5.4.2
# CUDA 11.7
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
# CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# CPU only
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
```

### v1.13.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

##### Linux and Windows

```
# ROCM 5.2 (Linux only)
pip3 install torch torchvision torchaudio --extra-index-url
pip install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# CPU only
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

### v1.13.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# CPU Only
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
```

##### Linux and Windows

```
# ROCM 5.2 (Linux only)
pip3 install torch torchvision torchaudio --extra-index-url
pip install torch==1.13.0+rocm5.2 torchvision==0.14.0+rocm5.2 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/rocm5.2
# CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# CPU only
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

### v1.12.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
```

##### Linux and Windows

```
# ROCM 5.1.1 (Linux only)
pip install torch==1.12.1+rocm5.1.1 torchvision==0.13.1+rocm5.1.1 torchaudio==0.12.1 --extra-index-url  https://download.pytorch.org/whl/rocm5.1.1
# CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# CUDA 10.2
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
# CPU only
pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

### v1.12.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0
```

##### Linux and Windows

```
# ROCM 5.1.1 (Linux only)
pip install torch==1.12.0+rocm5.1.1 torchvision==0.13.0+rocm5.1.1 torchaudio==0.12.0 --extra-index-url  https://download.pytorch.org/whl/rocm5.1.1
# CUDA 11.6
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
# CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
# CUDA 10.2
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
# CPU only
pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

### v1.11.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# CPU Only
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
```

##### Linux and Windows

```
# ROCM 4.5.2 (Linux only)
pip install torch==1.11.0+rocm4.5.2 torchvision==0.12.0+rocm4.5.2 torchaudio==0.11.0 --extra-index-url  https://download.pytorch.org/whl/rocm4.5.2

# CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 10.2
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102

# CPU only
pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

### v1.10.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
```

##### Linux and Windows

```
# ROCM 4.2 (Linux only)
pip install torch==1.10.1+rocm4.2 torchvision==0.11.2+rocm4.2 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.1 (Linux only)
pip install torch==1.10.1+rocm4.1 torchvision==0.11.2+rocm4.1 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.0.1 (Linux only)
pip install torch==1.10.1+rocm4.0.1 torchvision==0.10.2+rocm4.0.1 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```


### v1.10.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
```

##### Linux and Windows

```
# ROCM 4.2 (Linux only)
pip install torch==1.10.0+rocm4.2 torchvision==0.11.0+rocm4.2 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.1 (Linux only)
pip install torch==1.10.0+rocm4.1 torchvision==0.11.0+rocm4.1 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.0.1 (Linux only)
pip install torch==1.10.0+rocm4.0.1 torchvision==0.10.1+rocm4.0.1 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.1
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```


### v1.9.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.9.0 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1
```

##### Linux and Windows

```
# ROCM 4.2 (Linux only)
pip install torch==1.9.0+rocm4.2 torchvision==0.10.1+rocm4.2 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.1 (Linux only)
pip install torch==1.9.0+rocm4.1 torchvision==0.10.1+rocm4.1 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.0.1 (Linux only)
pip install torch==1.9.1+rocm4.0.1 torchvision==0.10.1+rocm4.0.1 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.9.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
```

##### Linux and Windows

```
# ROCM 4.2 (Linux only)
pip install torch==1.9.0+rocm4.2 torchvision==0.10.0+rocm4.2 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.1 (Linux only)
pip install torch==1.9.0+rocm4.1 torchvision==0.10.0+rocm4.1 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 4.0.1 (Linux only)
pip install torch==1.9.0+rocm4.0.1 torchvision==0.10.0+rocm4.0.1 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.8.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
```

#####  Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
```

##### Linux and Windows

```
# ROCM 4.0.1 (Linux only)
pip install torch==1.8.1+rocm4.0.1 torchvision==0.9.1+rocm4.0.1 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# ROCM 3.10 (Linux only)
pip install torch==1.8.1+rocm3.10 torchvision==0.9.1+rocm3.10 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.1
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```


### v1.8.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch
```

##### Linux and Windows

```
# CUDA 10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.1
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# CPU Only
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```

##### Linux and Windows

```
# RocM 4.0.1 (Linux only)
pip install torch -f https://download.pytorch.org/whl/rocm4.0.1/torch_stable.html
pip install ninja
pip install 'git+https://github.com/pytorch/vision.git@v0.9.0'

# CUDA 11.1
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0

# CPU only
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.7.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.2
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=9.2 -c pytorch

# CUDA 10.1
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# CUDA 10.2
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch

# CUDA 11.0
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# CPU Only
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2
```

##### Linux and Windows

```
# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.7.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.2
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=9.2 -c pytorch

# CUDA 10.1
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.1 -c pytorch

# CUDA 10.2
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.0
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# CPU Only
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0
```

##### Linux and Windows

```
# CUDA 11.0
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0

# CUDA 10.1
pip install torch==1.7.0+cu101 torchvision==0.8.0+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.0+cu92 torchvision==0.8.0+cu92 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.0+cpu torchvision==0.8.0+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.6.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.2
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=9.2 -c pytorch

# CUDA 10.1
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch

# CUDA 10.2
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

# CPU Only
conda install pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.6.0 torchvision==0.7.0
```

##### Linux and Windows

```
# CUDA 10.2
pip install torch==1.6.0 torchvision==0.7.0

# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.5.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.5.1 torchvision==0.6.1 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.2
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=9.2 -c pytorch

# CUDA 10.1
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch

# CUDA 10.2
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch

# CPU Only
conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.5.1 torchvision==0.6.1
```

##### Linux and Windows

```
# CUDA 10.2
pip install torch==1.5.1 torchvision==0.6.1

# CUDA 10.1
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.5.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.5.0 torchvision==0.6.0 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.2
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=9.2 -c pytorch

# CUDA 10.1
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch

# CUDA 10.2
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch

# CPU Only
conda install pytorch==1.5.0 torchvision==0.6.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.5.0 torchvision==0.6.0
```

##### Linux and Windows

```
# CUDA 10.2
pip install torch==1.5.0 torchvision==0.6.0

# CUDA 10.1
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.4.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.4.0 torchvision==0.5.0 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.2
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=9.2 -c pytorch

# CUDA 10.1
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

# CPU Only
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.4.0 torchvision==0.5.0
```

##### Linux and Windows

```
# CUDA 10.1
pip install torch==1.4.0 torchvision==0.5.0

# CUDA 9.2
pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.2.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.2.0 torchvision==0.4.0 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch

# CUDA 10.0
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# CPU Only
conda install pytorch==1.2.0 torchvision==0.4.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.2.0 torchvision==0.4.0
```

##### Linux and Windows

```
# CUDA 10.0
pip install torch==1.2.0 torchvision==0.4.0

# CUDA 9.2
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### v1.1.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch

# CUDA 10.0
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# CPU Only
conda install pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.1.0 torchvision==0.3.0
```

##### Linux and Windows

```
# CUDA 10.0
Download and install wheel from https://download.pytorch.org/whl/cu100/torch_stable.html

# CUDA 9.0
Download and install wheel from https://download.pytorch.org/whl/cu90/torch_stable.html

# CPU only
Download and install wheel from https://download.pytorch.org/whl/cpu/torch_stable.html
```

### v1.0.1

#### Conda

##### OSX

```
# conda
conda install pytorch==1.0.1 torchvision==0.2.2 -c pytorch
```

##### Linux and Windows

```
# CUDA 9.0
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch

# CUDA 10.0
conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch

# CPU Only
conda install pytorch-cpu==1.0.1 torchvision-cpu==0.2.2 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.0.1 torchvision==0.2.2
```

##### Linux and Windows

```
# CUDA 10.0
Download and install wheel from https://download.pytorch.org/whl/cu100/torch_stable.html

# CUDA 9.0
Download and install wheel from https://download.pytorch.org/whl/cu90/torch_stable.html

# CPU only
Download and install wheel from https://download.pytorch.org/whl/cpu/torch_stable.html
```

### v1.0.0

#### Conda

##### OSX

```
# conda
conda install pytorch==1.0.0 torchvision==0.2.1 -c pytorch
```

##### Linux and Windows

```
# CUDA 10.0
conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch

# CUDA 9.0
conda install pytorch==1.0.0 torchvision==0.2.1 cuda90 -c pytorch

# CUDA 8.0
conda install pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch

# CPU Only
conda install pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 cpuonly -c pytorch
```

#### Wheel

##### OSX

```
pip install torch==1.0.0 torchvision==0.2.1
```

##### Linux and Windows

```
# CUDA 10.0
Download and install wheel from https://download.pytorch.org/whl/cu100/torch_stable.html

# CUDA 9.0
Download and install wheel from https://download.pytorch.org/whl/cu90/torch_stable.html

# CUDA 8.0
Download and install wheel from https://download.pytorch.org/whl/cu80/torch_stable.html

# CPU only
Download and install wheel from https://download.pytorch.org/whl/cpu/torch_stable.html
```

## 1.0.0 미만 버전 설치하기

### Via conda

> This should be used for most previous macOS version installs.

To install a previous version of PyTorch via Anaconda or Miniconda,
replace "0.4.1" in the following commands with the desired version
(i.e., "0.2.0").

Installing with CUDA 9

`conda install pytorch=0.4.1 cuda90 -c pytorch`

or

`conda install pytorch=0.4.1 cuda92 -c pytorch`

Installing with CUDA 8

`conda install pytorch=0.4.1 cuda80 -c pytorch`

Installing with CUDA 7.5

`conda install pytorch=0.4.1 cuda75 -c pytorch`

Installing without CUDA

`conda install pytorch=0.4.1 -c pytorch`

### From source

It is possible to checkout an older version of [PyTorch](https://github.com/pytorch/pytorch)
and build it.
You can list tags in PyTorch git repository with `git tag` and checkout a
particular one (replace '0.1.9' with the desired version) with

`git checkout v0.1.9`

Follow the install from source instructions in the README.md of the PyTorch
checkout.

### Via pip

Download the `whl` file with the desired version from the following html pages:

- <https://download.pytorch.org/whl/cpu/torch_stable.html> # CPU-only build
- <https://download.pytorch.org/whl/cu80/torch_stable.html> # CUDA 8.0 build
- <https://download.pytorch.org/whl/cu90/torch_stable.html> # CUDA 9.0 build
- <https://download.pytorch.org/whl/cu92/torch_stable.html> # CUDA 9.2 build
- <https://download.pytorch.org/whl/cu100/torch_stable.html> # CUDA 10.0 build

Then, install the file with `pip install [downloaded file]`


Note: most pytorch versions are available only for specific CUDA versions. For example pytorch=1.0.1 is not available for CUDA 9.2

### (Old) PyTorch Linux binaries compiled with CUDA 7.5

These predate the html page above and have to be manually installed by downloading the wheel file and `pip install downloaded_file`

- [cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl)
- [cu75/torch-0.3.0.post4-cp27-cp27m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp27-cp27m-linux_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post3-cp27-cp27m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp27-cp27m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp36-cp36m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp36-cp36m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp35-cp35m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp35-cp35m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post2-cp27-cp27m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post2-cp27-cp27m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp36-cp36m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp36-cp36m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp35-cp35m-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp27-cp27mu-manylinux1_x86_64.whl)
- [cu75/torch-0.2.0.post1-cp27-cp27m-manylinux1_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.2.0.post1-cp27-cp27m-manylinux1_x86_64.whl)
- [cu75/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.12.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.12.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.12.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post5-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post5-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.11.post4-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post4-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post4-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post4-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.11.post4-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.11.post4-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.10.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.10.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.10.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.10.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.9.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.9.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.9.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.9.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.8.post1-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.8.post1-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.8.post1-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.8.post1-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.8.post1-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.8.post1-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.7.post2-cp36-cp36m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.7.post2-cp36-cp36m-linux_x86_64.whl)
- [cu75/torch-0.1.7.post2-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.7.post2-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.7.post2-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.7.post2-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.6.post22-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post22-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.6.post22-cp27-none-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post22-cp27-none-linux_x86_64.whl)
- [cu75/torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl)
- [cu75/torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/cu75/torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl)

### Windows binaries

- [cpu/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cu80/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cu90/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cu100/torch-1.0.0-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu100/torch-1.0.0-cp35-cp35m-win_amd64.whl)
- [cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cu80/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl)
- [cpu/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cu80/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cu90/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu100/torch-1.0.0-cp37-cp37m-win_amd64.whl)
- [cpu/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cu80/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl](https://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-win_amd64.whl)
- [cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cu80/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl](https://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-win_amd64.whl)
- [cpu/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cpu/torch-0.4.1-cp37-cp37m-win_amd64.whl)
- [cu80/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu80/torch-0.4.1-cp37-cp37m-win_amd64.whl)
- [cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu90/torch-0.4.1-cp37-cp37m-win_amd64.whl)
- [cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl](https://download.pytorch.org/whl/cu92/torch-0.4.1-cp37-cp37m-win_amd64.whl)

### Mac and misc. binaries

For recent macOS binaries, use `conda`:

e.g.,

`conda install pytorch=0.4.1 cuda90 -c pytorch`
`conda install pytorch=0.4.1 cuda92 -c pytorch`
`conda install pytorch=0.4.1 cuda80 -c pytorch`
`conda install pytorch=0.4.1 -c pytorch` # No CUDA

- [torchvision-0.1.6-py3-none-any.whl](https://download.pytorch.org/whl/torchvision-0.1.6-py3-none-any.whl)
- [torchvision-0.1.6-py2-none-any.whl](https://download.pytorch.org/whl/torchvision-0.1.6-py2-none-any.whl)
- [torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl)
- [torch-1.0.0-cp36-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-none-macosx_10_7_x86_64.whl)
- [torch-1.0.0-cp35-none-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp35-none-macosx_10_6_x86_64.whl)
- [torch-1.0.0-cp27-none-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/cpu/torch-1.0.0-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.4.0-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.4.0-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.4.0-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.4.0-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.4.0-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.4.0-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.3.1-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.3.1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.1-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.3.0.post4-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.0.post4-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.3.0.post4-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.0.post4-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.3.0.post4-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.3.0.post4-cp27-none-macosx_10_6_x86_64.whl)
- [torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post3-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post3-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post3-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post3-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post2-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post2-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post1-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post1-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.2.0.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.2.0.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post2-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post2-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post1-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post1-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.12.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.12.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post5-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post5-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post5-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post5-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post5-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post5-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post4-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post4-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post4-cp35-cp35m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post4-cp35-cp35m-macosx_10_7_x86_64.whl)
- [torch-0.1.11.post4-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.11.post4-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.10.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.10.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.10.post1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.10.post1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.10.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.10.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post2-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post2-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.9.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.9.post1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.9.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.9.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.8.post1-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.8.post1-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.8.post1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.8.post1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.8.post1-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.8.post1-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.7.post2-cp36-cp36m-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.7.post2-cp36-cp36m-macosx_10_7_x86_64.whl)
- [torch-0.1.7.post2-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.7.post2-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.7.post2-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.7.post2-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.6.post22-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post22-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1.6.post22-cp27-none-macosx_10_7_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post22-cp27-none-macosx_10_7_x86_64.whl)
- [torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post20-cp35-cp35m-linux_x86_64.whl)
- [torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl)
- [torch-0.1.6.post17-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post17-cp35-cp35m-linux_x86_64.whl)
- [torch-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl)
- [torch-0.1-cp35-cp35m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1-cp35-cp35m-macosx_10_6_x86_64.whl)
- [torch-0.1-cp27-cp27m-macosx_10_6_x86_64.whl](https://download.pytorch.org/whl/torch-0.1-cp27-cp27m-macosx_10_6_x86_64.whl)
- [torch_cuda80-0.1.6.post20-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post20-cp35-cp35m-linux_x86_64.whl)
- [torch_cuda80-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post20-cp27-cp27mu-linux_x86_64.whl)
- [torch_cuda80-0.1.6.post17-cp35-cp35m-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post17-cp35-cp35m-linux_x86_64.whl)
- [torch_cuda80-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl](https://download.pytorch.org/whl/torch_cuda80-0.1.6.post17-cp27-cp27mu-linux_x86_64.whl)

<script page-id="previous-versions" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>
<script src="{{ site.baseurl }}/assets/get-started-sidebar.js"></script>
