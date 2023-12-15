# macOS에서 설치하기
{:.no_toc}

PyTorch 는 macOS 에서 설치 및 사용할 수 있습니다. PyTorch를 설치할 시스템과 사용할 수 있는 GPU 에 따라, Mac에서의 처리 속도 측면에서의 PyTorch 사용 경험은 사람마다 다를 수 있습니다.

## 요구 사항
{: #mac-prerequisites}

### macOS 버전

PyTorch는 macOS 10.15 (Catalina) 이후 macOS에서 설치할 수 있습니다.

### Python
{: #mac-python}

Python 3.8 ~ 3.11 사이의 버전을 사용하기를 권장합니다. 해당 버전은 아나콘다 패키지 관리자 (아래 [참조](#아나콘다)), [HomeBrew](https://brew.sh), [Python 웹사이트](https://www.python.org/downloads/mac-osx/) 에서 설치할 수 있습니다.

### 패키지 관리자
{: #mac-package-manager}

PyTorch 바이너리는 [아나콘다](https://www.anaconda.com/download/#macos) 또는 [pip](https://pypi.org/project/pip/) 패키지 관리자를 통해 설치할 수 있습니다.
Python과 PyTorch 설치 환경을 쉽게 격리할 수 있는 아나콘다를 권장하고 있습니다.

#### 아나콘다

아나콘다는 [그래픽 설치 인스톨러](https://www.anaconda.com/download/#macos) 또는 명령줄 인스톨러를 사용할 수 있습니다.
명령줄 인스톨러를 사용하는 경우, installer link를 복사하여 붙여넣거나 인텔 맥에서는 아래와 같이 실행할 수 있습니다.

```bash
# 설치 시기에 따라 아나콘다 버전은 다를 수 있습니다.
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
# 프롬프트가 나오면 옵션을 선택합니다, 일반적으로 기본값을 사용합니다.
```
혹은 m1 맥의 경우 아래와 같이 실행할 수 있습니다.
```bash
# 설치 시기에 따라 아나콘다 버전은 다를 수 있습니다.
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
# 프롬프트가 나오면 옵션을 선택합니다, 일반적으로 기본값을 사용합니다.
```
#### pip

*Python 3*

Homebrew나 Python 웹사이트에서 Python을 설치했다면, `pip` 도 같이 설치됩니다.
Python 3.x 를 설치했다면, `pip3` 를 사용합니다.

> 팁: 심볼릭 링크를 통해 `pip3` 대신 `pip`을 사용할 수도 있습니다.

## 설치
{: #mac-installation}

### 아나콘다
{: #mac-anaconda}

아나콘다를 사용할 경우, 아래와 같이 PyTorch를 설치할 수 있습니다.

```bash
conda install pytorch torchvision -c pytorch
```

### pip
{: #mac-anaconda}

pip을 사용할 경우, 아래와 같이 설치할 수 있습니다.

```bash
# Python 3.x
pip3 install torch torchvision
```

## 검증
{: #mac-verification}

PyTorch가 제대로 설치되었는지 확인하기 위해, 아래와 같은 샘플 코드를 실행해 볼 수 있습니다.
아래는 무작위로 초기화된 tensor를 생성해보는 샘플 코드입니다.

```python
import torch
x = torch.rand(5, 3)
print(x)
```

출력 결과는 아래와 비슷한 형태여야 합니다.

```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

## 소스에서 빌드
{: #mac-from-source}

대부분의 PyTorch 사용자들은, 패키지 관리자를 통해 사전에 빌드된 바이너리를 사용하는 것이 제일 좋습니다. 정식으로 릴리즈 되지 않은 최신 PyTorch 코드를 사용하려고 하거나, PyTorch core에 대한 테스트나 개발을 하는 경우에는 직접 PyTorch를 빌드해야합니다.

### 요구사항
{: #mac-prerequisites-2}

1. [선택사항] [아나콘다](#아나콘다) 설치
2. [https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source) 참조하여 빌드 (영문)

[위 섹션](#mac-verification)을 참조하여 잘 설치되었는지 검증 할 수 있습니다.