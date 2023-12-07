# Windows에 설치하기
{:.no_toc}
  
PyTorch는 여러 버전의 Windows에 설치하고 사용할 수 있습니다. 시스템과 컴퓨터의 사양에 의해 Windows에서 PyTorch를 사용하는 경험이 처리 시간에 따라 갈릴 수 있습니다. 필수는 아니지만, PyTorch의 [CUDA](https://developer.nvidia.com/cuda-zone) [지원](https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html?highlight=cuda#cuda-tensors)의 모든 성능을 이용하기 위해서는 NVIDIA GPU를 사용하는 Windows 시스템을 사용하는것을 권장합니다.

## 선수사항
{: #windows-prerequisites}

### 지원 Windows 버전

파이토치는 다음과 같은 Windows 버전을 지원합니다.

* [Windows](https://www.microsoft.com/en-us/windows) 7 그리고 그 이상; [Windows 10](https://www.microsoft.com/en-us/software-download/windows10ISO) 또는 그 이상을 권장합니다.
* [Windows Server 2008](https://docs.microsoft.com/en-us/windows-server/windows-server) r2 그리고 그 이상

> 여기 나와있는 설치 방법은 대부분의 Windows 버전에 적용됩니다. 표시된 특정 예시는 Windows 10 Enterprise 장치에서 실행됩니다.

### Python
{: #windows-python}

현재 Windows에서 동작하는 PyTorch는 오직 Python 3.7-3.9 버전을 지원합니다. Python 2.x 버전은 지원하지 않습니다.

Windows에 기본적으로 설치되어있지 않기에, 여러 방식으로 Python을 설치할 수 있습니다.

* [Chocolatey](https://chocolatey.org/)
* [Python website](https://www.python.org/downloads/windows/)
* [Anaconda](#anaconda)

> 만약 PyTorch를 설치하기 위해 Anaconda를 사용한다면, PyTorch 어플리케이션을 실행하기 위한 샌드박스 버전의 Python이 설치됩니다.

> 만약 Chocolatey를 사용한다면, 그리고 이전에 Chocolatey를 설치하지 않았다면, [명령어 프롬프트를 관리자 권한으로 실행](https://www.howtogeek.com/194041/how-to-open-the-command-prompt-as-administrator-in-windows-8.1/)했는지 확인해야 합니다.

Chocolatey 기반의 설치를 위해, 다음 명령어를 [관리자 권한 프롬프트](https://www.howtogeek.com/194041/how-to-open-the-command-prompt-as-administrator-in-windows-8.1/)에서 실행합니다:

```bash
choco install python
```

### 패키지 매니저
{: #windows-package-manager}

PyTorch 바이너리를 설치하기 위해, 적어도 둘 중 하나 이상의 지원되는 패키지 매니저가 필요합니다. [Anaconda](https://www.anaconda.com/download/#windows) 그리고 [pip](https://pypi.org/project/pip/) 입니다. 권장되는 패키지 매니저는 Anaconda이며, Python과 `pip`를 포함한 PyTorch의 모든 의존성을 한번에, 샌드박스로 설치합니다.

#### Anaconda

Anaconda를 설치하기 위해, Python 3.x 의 [64-bit GUI 설치기](https://www.anaconda.com/download/#windows)를 사용합니다. 설치기 링크를 클릭한 후 `실행`을 선택합니다. Anaconda를 다운로드하고 설치기 프롬프트가 나타납니다. 기본 옵션은 일반적으로 같습니다.

#### pip

만약 [위의](#windows-python) 권장 설치 방법으로 Python을 설치했다면, [pip](https://pypi.org/project/pip/) 는 이미 설치되어 있습니다.

## 설치
{: #windows-installation}

### Anaconda
{: #windows-anaconda}

Anaconda를 이용해 PyTorch를 설치하기 위해서는, `Start | Anaconda3 | Anaconda Prompt` 를 통해 Anaconda 프롬프트를 열어야 합니다.

#### CUDA없이 설치

Acaconda를 통해 PyTorch를 설치하고, [CUDA를 지원하는](https://developer.nvidia.com/cuda-zone) 시스템이 없거나 CUDA가 필요하지 않다면, 위의 선택기에서 OS: Windows, 패키지: Conda, CUDA: None을 선택합니다.
그 후 나타나는 명령어를 실행합니다.

#### CUDA포함 설치

Anaconda를 통해 PyTorch를 설치하고, [CUDA를 지원하는](https://developer.nvidia.com/cuda-zone) 시스템을 갖고 있다면, 위의 선택기에서, OS: Windows, 패키지: Conda, 장치에 적합한 CUDA 버전을 선택합니다. 주로 최신 CUDA 버전이 더 좋습니다.
그 후 나타나는 명령어를 실행합니다.

### pip
{: #windows-pip}

#### CUDA없이 설치

pip를 통해 PyTorch를 설치하고, [CUDA를 지원하는](https://developer.nvidia.com/cuda-zone) 시스템이 없거나 CUDA가 필요하지 않다면, 위의 선택기에서 OS: Windows, 패키지: Pip, CUDA: None을 선택합니다.
그 후 나타나는 명령어를 실행합니다. 

#### CUDA포함 설치

pip 통해 PyTorch를 설치하고, [CUDA를 지원하는](https://developer.nvidia.com/cuda-zone) 시스템을 갖고 있다면, 위의 선택기에서, OS: Windows, 패키지: Pip, 장치에 적합한 CUDA 버전을 선택합니다. 주로 최신 CUDA 버전이 더 좋습니다.
그 후 나타나는 명령어를 실행합니다.

## 확인
{: #windows-verification}

PyTorch가 정상적으로 설치되었는지 확인하기 위해, 예제 PyTorch 코드를 실행시켜 설치를 확인할 수 있습니다. 아래 코드로 무작위로 초기화되는 텐서를 만듭니다.

명령줄에 다음과 같이 입력합니다.


```bash
python
```

이후 다음의 코드를 입력합니다.

```python
import torch
x = torch.rand(5, 3)
print(x)
```

출력은 다음과 비슷하게 나옵니다.

```
tensor([[0.3380, 0.3845, 0.3217],
        [0.8337, 0.9050, 0.2650],
        [0.2979, 0.7141, 0.9069],
        [0.1449, 0.1132, 0.1375],
        [0.4675, 0.3947, 0.1426]])
```

추가로, PyTorch로 GPU 드라이버와 CUDA를 활성화 및 접근 가능한지 확인하기 위해, 아래 명령어를 실행하여 CUDA 드라이버가 활성화되는지 확인합니다.

```python
import torch
torch.cuda.is_available()
```

## 소스에서 빌드하기
{: #windows-from-source}

대다수의 PyTorch 사용자를 위해, 패키지 매니저를 통한 미리 빌드된 바이너리를 설치하는것이 최고의 경험을 제공합니다. 하지만, 가장 최신(bleeding edge)의 PyTorch 코드를 설치하기 원하거나, 테스트 또는 PyTorch의 코어의 실제 개발이 필요할 때가 있습니다. 최신 PyTorch 코드를 설치하기 위해서는, [소스에서 PyTorch를 빌드](https://github.com/pytorch/pytorch#from-source)해야 합니다.

### 선수사항
{: #windows-prerequisites-2}

1. [Anaconda](#anaconda)를 설치합니다.
2. 만약 [CUDA를 지원하는 GPU](https://developer.nvidia.com/cuda-gpus)를 가진 시스템이면, [CUDA](https://developer.nvidia.com/cuda-downloads)를 설치합니다.
3. 만약 Windows에서 빌드한다면, MSVC 툴셋을 포함한 Visual Studio와 NVTX가 필요합니다. 의존성에 대한 자세한 준비사항은 [여기](https://github.com/pytorch/pytorch#from-source)에서 확인할 수 있습니다.
4. 다음 링크의 지시 사항을 따릅니다: [https://github.com/pytorch/pytorch#from-source](https://github.com/pytorch/pytorch#from-source)

[위에](#windows-verification) 설명한대로 제대로 설치가 되었는지 확인할 수 있습니다.