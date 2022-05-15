# 파이토치 한국 사용자 모임 홈페이지

## 개요

[파이토치 한국 사용자 모임 홈페이지](https://pytorch.kr) 저장소에 오신 것을 환영합니다! \
파이토치 한국 사용자 모임 홈페이지는 [PyTorch 공식 홈페이지 저장소](https://github.com/pytorch/pytorch.github.io)를 복제하여 만들었습니다. \
홈페이지 빌드를 위해 [Jekyll](https://jekyllrb.com/)과 [Bootstrap](https://getbootstrap.com/) 등을 사용하고 있습니다.

## 빌드하기

이 저장소를 빌드하기 위해 필요한 절차를 안내합니다.

### 필요 도구

이 저장소를 빌드하기 위해서는 아래 도구들의 필요합니다.

- [rbenv](https://github.com/rbenv/rbenv)
- [ruby-build](https://github.com/rbenv/ruby-build)
- [nvm](https://github.com/creationix/nvm)

macOS에서 [homebrew](https://brew.sh/)를 사용하신다면, 아래 명령어로 설치하실 수 있습니다:

```sh
  brew install rbenv ruby-build nvm
```

### 빌드 절차

#### ruby 및 bundler, 필요 패키지 설치

아래 명령어로 이 저장소에서 필요로 하는 ruby 버전에 맞춰 ruby와 bundler, 필요 패키지들을 설치합니다.

```sh
  rbenv install `cat .ruby-version`   # ruby 설치
  gem install bundler -v 2.3.13       # bundler 설치
  rbenv rehash
  bundle install                      # 필요 패키지 설치
  rbenv rehash
```

#### node.js 및 필요 패키지 설치

아래 명령어로 이 저장소에서 필요로 하는 node.js 버전에 맞춰 node.js 및 yarn, 필요 패키지들을 설치합니다.

```sh
  nvm install       # node.js 설치
  nvm use           # node.js 버전 적용
  npm install yarn  # 패키지 도구 설치
  yarn install      # 필요 패키지 설치
```

#### 홈페이지 빌드

아래 명령어로 jekyll을 사용하여 빌드합니다. 변경 사항을 실시간으로 확인하려면 `make build` 대신 `make serve` 명령어를 사용하면 됩니다.

```sh
  make build    # 빌드 결과물은 `./_site` 디렉토리에 저장됩니다.
  # make serve
```

* 참고: [기여해주신 분들](https://pytorch.kr/about/contributors) 페이지에서는 [jekyll/github-metadata](https://github.com/jekyll/github-metadata/tree/main/lib/jekyll-github-metadata) 플러그인을 사용하고 있으며 빌드를 위해서 `JEKYLL_GITHUB_TOKEN`이 필요합니다. \
  [기여해주신 분들](https://pytorch.kr/about/contributors) 페이지를 빌드해야 하는 경우, [jekyll/github-metadata](https://github.com/jekyll/github-metadata/tree/main/lib/jekyll-github-metadata)의 [Authentication 문서](https://github.com/jekyll/github-metadata/blob/main/docs/authentication.md)를 참고해주세요.

## 배포하기

[파이토치 한국 사용자 모임 홈페이지](https://pytorch.kr)는 [GitHub Pages](https://pages.github.com/)를 사용합니다. \
홈페이지는 [GitHub Actions](https://docs.github.com/en/actions)를 사용하여 자동으로 빌드 및 `gh-pages` 브랜치에 배포됩니다.


## 기여하기

홈페이지에서 잘못된 내용 또는 변경 / 추가하고 싶은 내용이 있다면 이 저장소에 기여하실 수 있습니다. \
[기여하기](CONTRIBUTING.md) 문서를 참고해주세요.
