---
layout: default
title: 기여하신 분들 소개 | PyTorchKR
permalink: contributors/
background-class: coc-background
body-class: features
---

<div class="jumbotron jumbotron-fluid">
    <div class="container">
        <h1>기여해주신 분들</h1>
        <p class="lead">
          파이토치 한국 사용자 모임은 많은 분들의 자발적인 기여로 성장하고 있습니다. <br />
          기여해주신 분들을 아래에서 만나보실 수 있습니다.
        </p>
    </div>
</div>

<div class="main-content-wrapper">
  <div class="main-content">
    <div class="container contributors">
      {% for repository in site.github.public_repositories %}
      {% if repository.contributors.size > 1 %}
      <h2><a href="{{ repository.html_url }}" target="_blank">{{ repository.name }} 저장소</a></h2>
        <ul>
        {% if repository.description.size >= 1 %}
          <li><b>소개</b>: {{ repository.description }}</li>
        {% endif %}
          <li><b>기여해주신 분들</b>: </li>
          <div class="row">
          {% for contributor in repository.contributors %}
            <div class="col-xl-2 col-lg-3 col-md-4 col-sm-6 col-12">
              <div class="card contributor-card">
                <div class="card-body">
                  <a href="{{ contributor.html_url }}" target="_blank">
                    <img src="{{ contributor.avatar_url }}" />
                    <p class="card-summary">@{{ contributor.login }}</p>
                  </a>
                </div>
              </div>
            </div>
          {% endfor %}
          </div>
        </ul>
      {% endif %}
      {% endfor %}
    </div>
  </div>
</div>