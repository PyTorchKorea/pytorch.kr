---
layout: get_started
title: 클라우드에서 시작하기
permalink: /get-started/cloud-partners/
background-class: get-started-background
body-class: get-started
order: 2
published: true
get-started-via-cloud: true
---

## 클라우드에서 시작하기

<div class="container-fluid quick-start-module quick-starts">
  <div class="cloud-options-col">
    <p>클라우드 플랫폼은 딥러닝 모델을 학습하고 배포하기 위한 강력한 하드웨어와 인프라를 제공합니다. 아래에서 클라우드 플랫폼을 선택하여 PyTorch를 시작해보세요.</p>
    {% include quick_start_cloud_options.html %}
  </div>
</div>

---

{% capture aws %}
{% include_relative installation/aws.md %}
{% endcapture %}

{% capture azure %}
{% include_relative installation/azure.md %}
{% endcapture %}

{% capture google-cloud %}
{% include_relative installation/google-cloud.md %}
{% endcapture %}


<div id="cloud">
  <div class="platform aws">{{aws | markdownify }}</div>
  <div class="platform google-cloud">{{google-cloud | markdownify }}</div>
  <div class="platform microsoft-azure">{{azure | markdownify }}</div>
</div>

<script page-id="get-started-via-cloud-partners" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>
