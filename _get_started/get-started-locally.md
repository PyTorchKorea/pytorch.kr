---
layout: get_started
title: 직접 설치하기
permalink: /get-started/locally/
background-class: get-started-background
body-class: get-started
order: 1
published: true
get-started-locally: true
redirect_from: "/get-started/"
---

## 직접 설치하기

<div class="container-fluid quick-start-module quick-starts">
  <div class="row">
    <div class="col-md-12">
      {% include quick_start_local.html %}
    </div>
  </div>
</div>

---

{% capture mac %}
{% include_relative installation/mac.md %}
{% endcapture %}

{% capture linux %}
{% include_relative installation/linux.md %}
{% endcapture %}

{% capture windows %}
{% include_relative installation/windows.md %}
{% endcapture %}


<div id="installation">
  <div class="os macos">{{ mac | markdownify }}</div>
  <div class="os linux selected">{{ linux | markdownify }}</div>
  <div class="os windows">{{ windows | markdownify }}</div>
</div>

<script page-id="get-started-locally" src="{{ site.baseurl }}/assets/menu-tab-selection.js"></script>
<script src="{{ site.baseurl }}/assets/quick-start-module.js"></script>
<script src="{{ site.baseurl }}/assets/show-screencast.js"></script>
