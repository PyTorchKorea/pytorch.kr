/**
 * PyTorch.KR Community Widget
 * Fetches latest topics from discuss.pytorch.kr and renders them on the homepage.
 * Requires CORS to be enabled on Discourse:
 *   Admin → Settings → Security → cors origins → https://pytorch.kr
 */
(function () {
  'use strict';

  var DISCOURSE_URL = 'https://discuss.pytorch.kr';
  var API_URL = DISCOURSE_URL + '/latest.json';
  var TOPIC_COUNT = 8;
  var FETCH_TIMEOUT_MS = 8000;
  var CONTAINER_ID = 'community-topics-list';

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function formatDate(isoString) {
    var d = new Date(isoString);
    var now = new Date();
    var diffMs = now - d;
    var diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    var diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffHours < 1) return '방금 전';
    if (diffHours < 24) return diffHours + '시간 전';
    if (diffDays < 7) return diffDays + '일 전';

    var month = d.getMonth() + 1;
    var day = d.getDate();
    return month + '월 ' + day + '일';
  }

  function formatNumber(num) {
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return String(num);
  }

  function buildThumbnail(imageUrl) {
    if (imageUrl) {
      return (
        '<div class="topic-thumbnail">' +
          '<img src="' + escapeHtml(imageUrl) + '" alt="" width="80" height="60" loading="lazy">' +
        '</div>'
      );
    }
    return (
      '<div class="topic-thumbnail topic-thumbnail--placeholder">' +
        '<span>💬</span>' +
      '</div>'
    );
  }

  function buildTopicCard(topic) {
    var url = DISCOURSE_URL + '/t/' + encodeURIComponent(topic.slug) + '/' + encodeURIComponent(topic.id);
    return (
      '<div class="col-md-6">' +
        '<a href="' + url + '" target="_blank" rel="noopener" class="community-topic-item">' +
          buildThumbnail(topic.image_url) +
          '<div class="topic-body">' +
            '<div class="topic-title">' + escapeHtml(topic.title) + '</div>' +
            '<div class="topic-meta">' +
              '<span class="topic-date">' + formatDate(topic.last_posted_at || topic.created_at) + '</span>' +
              '<span class="topic-stat">👁 ' + formatNumber(topic.views) + '</span>' +
              '<span class="topic-stat">💬 ' + topic.posts_count + '</span>' +
              (topic.like_count > 0 ? '<span class="topic-stat">❤️ ' + topic.like_count + '</span>' : '') +
            '</div>' +
          '</div>' +
        '</a>' +
      '</div>'
    );
  }

  function renderTopics(topics) {
    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    var filtered = topics.filter(function (t) { return !t.pinned; });
    var display = filtered.slice(0, TOPIC_COUNT);

    if (display.length === 0) {
      renderFallback(container);
      return;
    }

    var html = '<div class="row g-2">' + display.map(buildTopicCard).join('') + '</div>';
    container.innerHTML = html;
  }

  function renderFallback(container) {
    container.innerHTML =
      '<div class="community-fallback">' +
        '<p>커뮤니티의 최신 글을 확인해보세요!</p>' +
        '<a href="' + DISCOURSE_URL + '" target="_blank" rel="noopener" class="btn btn-lg community-btn">커뮤니티 바로가기 →</a>' +
      '</div>';
  }

  function fetchWithTimeout(url, timeoutMs) {
    if (typeof AbortController === 'function') {
      var controller = new AbortController();
      var timer = setTimeout(function () { controller.abort(); }, timeoutMs);
      return fetch(url, { signal: controller.signal }).finally(function () {
        clearTimeout(timer);
      });
    }
    return fetch(url);
  }

  var DEV_MOCK_TOPICS = [
    { id: 9110, title: 'Google, OpenClaw 및 OpenCode 등으로 인한 OAuth 약관 위배로 블럭된 Antigravity 계정에 복구 방법을 제공한다고 밝혀', slug: 'google-openclaw-opencode-oauth-antigravity', image_url: null, last_posted_at: new Date(Date.now() - 5 * 60 * 1000).toISOString(), views: 2, posts_count: 3, like_count: 0 },
    { id: 9106, title: 'Anthropic, AI 기반 취약점 탐지 도구 Claude Code Security 발표 및 연구용 미리보기 사용 신청 시작', slug: 'anthropic-ai-claude-code-security', image_url: null, last_posted_at: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(), views: 6, posts_count: 1, like_count: 0 },
    { id: 9045, title: '저 github에 pose-transfer모델을 한번 올려봤어요. 괜찮을까요?', slug: 'github-pose-transfer', image_url: 'https://discuss.pytorch.kr/uploads/default/optimized/2X/a/abc123_2_80x60.png', last_posted_at: new Date(Date.now() - 7 * 60 * 60 * 1000).toISOString(), views: 67, posts_count: 2, like_count: 4 },
    { id: 4833, title: '4070 Ti같은 12GB 메모리를 가진 GPU에서 작동하는 LLM이 있을까요?', slug: '4070-ti-12gb-gpu-llm', image_url: null, last_posted_at: new Date(Date.now() - 8 * 60 * 60 * 1000).toISOString(), views: 885, posts_count: 7, like_count: 1 },
    { id: 9091, title: 'Codex 하네스의 비밀 풀기: App Server 구축기 (Unlocking the Codex harness: how we built the App Server)', slug: 'codex-app-server', image_url: null, last_posted_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), views: 99, posts_count: 1, like_count: 0 },
    { id: 9080, title: 'PyTorch 2.6 새로운 기능 살펴보기 - torch.compile 성능 개선 및 FlexAttention 안정화', slug: 'pytorch-2-6-new-features', image_url: null, last_posted_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), views: 342, posts_count: 5, like_count: 12 },
    { id: 9072, title: 'LoRA fine-tuning 시 메모리 부족 오류 해결 방법 공유합니다', slug: 'lora-fine-tuning-oom-fix', image_url: null, last_posted_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(), views: 215, posts_count: 8, like_count: 7 },
    { id: 9060, title: 'vLLM vs TGI - 추론 서버 성능 비교 및 운영 경험 공유', slug: 'vllm-vs-tgi-comparison', image_url: null, last_posted_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(), views: 1200, posts_count: 14, like_count: 23 }
  ];

  function isLocalhost() {
    return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
  }

  function init() {
    if (typeof fetch !== 'function') return;

    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

    if (isLocalhost()) {
      renderTopics(DEV_MOCK_TOPICS);
      return;
    }

    fetchWithTimeout(API_URL, FETCH_TIMEOUT_MS)
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (data) {
        var topics = (data.topic_list && data.topic_list.topics) || [];
        renderTopics(topics);
      })
      .catch(function () {
        renderFallback(container);
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
