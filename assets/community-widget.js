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
  var TOPIC_COUNT = 5;
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

  function buildTopicCard(topic) {
    var url = DISCOURSE_URL + '/t/' + encodeURIComponent(topic.slug) + '/' + encodeURIComponent(topic.id);
    return (
      '<a href="' + url + '" target="_blank" rel="noopener" class="community-topic-item">' +
        '<div class="topic-title">' + escapeHtml(topic.title) + '</div>' +
        '<div class="topic-meta">' +
          '<span class="topic-date">' + formatDate(topic.last_posted_at || topic.created_at) + '</span>' +
          '<span class="topic-stat">👁 ' + formatNumber(topic.views) + '</span>' +
          '<span class="topic-stat">💬 ' + topic.posts_count + '</span>' +
          (topic.like_count > 0 ? '<span class="topic-stat">❤️ ' + topic.like_count + '</span>' : '') +
        '</div>' +
      '</a>'
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

    var html = display.map(buildTopicCard).join('');
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

  function init() {
    if (typeof fetch !== 'function') return;

    var container = document.getElementById(CONTAINER_ID);
    if (!container) return;

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
