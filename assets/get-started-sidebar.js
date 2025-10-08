// Create the sidebar menus for each OS and Cloud Partner

// For pytorch.md page: build sidebar from all h2/h3 tags in the article
if ($('.pytorch-2').length > 0) {
  buildSidebarMenu('.pytorch-article', '#get-started-locally-sidebar-list');
  $('.get-started-locally-sidebar li').show(); // Show all items for pytorch page
}

$([".macos", ".linux", ".windows"]).each(function(index, osClass) {
  buildSidebarMenu(osClass, "#get-started-locally-sidebar-list");
});

$([".alibaba", ".aws", ".microsoft-azure", ".google-cloud", ".lightning-studios"]).each(function(index, cloudPartner) {
  buildSidebarMenu(cloudPartner, "#get-started-cloud-sidebar-list");
});

$(["macos", "linux", "windows"]).each(function(index, osClass) {
  $("#" + osClass).on("click", function() {
    showSidebar(osClass, ".get-started-locally-sidebar li");
  });
});

// Show cloud partner side nav on click or hide side nav if already open
$(["alibaba", "aws", "microsoft-azure", "google-cloud", "lightning-studios"]).each(function(index, sidebarClass) {
  $("#" + sidebarClass).click(function() {
    showSidebar(sidebarClass, ".get-started-cloud-sidebar li");
    // alibaba filter for centering cloud module
    if (sidebarClass == "alibaba") {
      $(".article-wrapper").parent().removeClass("col-md-8 offset-md-1").addClass("col-md-12");
      $(".cloud-nav").hide();
    } else {
      $(".article-wrapper").parent().removeClass("col-md-12").addClass("col-md-8 offset-md-1");
      $(".cloud-nav").show();
    }
    if ($("#" + sidebarClass).parent().hasClass("open")) {
      $(".get-started-cloud-sidebar li").hide();
      $(".cloud-nav").hide();
      $(".article-wrapper").parent().removeClass("col-md-8 offset-md-1").addClass("col-md-12");
    }
  })
});

function buildSidebarMenu(menuClass, menuItem) {
  var currentH2 = null;
  var h2Index = 0;

  $(menuClass + " > h2," + menuClass + " > h3").each(function(index, element) {
    menuClass = menuClass.replace(".", "");
    var tagName = $(element).get(0).tagName;

    if (tagName == "H2") {
      // H2 항목: 토글 가능한 메인 아이템
      h2Index++;
      var h2Id = "sidebar-h2-" + h2Index;
      currentH2 = h2Id;

      $(menuItem).append(
        "<li class='" + menuClass + " sidebar-h2' data-h2-id='" + h2Id + "' style='display:none'>" +
          "<a href='#" + this.id + "' class='sidebar-h2-link'>" +
            "<span class='toggle-icon'>▸</span>" +
            "<span class='menu-text'>" + this.textContent + "</span>" +
          "</a>" +
        "</li>"
      );
    } else if (tagName == "H3" && currentH2) {
      // H3 항목: 숨겨진 서브아이템
      $(menuItem).append(
        "<li class='" + menuClass + " subitem sidebar-h3' data-parent='" + currentH2 + "' style='display:none'>" +
          "<a href='#" + this.id + "'>" + this.textContent + "</a>" +
        "</li>"
      );
    }
  });
}

function showSidebar(selectedClass, menuItem) {
  // Hide all of the menu items at first
  // Then filter for the selected OS/cloud partner
  $(menuItem)
    .hide()
    .filter(function() {
      return $(this)
        .attr("class")
        .includes(selectedClass);
    })
    .show();
}

// H2 토글 기능
$(document).on("click", ".get-started-locally-sidebar li.sidebar-h2 .sidebar-h2-link", function(e) {
  var $li = $(this).closest("li");
  var h2Id = $li.data("h2-id");
  var $icon = $li.find(".toggle-icon");
  var $children = $(".get-started-locally-sidebar li.sidebar-h3[data-parent='" + h2Id + "']");

  // 토글 아이콘 클릭 시에만 펼침/접힘, 텍스트 클릭 시에는 스크롤만
  var isIconClick = $(e.target).hasClass("toggle-icon") || $(e.target).closest(".toggle-icon").length > 0;

  if (isIconClick) {
    e.preventDefault();

    // 토글 상태 변경
    if ($li.hasClass("expanded")) {
      $li.removeClass("expanded");
      $icon.text("▸");
      $children.slideUp(200);
    } else {
      $li.addClass("expanded");
      $icon.text("▾");
      $children.slideDown(200);
    }
  } else {
    // 텍스트 클릭 시 해당 섹션으로 스크롤
    removeActiveClass();
    addActiveClass($li);
  }
});

// H3 링크 클릭 시
$(document).on("click", ".get-started-locally-sidebar li.sidebar-h3 a", function() {
  removeActiveClass();
  addActiveClass($(this).closest("li"));
});

// 스크롤 시 현재 섹션 하이라이트 및 자동 확장
var scrollTimeout;
$(window).on("scroll", function() {
  clearTimeout(scrollTimeout);
  scrollTimeout = setTimeout(function() {
    updateActiveSection();
  }, 100);
});

function updateActiveSection() {
  if ($('.pytorch-2').length === 0) return;

  var scrollPos = $(window).scrollTop() + 100;
  var $h2Elements = $(".pytorch-article > h2");
  var $h3Elements = $(".pytorch-article > h3");
  var currentH2 = null;
  var currentH3 = null;

  // 현재 보고 있는 H2 찾기
  $h2Elements.each(function() {
    if ($(this).offset().top <= scrollPos) {
      currentH2 = $(this).attr("id");
    }
  });

  // 현재 보고 있는 H3 찾기
  $h3Elements.each(function() {
    if ($(this).offset().top <= scrollPos) {
      currentH3 = $(this).attr("id");
    }
  });

  if (currentH2) {
    var $activeH2 = $(".get-started-locally-sidebar li.sidebar-h2 a[href='#" + currentH2 + "']").closest("li");

    // 먼저 모든 섹션을 접고 현재 섹션만 확장
    if ($activeH2.length) {
      // 다른 모든 H2 섹션 접기
      $(".get-started-locally-sidebar li.sidebar-h2").not($activeH2).each(function() {
        var $otherH2 = $(this);
        if ($otherH2.hasClass("expanded")) {
          $otherH2.removeClass("expanded");
          $otherH2.find(".toggle-icon").text("▸");
          var otherId = $otherH2.data("h2-id");
          $(".get-started-locally-sidebar li.sidebar-h3[data-parent='" + otherId + "']").slideUp(200);
        }
      });

      // 현재 섹션 자동 확장
      if (!$activeH2.hasClass("expanded")) {
        var h2Id = $activeH2.data("h2-id");
        $activeH2.addClass("expanded");
        $activeH2.find(".toggle-icon").text("▾");
        $(".get-started-locally-sidebar li.sidebar-h3[data-parent='" + h2Id + "']").slideDown(200);
      }
    }

    removeActiveClass();

    if (currentH3) {
      var $activeH3 = $(".get-started-locally-sidebar li.sidebar-h3 a[href='#" + currentH3 + "']").closest("li");
      if ($activeH3.length) {
        addActiveClass($activeH3);
        return;
      }
    }

    addActiveClass($activeH2);
  }
}

// 페이지 로드 시 초기 섹션 업데이트 (모든 메뉴 접힌 상태로 시작)
$(document).ready(function() {
  if ($('.pytorch-2').length > 0) {
    setTimeout(function() {
      updateActiveSection();
    }, 100);
  }
});

function removeActiveClass() {
  $(".get-started-locally-sidebar li a").each(function() {
    $(this).removeClass("active");
  });
}

function addActiveClass(element) {
  $(element)
    .find("a")
    .addClass("active");
}

if ($("#get-started-locally-sidebar-list").text() == "") {
  $("#get-started-shortcuts-menu").hide();
}
