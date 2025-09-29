var mobileMenu = {
  bind: function() {
    $("[data-behavior='open-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      $(".mobile-main-menu").addClass("open");
      $("body").addClass('no-scroll');

      mobileMenu.listenForResize();
    });

    $("[data-behavior='close-mobile-menu']").on('click', function(e) {
      e.preventDefault();
      mobileMenu.close();
    });

    // 모바일에서 드롭다운 토글 (동적 처리)
    $("[class$='-mobile-toggle']").on('click', function(e) {
      e.preventDefault();
      var $this = $(this);
      var className = $this.attr('class').replace('-mobile-toggle', '-mobile-menu');
      var menu = $this.siblings('.' + className);
      menu.slideToggle();
    });
  },

  listenForResize: function() {
    $(window).on('resize.ForMobileMenu', function() {
      if ($(this).width() > 768) {
        mobileMenu.close();
      }
    });
  },

  close: function() {
    $(".mobile-main-menu").removeClass("open");
    $("body").removeClass('no-scroll');
    $(window).off('resize.ForMobileMenu');
  }
};
