$(document).ready(function(e) {

    var scrollToCenter = function(e) {
      var el = $( e.prop('hash') );
      if (e.prop('hash')) {
          var elHeight = el.height();
          var windowHeight = $(window).height();
          var offset;

          if (elHeight < windowHeight) {
            offset = ((elHeight / 2) - (windowHeight / 2));
          }
          else {
            offset = 0;
          }
          $.smoothScroll({offset: offset, speed: "auto", scrollTarget: el});
          return false;
       }
    };

    /* If current URL contains a hash: scroll to anchor */
    if (window.location.hash) {
        $(window).scrollTop(0,0);
        scrollToCenter($(window.location));
    }

    /* Animate anchor scrolling */
    $('a[href*="\\#"]').on('click', function(e) {
        e.preventDefault();
        scrollToCenter($(this));
    });

});
