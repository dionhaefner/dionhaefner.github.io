$(document).ready(function(e) {
    if (window.location.hash) {
        $(window).scrollTop(0,0);
        $('html,body').animate({
          scrollTop: $(window.location.hash).offset().top
        }, 1000);
    }

    $('a').smoothScroll();
});
