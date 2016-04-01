$(document).ready(function(e) {
    if (window.location.hash) {
        $(window).scrollTop(0,0);
        $('html,body').animate({
          scrollTop: $(window.location.hash).offset().top
        }, 1000);
    }

    $('a').smoothScroll({offset: -100});



    /* Every time the window is scrolled ... */
    $(window).scroll( function(){
    
        /* Check the location of each desired element */
        $('.hideme').each( function(i){
            
            var top_of_object = $(this).offset().top;
            var middle_of_window = $(window).scrollTop() + $(window).height() / 2;
            
            if( middle_of_window > top_of_object ){    
                $(this).animate({'opacity':'1'},500);
                    
            }
            
        }); 
    
    });
});

