$(document).ready(function(){
    $(".xp-menubar").on('click',function() {
      $("#sidebar").toggleClass('active');
      $("#contenido").toggleClass('active');
    });
    
    $('.xp-menubar, .body-overlay').on('click',function() {
       $("#sidebar, .body-overlay").toggleClass('show-nav');
    });
});