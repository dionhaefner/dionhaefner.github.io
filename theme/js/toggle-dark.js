// mostly from https://css-tricks.com/a-complete-guide-to-dark-mode-on-the-web

window.addEventListener('DOMContentLoaded', function() {
    // Select the button
    const btn = document.querySelector(".btn-toggle");
    // Check for dark theme preference at the OS level
    const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)");

    // Get the user's theme preference from local storage, if it's available
    const currentTheme = localStorage.getItem("theme");

    if ((prefersDarkScheme.matches && currentTheme != "light") || currentTheme == "dark") {
        btn.classList.add("fa-sun-o");
        document.body.classList.add("dark-theme");
    }
    else {
        btn.classList.add("fa-moon-o");
        document.body.classList.remove("dark-theme");
    }

    // Listen for a click on the button 
    btn.addEventListener("click", function() {
        document.body.classList.toggle("dark-theme");
        var theme = document.body.classList.contains("dark-theme") ? "dark" : "light";
        if (theme == "light") {
            btn.classList.add("fa-moon-o");
            btn.classList.remove("fa-sun-o");
        } else {
            btn.classList.remove("fa-moon-o");
            btn.classList.add("fa-sun-o");
        }
        // Finally, let's save the current preference to localStorage to keep using it
        localStorage.setItem("theme", theme);
    });
});