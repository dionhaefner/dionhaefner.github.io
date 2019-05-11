    function decode(a) {
      // ROT13 : a Caesar cipher
      // letter -> letter' such that code(letter') = (code(letter) + 13) modulo 26
      return a.replace(/[a-zA-Z]/g, function(c){
        return String.fromCharCode((c <= "Z" ? 90 : 122) >= (c = c.charCodeAt(0) + 13) ? c : c - 26);
      })
    };
