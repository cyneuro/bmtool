window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.arithmatex').forEach(function(element) {
    element.classList.add('mathjax_process');
  });
});
