// Collapse navigation sections by default
document.addEventListener("DOMContentLoaded", function () {
  // Uncheck all nested navigation checkboxes to collapse them
  document
    .querySelectorAll(".md-nav__item--nested .md-nav__toggle")
    .forEach(function (toggle) {
      toggle.checked = false;
    });
});
