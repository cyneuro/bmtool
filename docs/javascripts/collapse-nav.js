// Collapse navigation sections by default
document.addEventListener("DOMContentLoaded", function () {
  // Remove the active class from all nested navigation items to collapse them
  document.querySelectorAll(".md-nav__item--nested").forEach(function (item) {
    item.classList.remove("md-nav__item--nested--active");
  });
});
