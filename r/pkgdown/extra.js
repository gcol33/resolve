// A block written in R and in Python sits in a `.lang-pair` div: the R chunk with its output first,
// then the Python block. Each pair gets one R | Python control above it, and the choice is one
// setting for the whole site, kept in localStorage and read from `?lang=` on a link. Prose that
// holds for one language alone goes in a `.lang-r-only` or `.lang-py-only` div.
(function () {
  var KEY = "resolve-lang";
  var LANGS = [["r", "R"], ["python", "Python"]];
  var root = document.documentElement;

  function stored() {
    try { return localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function store(lang) {
    try { localStorage.setItem(KEY, lang); } catch (e) { /* private window: the choice lasts the page */ }
  }
  function valid(lang) {
    return lang === "r" || lang === "python" ? lang : null;
  }

  var requested = valid(new URLSearchParams(window.location.search).get("lang"));
  if (requested) store(requested);
  root.setAttribute("data-lang", requested || valid(stored()) || "r");

  var style = document.createElement("style");
  style.textContent = [
    'html[data-lang="r"] .lang-pair > [data-part="python"],',
    'html[data-lang="python"] .lang-pair > [data-part="r"],',
    'html[data-lang="r"] .lang-py-only,',
    'html[data-lang="python"] .lang-r-only { display: none; }',
    ".lang-bar { display: flex; justify-content: flex-end; margin-bottom: .25rem; }",
    ".lang-tabs { display: inline-flex; border: 1px solid var(--bs-border-color);",
    "  border-radius: .375rem; overflow: hidden; font-size: .8125rem; line-height: 1.5; }",
    ".lang-tabs button { border: 0; background: transparent; color: var(--bs-secondary-color);",
    "  padding: .0625rem .75rem; cursor: pointer; font: inherit; }",
    ".lang-tabs button + button { border-left: 1px solid var(--bs-border-color); }",
    '.lang-tabs button[aria-selected="true"] { background: var(--bs-tertiary-bg);',
    "  color: var(--bs-body-color); font-weight: 500; }",
    ".lang-tabs button:focus-visible { outline: 2px solid var(--bs-primary); outline-offset: -2px; }",
    ".lang-pair > [data-part] > :last-child { margin-bottom: 0; }"
  ].join("\n");
  document.head.appendChild(style);

  function wrap(nodes, part) {
    var div = document.createElement("div");
    div.setAttribute("data-part", part);
    nodes[0].parentNode.insertBefore(div, nodes[0]);
    nodes.forEach(function (node) { div.appendChild(node); });
    return div;
  }

  function split(pair) {
    var children = Array.prototype.slice.call(pair.children);
    var at = children.findIndex(function (node) {
      return node.querySelector("pre.python, pre.bash") || node.matches("pre.python, pre.bash");
    });
    if (at < 1) return false;
    wrap(children.slice(0, at), "r");
    wrap(children.slice(at), "python");
    return true;
  }

  function buttons(pair) {
    var bar = document.createElement("div");
    bar.className = "lang-bar";
    var tabs = document.createElement("div");
    tabs.className = "lang-tabs";
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Language");
    LANGS.forEach(function (entry) {
      var button = document.createElement("button");
      button.type = "button";
      button.setAttribute("role", "tab");
      button.setAttribute("data-lang-button", entry[0]);
      button.textContent = entry[1];
      button.addEventListener("click", function () { choose(entry[0], button); });
      tabs.appendChild(button);
    });
    bar.appendChild(tabs);
    pair.insertBefore(bar, pair.firstChild);
  }

  function mark() {
    var lang = root.getAttribute("data-lang");
    document.querySelectorAll("[data-lang-button]").forEach(function (button) {
      button.setAttribute("aria-selected", String(button.getAttribute("data-lang-button") === lang));
    });
  }

  // The control that was clicked stays where it is on the screen: the two languages differ in
  // height, and the blocks above it change together with its own.
  function choose(lang, button) {
    var before = button.getBoundingClientRect().top;
    root.setAttribute("data-lang", lang);
    store(lang);
    mark();
    window.scrollBy(0, button.getBoundingClientRect().top - before);
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".lang-pair").forEach(function (pair) {
      if (split(pair)) buttons(pair);
    });
    mark();
  });
})();
