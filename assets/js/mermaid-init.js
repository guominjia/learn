(function () {
  function renderMermaid() {
    if (!window.mermaid) {
      return;
    }

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose"
    });

    var codeBlocks = document.querySelectorAll(
      "pre code.language-mermaid, pre code.lang-mermaid"
    );

    codeBlocks.forEach(function (codeBlock) {
      var pre = codeBlock.closest("pre");
      if (!pre) {
        return;
      }

      var container = document.createElement("div");
      container.className = "mermaid";
      container.textContent = codeBlock.textContent;
      pre.replaceWith(container);
    });

    window.mermaid.run({
      querySelector: ".mermaid"
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMermaid);
  } else {
    renderMermaid();
  }
})();
