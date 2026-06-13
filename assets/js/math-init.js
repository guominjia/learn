(function () {
  function renderMath() {
    if (!window.renderMathInElement) {
      return;
    }

    document.querySelectorAll('.content-body').forEach(function (container) {
      if (container.dataset.mathRendered === 'true') {
        return;
      }

      window.renderMathInElement(container, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true }
        ],
        throwOnError: false,
        strict: 'ignore'
      });

      container.dataset.mathRendered = 'true';
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderMath);
  } else {
    renderMath();
  }
})();
