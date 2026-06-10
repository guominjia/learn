(function () {
  // Toggle header compact state on scroll
  function updateHeaderState() {
    const header = document.querySelector('.site-header');
    if (!header) return;

    if (window.scrollY > 60) {
      header.classList.add('is-scrolled');
    } else {
      header.classList.remove('is-scrolled');
    }
  }

  // Generate table of contents from page content
  function generateTOC() {
    const contentBody = document.querySelector('.content-body');
    if (!contentBody) return;

    const headings = Array.from(contentBody.querySelectorAll('h2, h3'))
      .filter(h => h.textContent.trim());

    if (headings.length === 0) return;

    // Add IDs to headings if not present
    headings.forEach((h, idx) => {
      if (!h.id) {
        h.id = 'heading-' + idx;
      }
    });

    // Create TOC nav
    const tocNav = document.createElement('nav');
    tocNav.className = 'page-toc';
    tocNav.setAttribute('aria-label', 'Page contents');

    const tocList = document.createElement('ul');
    let currentLevel = 0;
    let currentList = tocList;
    const stack = [{ level: 0, element: tocList }];

    headings.forEach(h => {
      const level = parseInt(h.tagName[1]);
      const text = h.textContent;
      const id = h.id;

      // Adjust nesting level
      while (currentLevel >= level && stack.length > 1) {
        stack.pop();
        currentLevel--;
      }

      if (level > currentLevel) {
        for (let i = currentLevel; i < level; i++) {
          const newList = document.createElement('ul');
          const lastItem = currentList.lastElementChild;
          if (lastItem) {
            lastItem.appendChild(newList);
          } else {
            currentList.appendChild(newList);
          }
          stack.push({ level: i + 1, element: newList });
          currentList = newList;
        }
        currentLevel = level;
      }

      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = '#' + id;
      link.textContent = text;
      link.className = 'toc-link';
      link.dataset.targetId = id;
      item.appendChild(link);
      currentList.appendChild(item);
    });

    tocNav.appendChild(tocList);
    
    // Insert TOC before content body
    const contentWrap = document.querySelector('.content-wrap');
    if (contentWrap && contentWrap.querySelector('.content-head')) {
      contentWrap.querySelector('.content-head').insertAdjacentElement('afterend', tocNav);
    }

    // Setup IntersectionObserver for active TOC highlight
    setupTOCObserver(headings);
  }

  // Setup IntersectionObserver to highlight TOC active link
  function setupTOCObserver(headings) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const id = entry.target.id;
        const tocLink = document.querySelector(`.toc-link[data-target-id="${id}"]`);
        if (tocLink) {
          if (entry.isIntersecting) {
            document.querySelectorAll('.toc-link').forEach(link => {
              link.classList.remove('is-active');
            });
            tocLink.classList.add('is-active');
          }
        }
      });
    }, {
      rootMargin: '-80px 0px -66%'
    });

    headings.forEach(h => observer.observe(h));
  }

  // Smooth scroll behavior (fallback for older browsers)
  function enableSmoothScroll() {
    document.documentElement.style.scrollBehavior = 'smooth';
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      generateTOC();
      enableSmoothScroll();
      updateHeaderState();
    });
  } else {
    generateTOC();
    enableSmoothScroll();
    updateHeaderState();
  }

  window.addEventListener('scroll', updateHeaderState, { passive: true });
})();
