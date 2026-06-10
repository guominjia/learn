(function () {
  const header = document.querySelector('.site-header');
  if (!header) return;

  let isScrolled = false;

  // Toggle header compact state on scroll
  function updateHeaderScroll() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const shouldBeCompact = scrollTop > 80;

    if (shouldBeCompact !== isScrolled) {
      isScrolled = shouldBeCompact;
      if (isScrolled) {
        header.classList.add('is-scrolled');
      } else {
        header.classList.remove('is-scrolled');
      }
    }
  }

  // Passive scroll listener for better performance
  window.addEventListener('scroll', updateHeaderScroll, { passive: true });

  // Initial check
  updateHeaderScroll();

  // Generate table of contents from page content
  function generateTOC() {
    const contentBody = document.querySelector('.content-body');
    if (!contentBody) return;

    const headings = Array.from(contentBody.querySelectorAll('h2, h3')).filter(
      h => h.textContent.trim()
    );

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
    let currentH2Item = null;

    headings.forEach(h => {
      const level = h.tagName.toLowerCase();
      const text = h.textContent;
      const id = h.id;

      const item = document.createElement('li');
      const link = document.createElement('a');
      link.href = '#' + id;
      link.textContent = text;
      link.className = 'toc-link';
      link.dataset.targetId = id;
      item.appendChild(link);

      if (level === 'h2') {
        tocList.appendChild(item);
        currentH2Item = item;
        return;
      }

      if (level === 'h3' && currentH2Item) {
        let subList = currentH2Item.querySelector(':scope > ul');
        if (!subList) {
          subList = document.createElement('ul');
          currentH2Item.appendChild(subList);
        }
        subList.appendChild(item);
      } else {
        tocList.appendChild(item);
      }
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

  // Initialize TOC when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', generateTOC);
  } else {
    generateTOC();
  }
})();
