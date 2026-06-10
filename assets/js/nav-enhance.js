(function () {
  const header = document.querySelector('.site-header');
  if (!header) return;

  let isScrolled = false;

  // Toggle header compact state on scroll
  function updateHeaderScroll() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const shouldBeCompact = scrollTop > 40;
    const toc = document.querySelector('.page-toc');

    if (shouldBeCompact !== isScrolled) {
      isScrolled = shouldBeCompact;
      if (isScrolled) {
        header.classList.add('is-scrolled');
        if (toc) toc.classList.add('is-scrolled');
      } else {
        header.classList.remove('is-scrolled');
        if (toc) toc.classList.remove('is-scrolled');
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

    const contentWrap = document.querySelector('.content-wrap');
    if (!contentWrap) return;

    const oldTocInContent = contentWrap.querySelector('.page-toc');
    if (oldTocInContent) oldTocInContent.remove();

    const sidebar = document.querySelector('.sidebar');
    const oldTocInSidebar = sidebar ? sidebar.querySelector('.page-toc') : null;
    if (oldTocInSidebar) oldTocInSidebar.remove();

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

    if (header.classList.contains('is-scrolled')) {
      tocNav.classList.add('is-scrolled');
    }

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

    const tocToggle = document.createElement('button');
    tocToggle.type = 'button';
    tocToggle.className = 'page-toc-toggle';
    tocToggle.setAttribute('aria-expanded', 'false');
    tocToggle.setAttribute('aria-label', 'Toggle table of contents');
    tocToggle.innerHTML = '<span></span><span></span><span></span>';

    tocNav.classList.add('is-collapsed');
    tocToggle.addEventListener('click', () => {
      const isCollapsed = tocNav.classList.toggle('is-collapsed');
      tocToggle.setAttribute('aria-expanded', String(!isCollapsed));
    });

    tocNav.appendChild(tocToggle);
    tocNav.appendChild(tocList);
    
    // Prefer sidebar placement to avoid covering article content
    if (sidebar) {
      sidebar.prepend(tocNav);
    } else if (contentWrap.querySelector('.content-head')) {
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
