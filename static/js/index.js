'use strict';

async function updateGitHubStars() {
  const count = document.getElementById('github-star-count');
  if (!count) return;
  const cacheKey = 'tokenhsi-github-stars';
  const showCount = (stars) => {
    count.textContent = stars.toLocaleString('en-US');
    count.setAttribute('aria-label', `${stars} GitHub stars`);
  };
  try {
    const cached = JSON.parse(localStorage.getItem(cacheKey));
    if (Number.isInteger(cached?.stars) && cached.stars >= 0) {
      showCount(cached.stars);
      const age = Date.now() - cached.updatedAt;
      if (age >= 0 && age < 60 * 60 * 1000) return;
    }
  } catch { /* The live request also works when browser storage is unavailable. */ }
  try {
    const response = await fetch('https://api.github.com/repos/liangpan99/TokenHSI', {
      headers: { Accept: 'application/vnd.github+json' },
    });
    if (!response.ok) return;
    const { stargazers_count: stars } = await response.json();
    if (!Number.isInteger(stars) || stars < 0) return;
    showCount(stars);
    try {
      localStorage.setItem(cacheKey, JSON.stringify({ stars, updatedAt: Date.now() }));
    } catch { /* Keep the displayed count even if storage is blocked. */ }
  } catch { /* Keep the last known count if GitHub is temporarily unavailable. */ }
}
updateGitHubStars();

document.querySelectorAll('[data-gallery]').forEach((gallery) => {
  const track = gallery.querySelector('.gallery-track');
  const controls = gallery.querySelector('.gallery-controls');
  const previous = controls.querySelector('[data-direction="-1"]');
  const next = controls.querySelector('[data-direction="1"]');
  controls.hidden = false;

  const updateControls = () => {
    previous.disabled = track.scrollLeft <= 2;
    next.disabled = track.scrollLeft + track.clientWidth >= track.scrollWidth - 2;
  };
  controls.querySelectorAll('button').forEach((button) => {
    button.addEventListener('click', () => {
      const step = track.firstElementChild.getBoundingClientRect().width + parseFloat(getComputedStyle(track).columnGap);
      track.scrollBy({
        left: Number(button.dataset.direction) * step,
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
      });
    });
  });
  track.addEventListener('scroll', updateControls, { passive: true });
  window.addEventListener('resize', updateControls, { passive: true });
  updateControls();
});

// Load clips near the viewport; native controls keep playback in the reader's hands.
const videos = document.querySelectorAll('video');
if ('IntersectionObserver' in window) {
  const loadObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(({ target, isIntersecting }) => {
      if (!isIntersecting) return;
      target.preload = 'metadata';
      observer.unobserve(target);
    });
  }, { rootMargin: '200px' });
  videos.forEach((video) => loadObserver.observe(video));
}

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.hidden = false;
  button.addEventListener('click', async () => {
    const code = document.getElementById(button.dataset.copy);
    const status = document.querySelector('.copy-status');
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      status.textContent = 'BibTeX copied to clipboard.';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
    }
  });
});
