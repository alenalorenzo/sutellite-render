(() => {
  const storageKey = "sutellite-accessibility";
  const changeEvent = "sutellite-accessibility-change";
  const root = document.documentElement;
  let preferences;

  function readPreferences() {
    let saved;
    try {
      saved = JSON.parse(localStorage.getItem(storageKey));
    } catch {}
    return { colorVision: saved?.colorVision === true, dyslexia: saved?.dyslexia === true };
  }

  function apply(next) {
    if (preferences?.colorVision === next.colorVision && preferences?.dyslexia === next.dyslexia) return;
    preferences = next;
    root.toggleAttribute("data-color-vision", next.colorVision);
    root.toggleAttribute("data-dyslexia", next.dyslexia);
    document.dispatchEvent(new CustomEvent(changeEvent, { detail: next }));
  }

  // The embedded map can also receive preferences when browser storage is disabled.
  window.setSutelliteAccessibility = apply;
  apply(readPreferences());
  window.addEventListener("storage", event => {
    if (event.key === storageKey || event.key === null) {
      apply(readPreferences());
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    const menu = document.getElementById("accessibility-menu");
    if (!menu) return;
    const summary = menu.querySelector("summary");
    const inputs = menu.querySelectorAll('input[type="checkbox"]');
    const syncInputs = () => {
      for (const input of inputs) input.checked = preferences[input.value];
    };
    syncInputs();
    document.addEventListener(changeEvent, syncInputs);

    for (const input of inputs) {
      input.addEventListener("change", () => {
        const next = { ...preferences, [input.value]: input.checked };
        apply(next);
        try {
          localStorage.setItem(storageKey, JSON.stringify(next));
        } catch {}
      });
    }
    document.addEventListener("keydown", event => {
      if (event.key !== "Escape" || !menu.open) return;
      event.preventDefault();
      menu.open = false;
      summary.focus();
    });
    document.addEventListener("click", event => {
      if (menu.open && !menu.contains(event.target)) menu.open = false;
    });
    document.addEventListener("focusin", event => {
      if (menu.open && !menu.contains(event.target)) menu.open = false;
    });
  });
})();
