(() => {
  const storageKey = "sutellite-language";
  const changeEvent = "sutellite-language-change";
  const locales = { en: "en", es: "es-ES", eu: "eu-ES" };
  const months = {
    en: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    es: ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"],
    eu: ["Urtarrila", "Otsaila", "Martxoa", "Apirila", "Maiatza", "Ekaina", "Uztaila", "Abuztua", "Iraila", "Urria", "Azaroa", "Abendua"],
  };
  const root = document.documentElement;
  const originals = new WeakMap();
  const attributes = new WeakMap();
  const messages = new WeakMap();
  let language = readLanguage();
  root.lang = language;

  function readLanguage() {
    try {
      const saved = localStorage.getItem(storageKey);
      if (Object.hasOwn(locales, saved)) return saved;
    } catch {}
    return "en";
  }

  function t(key, parameters = {}) {
    const entry = Object.hasOwn(sutelliteTranslations, key) ? sutelliteTranslations[key] : null;
    return String(entry?.[language] ?? key ?? "").replace(/\{(\w+)\}/g, (token, name) => parameters[name] ?? token);
  }

  function numberFormat(options) {
    const formatters = Object.fromEntries(Object.entries(locales).map(([code, locale]) =>
      [code, new Intl.NumberFormat(code === "eu" ? [locale, locales.es] : locale, options)]));
    return { format: value => formatters[language].format(value) };
  }

  function monthName(month, style = "long") {
    const label = months[language][month - 1];
    return style === "short" ? label.slice(0, 3) : label;
  }

  function periodName(year, month) {
    if (!month) return String(year);
    const label = monthName(month);
    return t("{month} {year}", { year, month: label, lowerMonth: label.toLocaleLowerCase(locales[language]) });
  }

  function text(element, key, parameters) {
    element.setAttribute("data-i18n", "");
    messages.set(element, { key, parameters });
    element.textContent = t(key, parameters);
  }

  function localizePage() {
    for (const element of document.querySelectorAll("[data-i18n]")) {
      const message = messages.get(element);
      if (message) {
        element.textContent = t(message.key, message.parameters);
        continue;
      }
      for (const node of element.childNodes) {
        if (node.nodeType !== Node.TEXT_NODE || !node.data.trim()) continue;
        if (!originals.has(node)) originals.set(node, node.data);
        node.data = originals.get(node).replace(/\S(?:[\s\S]*\S)?/, key => t(key));
      }
    }
    for (const element of document.querySelectorAll("[data-i18n-attrs]")) {
      if (!attributes.has(element)) {
        attributes.set(element, element.dataset.i18nAttrs.split(" ").map(name => [name, element.getAttribute(name)]));
      }
      for (const [name, value] of attributes.get(element)) element.setAttribute(name, t(value));
    }
    for (const button of document.querySelectorAll("[data-language]")) {
      button.setAttribute("aria-pressed", String(button.dataset.language === language));
    }
  }

  function setLanguage(code, persist = true) {
    if (!Object.hasOwn(locales, code) || code === language) return;
    language = code;
    root.lang = language;
    if (persist) {
      try { localStorage.setItem(storageKey, language); } catch {}
    }
    localizePage();
    document.dispatchEvent(new CustomEvent(changeEvent, { detail: language }));
  }

  window.sutelliteI18n = { t, text, monthName, periodName, numberFormat, setLanguage, get language() { return language; } };
  window.addEventListener("storage", event => {
    if (event.key === storageKey || event.key === null) setLanguage(readLanguage(), false);
  });
  document.addEventListener("DOMContentLoaded", () => {
    localizePage();
    for (const button of document.querySelectorAll("[data-language]")) {
      button.addEventListener("click", () => setLanguage(button.dataset.language));
    }
  });
})();
