// Force Dark Theme and Remove Theme Toggle Button
(function () {
  "use strict";

  // Force dark mode immediately
  document.documentElement.setAttribute("data-theme", "dark");
  document.body.setAttribute("data-theme", "dark");

  // Remove "Welcome, username" text from user-tools
  function removeWelcomeText() {
    const userTools = document.getElementById("user-tools");
    if (userTools) {
      // Remove text nodes that contain "Welcome" or "خوش آمدید"
      Array.from(userTools.childNodes).forEach((node) => {
        if (node.nodeType === Node.TEXT_NODE) {
          const text = node.textContent.trim();
          if (
            text.includes("Welcome") ||
            text.includes("خوش آمدید") ||
            text.includes("،") ||
            text.includes(",") ||
            text.length < 3 // Remove short text nodes (separators)
          ) {
            node.remove();
          }
        }
      });

      // Also remove any strong or username elements
      const unwantedElements = userTools.querySelectorAll(
        "strong, .username, [class*='welcome']"
      );
      unwantedElements.forEach((el) => el.remove());
    }
  }

  // Remove theme toggle button when DOM is ready
  function removeThemeToggle() {
    const selectors = [
      ".theme-toggle",
      "#theme-toggle",
      'button[class*="theme"]',
      'a[class*="theme-toggle"]',
      "[data-theme-toggle]",
      'button[title*="theme"]',
      'button[title*="Theme"]',
      'a[href*="theme"]',
    ];

    selectors.forEach((selector) => {
      const elements = document.querySelectorAll(selector);
      elements.forEach((el) => {
        el.remove();
      });
    });
  }

  // Run when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      removeThemeToggle();
      removeWelcomeText();
    });
  } else {
    removeThemeToggle();
    removeWelcomeText();
  }

  // Watch for dynamically added theme toggles and welcome text
  const observer = new MutationObserver(() => {
    removeThemeToggle();
    removeWelcomeText();
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Prevent theme changes via localStorage
  const originalSetItem = localStorage.setItem;
  localStorage.setItem = function (key, value) {
    if (key.includes("theme") || key.includes("color-scheme")) {
      value = "dark";
    }
    originalSetItem.call(this, key, value);
  };

  // Set dark theme in localStorage
  localStorage.setItem("django-admin-theme", "dark");
  localStorage.setItem("theme", "dark");
})();
