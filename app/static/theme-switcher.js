/**
 * Theme Switcher Utility
 * Manages theme mode switching (Quilt, Liquor, Grocery) and dark mode
 */

const ThemeSwitcher = {
  // Available themes
  themes: {
    quilt: 'Quilt Default',
    liquor: 'Liquor',
    grocery: 'Grocery',
  },
  
  // Get current theme from localStorage or default to 'quilt'
  getCurrentTheme() {
    return localStorage.getItem('store-insights-theme') || 'quilt';
  },
  
  // Set theme
  setTheme(themeName) {
    if (!this.themes[themeName]) {
      console.warn(`Theme "${themeName}" not found. Using default.`);
      themeName = 'quilt';
    }
    
    document.documentElement.setAttribute('data-theme', themeName);
    localStorage.setItem('store-insights-theme', themeName);
    
    // Dispatch custom event for components to react
    window.dispatchEvent(new CustomEvent('themechange', { 
      detail: { theme: themeName } 
    }));
  },
  
  // Get current dark mode state
  isDarkMode() {
    return document.documentElement.classList.contains('dark');
  },
  
  // Toggle dark mode
  toggleDarkMode() {
    const isDark = this.isDarkMode();
    if (isDark) {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('store-insights-dark-mode', 'false');
    } else {
      document.documentElement.classList.add('dark');
      localStorage.setItem('store-insights-dark-mode', 'true');
    }
    
    // Dispatch custom event
    window.dispatchEvent(new CustomEvent('darkmodechange', { 
      detail: { isDark: !isDark } 
    }));
  },
  
  // Set dark mode
  setDarkMode(enabled) {
    if (enabled) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('store-insights-dark-mode', 'true');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('store-insights-dark-mode', 'false');
    }
  },
  
  // Initialize theme on page load
  init() {
    // Set theme
    const savedTheme = this.getCurrentTheme();
    this.setTheme(savedTheme);
    
    // Set dark mode
    const savedDarkMode = localStorage.getItem('store-insights-dark-mode');
    if (savedDarkMode === 'true') {
      this.setDarkMode(true);
    }
  },
  
  // ThemeSelector component is defined in HTML files where Babel can process JSX
  // This utility just provides the theme switching logic
};

// Auto-initialize on load
if (typeof window !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ThemeSwitcher.init());
  } else {
    ThemeSwitcher.init();
  }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ThemeSwitcher;
}

