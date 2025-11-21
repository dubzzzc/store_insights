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
  
  // Create a theme selector component (React)
  ThemeSelector({ currentTheme, onThemeChange, showDarkMode = true }) {
    const [theme, setTheme] = React.useState(currentTheme || ThemeSwitcher.getCurrentTheme());
    const [isDark, setIsDark] = React.useState(ThemeSwitcher.isDarkMode());
    
    React.useEffect(() => {
      const handleThemeChange = (e) => {
        setTheme(e.detail.theme);
        if (onThemeChange) onThemeChange(e.detail.theme);
      };
      
      const handleDarkModeChange = (e) => {
        setIsDark(e.detail.isDark);
      };
      
      window.addEventListener('themechange', handleThemeChange);
      window.addEventListener('darkmodechange', handleDarkModeChange);
      
      return () => {
        window.removeEventListener('themechange', handleThemeChange);
        window.removeEventListener('darkmodechange', handleDarkModeChange);
      };
    }, [onThemeChange]);
    
    const handleThemeSelect = (newTheme) => {
      ThemeSwitcher.setTheme(newTheme);
      setTheme(newTheme);
      if (onThemeChange) onThemeChange(newTheme);
    };
    
    const handleDarkModeToggle = () => {
      ThemeSwitcher.toggleDarkMode();
      setIsDark(ThemeSwitcher.isDarkMode());
    };
    
    return (
      <div className="flex items-center gap-4 p-4 border border-border rounded-lg bg-card">
        <div className="flex items-center gap-2">
          <Label>Theme:</Label>
          <Select
            value={theme}
            onChange={(e) => handleThemeSelect(e.target.value)}
            className="w-32"
          >
            {Object.entries(ThemeSwitcher.themes).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </Select>
        </div>
        
        {showDarkMode && (
          <Switch
            checked={isDark}
            onCheckedChange={handleDarkModeToggle}
            label="Dark Mode"
          />
        )}
      </div>
    );
  },
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

