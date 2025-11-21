/**
 * Component Library Loader
 * Fetches and processes component files with Babel Standalone
 * Ensures components are loaded before main scripts execute
 */

(function() {
  'use strict';
  
  const ComponentLoader = {
    loaded: false,
    loading: false,
    loadPromise: null,
    
    async loadComponentFile(url) {
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to load ${url}: ${response.statusText}`);
        }
        const code = await response.text();
        
        // Process with Babel Standalone
        if (typeof Babel !== 'undefined' && Babel.transform) {
          const transformed = Babel.transform(code, {
            presets: ['react'],
          }).code;
          
          // Execute the transformed code in global scope
          const script = document.createElement('script');
          script.textContent = transformed;
          document.head.appendChild(script);
          document.head.removeChild(script);
        } else {
          // Fallback: try to execute directly (might fail if JSX)
          console.warn('Babel not available, attempting direct execution');
          eval(code);
        }
      } catch (error) {
        console.error(`Error loading component file ${url}:`, error);
        throw error;
      }
    },
    
    async loadAll() {
      if (this.loaded) {
        return Promise.resolve();
      }
      
      if (this.loading) {
        return this.loadPromise;
      }
      
      this.loading = true;
      this.loadPromise = (async () => {
        // Wait for Babel to be available
        let attempts = 0;
        while (typeof Babel === 'undefined' && attempts < 50) {
          await new Promise(resolve => setTimeout(resolve, 50));
          attempts++;
        }
        
        if (typeof Babel === 'undefined') {
          throw new Error('Babel Standalone not found. Please ensure @babel/standalone is loaded.');
        }
        
        const componentFiles = [
          '/components/primitives.js',
          '/components/form-components.js',
          '/components/display-components.js',
          '/components/composite-components.js',
        ];
        
        try {
          for (const file of componentFiles) {
            await this.loadComponentFile(file);
          }
          this.loaded = true;
          this.loading = false;
          console.log('✓ Component library loaded successfully');
          
          // Dispatch event so other scripts know components are ready
          window.dispatchEvent(new CustomEvent('componentsLoaded'));
        } catch (error) {
          this.loading = false;
          console.error('✗ Failed to load component library:', error);
          throw error;
        }
      })();
      
      return this.loadPromise;
    },
  };
  
  // Start loading immediately
  ComponentLoader.loadAll().catch(err => {
    console.error('Component loader error:', err);
  });
  
  // Make available globally
  window.ComponentLoader = ComponentLoader;
})();

