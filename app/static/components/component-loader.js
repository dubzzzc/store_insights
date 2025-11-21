/**
 * Component Library Loader
 * Loads and processes component files with Babel Standalone
 * Uses async loading but sets a flag when complete
 */

(function() {
  'use strict';
  
  window.componentsLoaded = false;
  window.componentsLoading = true;
  
  async function loadComponentFile(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Failed to load ${url}:`, response.status, response.statusText);
        console.error('Response:', errorText.substring(0, 200));
        throw new Error(`Failed to load ${url}: ${response.status} ${response.statusText}`);
      }
      const code = await response.text();
      
      // Check if we got HTML instead of JavaScript (common 404 issue)
      if (code.trim().startsWith('<!DOCTYPE') || code.trim().startsWith('<html')) {
        console.error(`Received HTML instead of JavaScript for ${url}. Check file path.`);
        throw new Error(`Received HTML response for ${url}. File may not exist or path is incorrect.`);
      }
      
      // Process with Babel Standalone
      if (typeof Babel !== 'undefined' && Babel.transform) {
        try {
          const transformed = Babel.transform(code, {
            presets: ['react'],
          }).code;
          
          // Execute the transformed code in global scope
          const script = document.createElement('script');
          script.textContent = transformed;
          document.head.appendChild(script);
          document.head.removeChild(script);
        } catch (babelError) {
          console.error(`Babel transformation error for ${url}:`, babelError);
          throw babelError;
        }
      } else {
        throw new Error('Babel Standalone not found');
      }
    } catch (error) {
      console.error(`Error loading component file ${url}:`, error);
      throw error;
    }
  }
  
  async function loadAllComponents() {
    // Wait for Babel to be available
    let attempts = 0;
    while (typeof Babel === 'undefined' && attempts < 100) {
      await new Promise(resolve => setTimeout(resolve, 20));
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
        await loadComponentFile(file);
      }
      window.componentsLoaded = true;
      window.componentsLoading = false;
      console.log('✓ Component library loaded successfully');
      window.dispatchEvent(new CustomEvent('componentsLoaded'));
    } catch (error) {
      window.componentsLoading = false;
      console.error('✗ Failed to load component library:', error);
      throw error;
    }
  }
  
  // Start loading immediately
  loadAllComponents().catch(err => {
    console.error('Component loader error:', err);
    window.componentsLoaded = false;
    window.componentsLoading = false;
  });
})();

