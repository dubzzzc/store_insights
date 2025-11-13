import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.spirits.storeinsights',
  appName: 'Spirits Store Insights',
  webDir: 'app/static',
  server: {
    // For development, uncomment the line below to use local server
    // url: 'http://localhost:8000',
    // cleartext: true
    // For production, leave server config empty to use bundled files
  }
};

export default config;
