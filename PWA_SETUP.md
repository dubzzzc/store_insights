# PWA Setup Guide for Spirits Store Insights

## Overview
This app has been configured as a Progressive Web App (PWA) that can:
- ✅ Access camera on iOS/Android devices
- ✅ Be installed on home screen (looks like native app)
- ✅ Work offline (basic functionality)
- ✅ **NO app store approval required** (but can optionally be submitted)

## What's Been Added

### 1. Manifest.json
- Defines app metadata, icons, and permissions
- Located at: `app/static/manifest.json`

### 2. Service Worker
- Enables offline functionality and caching
- Located at: `app/static/service-worker.js`

### 3. HTML Updates
- Added PWA meta tags for iOS
- Linked manifest.json
- Registered service worker

## Camera Access

The app includes a camera utility (`camera-utils.js`) that works on both native apps and web browsers. It automatically uses the native camera API on iOS/Android devices and falls back to the browser API on web.

**Usage:**
```html
<script src="/camera-utils.js"></script>
<script>
  // Take a picture
  CameraUtils.takePicture({ quality: 90 })
    .then(image => {
      console.log('Image:', image.base64);
      // Use the image data
    })
    .catch(error => {
      console.error('Camera error:', error);
    });
</script>
```

The camera utility handles:
- Native camera on iOS/Android (via Capacitor)
- Browser camera on web (via MediaDevices API)
- Permission requests
- Error handling

## Icon Requirements

You need to create two icon files:
- `app/static/icon-192.png` (192x192 pixels)
- `app/static/icon-512.png` (512x512 pixels)

These icons will appear:
- On the home screen when installed
- In app switcher
- In app stores (if submitted)

**Icon Design Tips:**
- Use a simple, recognizable logo
- Ensure it looks good at small sizes
- Use solid colors or high contrast
- Test on both light and dark backgrounds

## Installation Instructions

### For Users (iOS):
1. Open the app in Safari
2. Tap the Share button
3. Select "Add to Home Screen"
4. The app will appear as a native app icon

### For Users (Android):
1. Open the app in Chrome
2. A banner will appear: "Add Spirits Store Insights to Home screen"
3. Tap "Add" or use the menu → "Add to Home screen"

## App Store Submission

### Option 1: PWA (No App Store Required)

You **don't need** app store approval to use the PWA. Users can install it directly from the browser.

### Option 2: Native Apps (App Store Submission)

The app is now configured with **Capacitor** for native iOS and Android deployment. See [MOBILE_APP_DEPLOYMENT.md](MOBILE_APP_DEPLOYMENT.md) for complete instructions.

**Quick Start:**
```bash
npm install
npm run cap:sync
npm run cap:ios      # Opens Xcode (macOS only)
npm run cap:android  # Opens Android Studio
```

**iOS App Store:**
- Native app via Capacitor
- Submit through Apple App Store Connect
- **Cost:** $99/year developer account

**Google Play Store:**
- Native app via Capacitor
- Submit through Google Play Console
- **Cost:** $25 one-time fee

## Testing

1. **Test on HTTPS:** PWAs require HTTPS (except localhost)
2. **Test installation:** Try installing on iOS and Android devices
3. **Test camera:** Verify camera permissions work
4. **Test offline:** Disable network and verify basic functionality

## Next Steps

1. **Create icons:** Generate the 192x192 and 512x512 PNG icons
2. **Add camera functionality:** Implement camera features in your app
3. **Test installation:** Test on real iOS/Android devices
4. **Optional:** Submit to app stores if desired

## Important Notes

- **HTTPS Required:**
  - PWAs only work over HTTPS (or localhost)
  - Your production server must have SSL certificate
  - Camera access also requires HTTPS

- **iOS Limitations:**
  - iOS Safari has some PWA limitations
  - Camera access works but may have restrictions
  - Some features work better in standalone mode

- **Android:**
  - Full PWA support
  - Camera access works well
  - Can be submitted to Play Store as TWA

## Resources

- [MDN: Progressive Web Apps](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
- [Web.dev: PWAs](https://web.dev/progressive-web-apps/)
- [PWA Builder](https://www.pwabuilder.com/) - Convert to native apps
- [Capacitor](https://capacitorjs.com/) - Native app framework

