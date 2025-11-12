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

The app now has camera permissions declared in the manifest. To actually use the camera, you'll need to add camera functionality using the browser's MediaDevices API:

```javascript
// Example camera access code
async function requestCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { facingMode: 'environment' } // Use back camera on mobile
    });
    // Use the stream for camera functionality
  } catch (error) {
    console.error('Camera access denied:', error);
  }
}
```

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

## App Store Submission (Optional)

You **don't need** app store approval to use the PWA, but you can optionally submit it:

### iOS App Store:
- Use tools like [PWA Builder](https://www.pwabuilder.com/) or [Capacitor](https://capacitorjs.com/)
- Convert PWA to native iOS app
- Submit through Apple App Store Connect
- **Cost:** $99/year developer account

### Google Play Store:
- Use [Trusted Web Activity (TWA)](https://developer.chrome.com/docs/android/trusted-web-activity/)
- Or use [Bubblewrap](https://github.com/GoogleChromeLabs/bubblewrap)
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

