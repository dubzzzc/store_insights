# Mobile App Deployment Guide

This guide covers deploying the Store Insights app as native iOS and Android apps using Capacitor.

## Overview

The app has been configured with Capacitor to wrap the existing web app as native iOS and Android applications. This allows you to:

- Submit to Apple App Store and Google Play Store
- Access native device features (camera, etc.)
- Provide a native app experience
- Maintain a single codebase (web + mobile)

## Prerequisites

### For iOS Development:
- macOS (required for Xcode)
- Xcode (latest version)
- Apple Developer Account ($99/year)
- CocoaPods: `sudo gem install cocoapods`

### For Android Development:
- Android Studio (latest version)
- Java Development Kit (JDK)
- Android SDK
- Google Play Developer Account ($25 one-time)

## Project Structure

```
store_insights/
├── app/
│   └── static/          # Web assets (HTML, JS, CSS)
│       ├── index.html   # Entry point for Capacitor
│       ├── camera-utils.js  # Camera utility for native/browser
│       └── ...
├── ios/                 # iOS native project (Xcode)
├── android/             # Android native project (Android Studio)
├── capacitor.config.ts  # Capacitor configuration
└── package.json         # NPM dependencies and scripts
```

## Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Sync web assets to native projects:**
   ```bash
   npm run cap:sync
   ```

   This copies your web files from `app/static` to the native projects.

## Development Workflow

### Making Changes

1. Edit files in `app/static/` (HTML, JS, CSS)
2. Run `npm run cap:sync` to copy changes to native projects
3. Test in native IDEs or on devices

### Opening Native Projects

**iOS (macOS only):**
```bash
npm run cap:ios
```
This opens the project in Xcode.

**Android:**
```bash
npm run cap:android
```
This opens the project in Android Studio.

## Camera Integration

The app includes camera functionality that works on both native apps and web browsers.

### Using the Camera

Include the camera utility script in your HTML:

```html
<script src="/camera-utils.js"></script>
<script>
  // Take a picture
  CameraUtils.takePicture({ quality: 90 })
    .then(image => {
      console.log('Image base64:', image.base64);
      console.log('Image data URL:', image.dataUrl);
      // Use the image data
    })
    .catch(error => {
      console.error('Camera error:', error);
    });

  // Check permission
  CameraUtils.checkCameraPermission()
    .then(status => {
      console.log('Camera permission:', status);
    });

  // Request permission
  CameraUtils.requestCameraPermission()
    .then(granted => {
      if (granted) {
        console.log('Camera permission granted');
      }
    });
</script>
```

The camera utility automatically:
- Uses native camera API on iOS/Android devices
- Falls back to browser API on web
- Handles permissions gracefully

## Building for Production

### iOS

1. Open project in Xcode:
   ```bash
   npm run cap:ios
   ```

2. In Xcode:
   - Select your development team
   - Choose a device or simulator
   - Product → Archive
   - Follow App Store submission process

3. **Requirements:**
   - App icons (all required sizes)
   - Launch screen configured
   - Privacy descriptions in Info.plist
   - Code signing configured

### Android

1. Open project in Android Studio:
   ```bash
   npm run cap:android
   ```

2. In Android Studio:
   - Build → Generate Signed Bundle / APK
   - Select "Android App Bundle"
   - Use your release keystore
   - Build the release bundle

3. **Requirements:**
   - Adaptive icons created
   - Splash screen configured
   - Permissions declared in AndroidManifest.xml
   - Release keystore configured

## App Store Submission

### Apple App Store

1. **Prepare assets:**
   - App icons (1024x1024 for App Store)
   - Screenshots for required device sizes:
     - 6.5" iPhone (iPhone 14 Pro Max, etc.)
     - 5.5" iPhone (iPhone 8 Plus, etc.)
     - iPad Pro 12.9"
   - App preview videos (optional)

2. **App Store Connect:**
   - Create new app
   - Fill in metadata (name, description, keywords)
   - Upload screenshots
   - Set pricing and availability
   - Complete age rating questionnaire
   - Add privacy policy URL

3. **Submit for review:**
   - Archive build in Xcode
   - Upload to App Store Connect
   - Submit for review

### Google Play Store

1. **Prepare assets:**
   - App icon (512x512)
   - Feature graphic (1024x500)
   - Screenshots for phone and tablets

2. **Play Console:**
   - Create new app
   - Fill in store listing
   - Upload AAB file
   - Complete data safety section
   - Complete content rating

3. **Submit for review:**
   - Upload signed AAB
   - Complete all required sections
   - Submit for review

## Configuration

### Capacitor Config

Edit `capacitor.config.ts`:

```typescript
import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.spirits.storeinsights',
  appName: 'Spirits Store Insights',
  webDir: 'app/static',
  server: {
    // For development, uncomment to use local server:
    // url: 'http://localhost:8000',
    // cleartext: true
  }
};

export default config;
```

### CORS Configuration

The FastAPI backend is configured to allow requests from mobile apps. Update `app/main.py` if you need to add additional origins:

```python
allow_origins=[
    "capacitor://localhost",  # iOS/Android native app
    "http://localhost",       # Local testing
    "http://localhost:8000", # Local FastAPI server
    "https://your-api-domain.com"  # Production API domain
]
```

## Troubleshooting

### iOS Issues

- **"CocoaPods not installed"**: Run `sudo gem install cocoapods`
- **Build errors**: Clean build folder in Xcode (Product → Clean Build Folder)
- **Code signing**: Ensure your Apple Developer account is configured in Xcode

### Android Issues

- **Gradle sync errors**: In Android Studio, File → Sync Project with Gradle Files
- **Build errors**: Clean project (Build → Clean Project)
- **Keystore issues**: Ensure release keystore is properly configured

### General Issues

- **Changes not appearing**: Run `npm run cap:sync` after making changes
- **Camera not working**: Check permissions in Info.plist (iOS) or AndroidManifest.xml (Android)
- **API calls failing**: Verify CORS configuration in `app/main.py`

## Testing

### On Physical Devices

**iOS:**
1. Connect iPhone/iPad via USB
2. Select device in Xcode
3. Click Run (or press Cmd+R)

**Android:**
1. Enable USB debugging on device
2. Connect via USB
3. Select device in Android Studio
4. Click Run

### Testing Checklist

- [ ] App launches successfully
- [ ] Camera functionality works
- [ ] API calls succeed
- [ ] Authentication flow works
- [ ] Offline mode works (if applicable)
- [ ] All major features functional

## Resources

- [Capacitor Documentation](https://capacitorjs.com/docs)
- [Apple App Store Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [Google Play Policies](https://play.google.com/about/developer-content-policy/)
- [Capacitor Camera Plugin](https://capacitorjs.com/docs/apis/camera)

## Support

For issues or questions:
1. Check Capacitor documentation
2. Review platform-specific guides (iOS/Android)
3. Check app store review guidelines

