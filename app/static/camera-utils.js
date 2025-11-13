/**
 * Camera utility for Capacitor and browser compatibility
 * Works with static HTML files using script tags
 * 
 * Usage:
 *   <script src="/camera-utils.js"></script>
 *   <script>
 *     CameraUtils.takePicture({ quality: 90 }).then(image => {
 *       console.log('Image:', image);
 *     });
 *   </script>
 */

(function() {
  'use strict';

  // Check if Capacitor is available (will be available in native app)
  function isCapacitorAvailable() {
    return typeof window !== 'undefined' && 
           window.Capacitor && 
           window.Capacitor.isNativePlatform && 
           window.Capacitor.isNativePlatform();
  }

  // Check if running in native app
  function isNativePlatform() {
    if (!isCapacitorAvailable()) return false;
    try {
      return window.Capacitor.getPlatform() !== 'web';
    } catch (e) {
      return false;
    }
  }

  /**
   * Take a picture using native camera or browser API
   * @param {Object} options - Camera options
   * @param {number} options.quality - Image quality (0-100, default: 90)
   * @param {boolean} options.allowEditing - Allow editing before capture (default: false)
   * @param {string} options.resultType - 'base64', 'dataUrl', or 'uri' (default: 'base64')
   * @param {string} options.source - 'camera' or 'photos' (default: 'camera')
   * @returns {Promise<Object>} Image data
   */
  async function takePicture(options) {
    options = options || {};
    const quality = options.quality || 90;
    const allowEditing = options.allowEditing || false;
    const resultType = options.resultType || 'base64';
    const source = options.source || 'camera';

    try {
      if (isNativePlatform() && window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.Camera) {
        // Use Capacitor Camera API for native platforms
        const Camera = window.Capacitor.Plugins.Camera;
        const CameraSource = window.Capacitor.Plugins.CameraSource || {
          Camera: 'CAMERA',
          Photos: 'PHOTOS'
        };
        
        const image = await Camera.getPhoto({
          quality: quality,
          allowEditing: allowEditing,
          resultType: resultType,
          source: source === 'camera' ? CameraSource.Camera : CameraSource.Photos
        });

        return {
          base64: image.base64String,
          dataUrl: image.dataUrl,
          path: image.path,
          webPath: image.webPath,
          format: image.format,
          width: image.width,
          height: image.height
        };
      } else {
        // Fall back to browser API
        return await takePictureBrowser({ quality: quality });
      }
    } catch (error) {
      console.error('Camera error:', error);
      throw new Error('Failed to take picture: ' + (error.message || String(error)));
    }
  }

  /**
   * Take picture using browser API
   * @param {Object} options - Camera options
   * @returns {Promise<Object>} Image data as base64
   */
  function takePictureBrowser(options) {
    options = options || {};
    const quality = options.quality || 90;
    
    return new Promise((resolve, reject) => {
      // Create video element
      const video = document.createElement('video');
      video.autoplay = true;
      video.playsinline = true;
      video.style.position = 'fixed';
      video.style.top = '0';
      video.style.left = '0';
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.zIndex = '9999';
      video.style.objectFit = 'cover';
      
      // Create canvas element
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // Create capture button
      const captureBtn = document.createElement('button');
      captureBtn.textContent = 'Capture';
      captureBtn.style.position = 'fixed';
      captureBtn.style.bottom = '20px';
      captureBtn.style.left = '50%';
      captureBtn.style.transform = 'translateX(-50%)';
      captureBtn.style.zIndex = '10000';
      captureBtn.style.padding = '12px 24px';
      captureBtn.style.fontSize = '16px';
      captureBtn.style.backgroundColor = '#0ea5e9';
      captureBtn.style.color = 'white';
      captureBtn.style.border = 'none';
      captureBtn.style.borderRadius = '8px';
      captureBtn.style.cursor = 'pointer';
      
      // Create cancel button
      const cancelBtn = document.createElement('button');
      cancelBtn.textContent = 'Cancel';
      cancelBtn.style.position = 'fixed';
      cancelBtn.style.bottom = '20px';
      cancelBtn.style.right = '20px';
      cancelBtn.style.zIndex = '10000';
      cancelBtn.style.padding = '12px 24px';
      cancelBtn.style.fontSize = '16px';
      cancelBtn.style.backgroundColor = '#dc2626';
      cancelBtn.style.color = 'white';
      cancelBtn.style.border = 'none';
      cancelBtn.style.borderRadius = '8px';
      cancelBtn.style.cursor = 'pointer';
      
      const cleanup = () => {
        if (video.srcObject) {
          video.srcObject.getTracks().forEach(track => track.stop());
        }
        video.remove();
        captureBtn.remove();
        cancelBtn.remove();
      };
      
      captureBtn.onclick = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const base64 = canvas.toDataURL('image/jpeg', quality / 100);
        cleanup();
        resolve({
          base64: base64.split(',')[1],
          dataUrl: base64,
          webPath: base64,
          format: 'jpeg'
        });
      };
      
      cancelBtn.onclick = () => {
        cleanup();
        reject(new Error('Camera capture cancelled'));
      };
      
      // Request camera access
      navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment' // Use back camera on mobile
        } 
      })
      .then(stream => {
        video.srcObject = stream;
        document.body.appendChild(video);
        document.body.appendChild(captureBtn);
        document.body.appendChild(cancelBtn);
        
        video.onloadedmetadata = () => {
          video.play();
        };
      })
      .catch(error => {
        cleanup();
        reject(new Error('Camera access denied: ' + (error.message || String(error))));
      });
    });
  }

  /**
   * Check camera permission status
   * @returns {Promise<string>} 'granted', 'denied', or 'prompt'
   */
  async function checkCameraPermission() {
    if (isNativePlatform() && window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.Camera) {
      try {
        const Camera = window.Capacitor.Plugins.Camera;
        const result = await Camera.checkPermissions();
        return result.camera || 'prompt';
      } catch (error) {
        return 'prompt';
      }
    } else {
      // Browser API
      try {
        if (navigator.permissions && navigator.permissions.query) {
          const result = await navigator.permissions.query({ name: 'camera' });
          return result.state;
        }
        // Fallback: try to request access
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        stream.getTracks().forEach(track => track.stop());
        return 'granted';
      } catch (e) {
        return 'denied';
      }
    }
  }

  /**
   * Request camera permission
   * @returns {Promise<boolean>} True if granted, false otherwise
   */
  async function requestCameraPermission() {
    if (isNativePlatform() && window.Capacitor && window.Capacitor.Plugins && window.Capacitor.Plugins.Camera) {
      try {
        const Camera = window.Capacitor.Plugins.Camera;
        const result = await Camera.requestPermissions({ permissions: ['camera'] });
        return result.camera === 'granted';
      } catch (error) {
        return false;
      }
    } else {
      // Browser API - permission is requested when getUserMedia is called
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        stream.getTracks().forEach(track => track.stop());
        return true;
      } catch (error) {
        return false;
      }
    }
  }

  // Export functions for use in HTML files
  if (typeof window !== 'undefined') {
    window.CameraUtils = {
      takePicture: takePicture,
      checkCameraPermission: checkCameraPermission,
      requestCameraPermission: requestCameraPermission,
      isNativePlatform: isNativePlatform,
      isCapacitorAvailable: isCapacitorAvailable
    };
  }
})();
