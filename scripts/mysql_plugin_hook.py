# PyInstaller runtime hook for MySQL connector plugins
# This ensures MySQL plugins are accessible in the bundled executable
# This hook runs BEFORE the main script, so plugins are registered early

import sys
import os

# In PyInstaller onefile builds, extracted files are in sys._MEIPASS
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Add _MEIPASS to sys.path FIRST so all imports work correctly
    if sys._MEIPASS not in sys.path:
        sys.path.insert(0, sys._MEIPASS)
    
    # Ensure mysql package is accessible from sys._MEIPASS
    mysql_path = os.path.join(sys._MEIPASS, 'mysql')
    if os.path.exists(mysql_path):
        # Verify plugins directory exists - try multiple possible locations
        plugins_paths = [
            os.path.join(sys._MEIPASS, 'mysql', 'connector', 'plugins'),
            os.path.join(sys._MEIPASS, 'mysql', 'connector', 'plugins'),
        ]
        
        # Also search for plugins directory
        import glob
        found_plugins = glob.glob(os.path.join(sys._MEIPASS, '**', 'plugins'), recursive=True)
        plugins_paths.extend(found_plugins)
        
        plugins_path = None
        for path in plugins_paths:
            if os.path.exists(path) and os.path.isdir(path):
                plugins_path = path
                break
        
        # CRITICAL: Pre-import and register all plugin modules BEFORE any mysql.connector usage
        # This ensures the plugins are registered in the plugin registry
        # The plugins package __init__.py automatically registers plugins when imported
        try:
            # Import mysql.connector first to ensure the package structure is available
            import mysql.connector
            # Import the plugins package __init__ to trigger plugin registration
            import mysql.connector.plugins
            # Now explicitly import each plugin to ensure they're loaded and registered
            plugin_modules = [
                'mysql.connector.plugins.mysql_native_password',
                'mysql.connector.plugins.caching_sha2_password',
                'mysql.connector.plugins.sha256_password',
                'mysql.connector.plugins.mysql_clear_password',
            ]
            for plugin_module in plugin_modules:
                try:
                    __import__(plugin_module)
                except (ImportError, ModuleNotFoundError):
                    # Plugin not available - continue
                    pass
                except Exception:
                    # Other errors - continue
                    pass
        except ImportError:
            # mysql.connector not available - skip plugin registration
            pass
