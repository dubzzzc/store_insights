# PyInstaller runtime hook for MySQL connector plugins
# This ensures MySQL plugins are accessible in the bundled executable

import sys
import os

# In PyInstaller onefile builds, extracted files are in sys._MEIPASS
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Ensure mysql package is accessible from sys._MEIPASS
    mysql_path = os.path.join(sys._MEIPASS, 'mysql')
    if os.path.exists(mysql_path):
        # Add _MEIPASS to sys.path so mysql package can be imported
        if sys._MEIPASS not in sys.path:
            sys.path.insert(0, sys._MEIPASS)
        
        # Verify plugins directory exists
        plugins_path = os.path.join(sys._MEIPASS, 'mysql', 'connector', 'plugins')
        if not os.path.exists(plugins_path):
            # Try to find it
            import glob
            found_plugins = glob.glob(os.path.join(sys._MEIPASS, '**', 'plugins'), recursive=True)
            if found_plugins:
                plugins_path = found_plugins[0]
