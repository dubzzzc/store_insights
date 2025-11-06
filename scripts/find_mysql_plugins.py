#!/usr/bin/env python3
"""Find MySQL connector plugin files for PyInstaller bundling"""
import os
import sys

try:
    import mysql.connector
    mysql_path = os.path.dirname(mysql.connector.__file__)
    plugins_path = os.path.join(mysql_path, 'plugins')
    
    print(f"MySQL connector path: {mysql_path}")
    print(f"Plugins path: {plugins_path}")
    print(f"Plugins exists: {os.path.exists(plugins_path)}")
    
    if os.path.exists(plugins_path):
        print("\nPlugin files:")
        for item in os.listdir(plugins_path):
            item_path = os.path.join(plugins_path, item)
            if os.path.isfile(item_path):
                print(f"  {item} ({os.path.getsize(item_path)} bytes)")
            elif os.path.isdir(item_path):
                print(f"  {item}/ (directory)")
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isfile(subitem_path):
                        print(f"    {subitem} ({os.path.getsize(subitem_path)} bytes)")
    
    print("\nFor PyInstaller, add to spec file:")
    print(f'    datas=[(r"{plugins_path}", "mysql/connector/plugins")],')
    
except ImportError as e:
    print(f"Error: {e}")
    print("MySQL connector not installed")
    sys.exit(1)

