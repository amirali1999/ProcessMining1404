#!/usr/bin/env python
"""
Force reload admin static files by adding version query parameter
"""
import os
import time

# Add version timestamp to force browser reload
version = int(time.time())

css_file = 'static/admin/css/admin_dark_theme.css'
print(f"CSS file last modified: {os.path.getmtime(css_file)}")
print(f"Version to use in URL: ?v={version}")
print(f"\nAdd this to your browser URL:")
print(f"http://localhost:8000/static/admin/css/admin_dark_theme.css?v={version}")
print(f"\nOr do a HARD REFRESH:")
print(f"- Chrome/Safari: Cmd + Shift + R")
print(f"- Or: Open DevTools (F12) > Network tab > Check 'Disable cache'")
