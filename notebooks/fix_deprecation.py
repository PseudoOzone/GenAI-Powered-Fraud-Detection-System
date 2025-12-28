#!/usr/bin/env python3
import re
from pathlib import Path

app_file = Path(__file__).parent / 'app.py'

with open(app_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace use_container_width with width parameter
content = content.replace('use_container_width=True', 'width="stretch"')
content = content.replace('use_container_width=False', 'width="content"')
content = content.replace(', width="stretch"', ', width="stretch"')
content = content.replace(', width="content"', ', width="content"')

with open(app_file, 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Fixed all use_container_width parameters')
