import logging
import os
import re

pattern = re.compile(r'[\[].*\]]')
res = pattern.findall("[sdaf], [dsfa],sdafas,dsaf,[sd],sd")
print(res)

d = "adf\nsadf"
print(d)
print(d.replace('\n',' '))