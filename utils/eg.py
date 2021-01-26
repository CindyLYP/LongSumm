import logging
import os
import re

pattern = re.compile(r'[\[].*\]]')
res = pattern.findall("[sdaf], [dsfa],sdafas,dsaf,[sd],sd")
print(res)

d = "adf\nsadf"
print(d)
print(d.replace('\n',' '))

a = ["dsaf", 'ggg', 'errer','12334.sa,','sd54ui']
b = sum(a)
print(b)