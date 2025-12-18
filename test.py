import time
import os
RESULT_DIR = "results"
# EXPIRE_SECONDS = 60 * 60  
EXPIRE_SECONDS = 600
now = time.time()

for fname in os.listdir(RESULT_DIR):
    path = os.path.join(RESULT_DIR, fname)
    if not os.path.isfile(path):
        continue
    print(fname)
    print(os.path.getmtime(path))
    print(now - os.path.getmtime(path))