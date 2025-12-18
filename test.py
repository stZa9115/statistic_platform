import time
import os
RESULT_DIR = "results"
# EXPIRE_SECONDS = 60 * 60  
EXPIRE_SECONDS = 60
now = time.time()

for fname in os.listdir(RESULT_DIR):
    path = os.path.join(RESULT_DIR, fname)
    if not os.path.isfile(path):
        continue
    print(fname)
    print(os.path.getmtime(path))
    print(now - os.path.getmtime(path))

    if now - os.path.getmtime(path) > EXPIRE_SECONDS:
        try:
            os.remove(path)
            print(f"[CLEANUP] removed {fname}")
        except Exception as e:
            print("[CLEANUP ERROR]", e)