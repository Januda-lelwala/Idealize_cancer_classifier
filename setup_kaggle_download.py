import os
import sys
import subprocess
import shutil
import platform

# Helper to run shell commands
def run(cmd, check=True, shell=True):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell)
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)

# 1. Install kaggle
run("pip install kaggle")

# 2. Install unzip if on Linux (skip on Mac/Windows)
if platform.system() == "Linux":
    run("apt update && apt install unzip -y")
else:
    print("Skipping 'apt update' and 'apt install unzip -y' (not a Linux system)")

# 3. Move kaggle.json to ~/.kaggle
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

if os.path.exists("kaggle.json"):
    shutil.move("kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))
    print("Moved kaggle.json to ~/.kaggle/")
else:
    print("kaggle.json not found in current directory.")
    sys.exit(1)

os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

# 4. Download the competition data
run("kaggle competitions download -c idealize-2025-datathon-competition")
