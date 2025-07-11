import os
import sys
import subprocess
import time

def run_command(cmd, check=True, shell=True):
    """Run a shell command and handle errors"""
    print(f"\n[RUNNING] {cmd}")
    print("-" * 80)
    result = subprocess.run(cmd, shell=shell)
    if check and result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}: {cmd}")
        sys.exit(1)
    print("-" * 80)
    return result

def main():
    # Start the pipeline
    print("\n===== STARTING PIPELINE =====")
    start_time = time.time()
    
    # Step 1: Install requirements
    print("\n[STEP 1] Installing requirements")
    run_command("pip install -r requirements.txt")
    print("✅ Requirements installed successfully")
    
    # Step 2: Download and extract Kaggle data
    print("\n[STEP 2] Downloading and extracting Kaggle data")
    run_command("python setup_kaggle_download.py")
    print("✅ Kaggle data downloaded and extracted successfully")
    
    # Step 3: Process data with CSV oversampler
    print("\n[STEP 3] Processing data with CSV oversampler")
    run_command("python csv_oversampler.py")
    print("✅ CSV oversampling completed successfully")
    
    # Pipeline completed
    elapsed_time = time.time() - start_time
    print(f"\n===== PIPELINE COMPLETED in {elapsed_time:.2f} seconds =====")
    print("All steps executed successfully. You can now proceed with training the model.")

if __name__ == "__main__":
    main()
