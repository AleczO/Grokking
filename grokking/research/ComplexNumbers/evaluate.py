import os
import subprocess
import re

def get_epoch(name):
    nums = re.findall(r'\d+', name)
    return int(nums[0]) if nums else 0

ckpt_dir = "experiments/run_L2_D32_20260522_151145/checkpoints/"
checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
checkpoints.sort(key=get_epoch)

print(f"Rozpoczynam bezpieczną ewaluację {len(checkpoints)} plików...")

for ckpt in checkpoints:
    ckpt_path = os.path.join(ckpt_dir, ckpt)
    
    try:
        result = subprocess.run(
            ["python", "single_eval.py", "--path", ckpt_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"CRASH przy {ckpt}: {result.stderr}")
            
    except Exception as e:
        print(f"Błąd systemowy przy {ckpt}: {e}")