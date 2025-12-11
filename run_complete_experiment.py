#!/usr/bin/env python3
"""
Master Experiment Runner for TAEG Project.
This script orchestrates the execution of ALL summarization methods (Extractive & Abstractive)
and finally runs the unified evaluation.

Methods run:
1. LexRank (Extractive Baseline)
2. LexRank-TA (Extractive TAEG Graph-Guided)
3. Pure Abstractive (BART, PEGASUS, PRIMERA) - Global Baseline
4. TAEG Abstractive (BART, PEGASUS, PRIMERA) - TAEG Graph-Guided

Usage:
    python run_complete_experiment.py
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(command, description):
    print("\n" + "="*80)
    print(f"🚀 {description.upper()}")
    print("="*80)
    start_time = time.time()
    
    try:
        # Run python script using the current python executable
        cmd_list = [sys.executable] + command.split()
        result = subprocess.run(cmd_list, check=True)
        
        elapsed = time.time() - start_time
        print(f"✅ Finished {description} in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        sys.exit(1)

def main():
    print("TAEG - FULL EXPERIMENT SUITE")
    print(" This script will run all summarization methods and generate a final comparison.")
    print(" WARNING: This may take a significant amount of time on CPU.")
    
    # Ensure output directory exists
    Path("outputs").mkdir(exist_ok=True)
    
    tasks = [
        # 1. Extractive Methods (Existing)
        ("run_taeg.py --method lexrank", "Extractive Baseline (LexRank)"),
        ("run_taeg.py --method lexrank-ta", "Extractive Guided (LexRank-TA)"),
        
        # 2. Pure Abstractive (Baselines)
        ("run_pure_abstractive.py --method bart", "Pure Abstractive (BART)"),
        ("run_pure_abstractive.py --method pegasus", "Pure Abstractive (PEGASUS)"),
        ("run_pure_abstractive.py --method primera", "Pure Abstractive (PRIMERA)"),
        
        # 3. TAEG Guided Abstractive
        ("run_taeg_abstractive.py --method bart", "TAEG Abstractive (BART)"),
        ("run_taeg_abstractive.py --method pegasus", "TAEG Abstractive (PEGASUS)"),
        ("run_taeg_abstractive.py --method primera", "TAEG Abstractive (PRIMERA)"),
        
        # 4. Evaluation
        ("compare_methods.py", "Final Evaluation & Comparison")
    ]
    
    total_start = time.time()
    success_count = 0
    
    for cmd, desc in tasks:
        if run_command(cmd, desc):
            success_count += 1
        else:
            print(f"⚠️ Skipping subsequent steps that might depend on this...")
            # We continue anyway to try to get as many results as possible
            
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("🎉 EXPERIMENT SUITE COMPLETED")
    print(f"⏱️ Total time: {total_elapsed/60:.2f} minutes")
    print(f"✅ Successful tasks: {success_count}/{len(tasks)}")
    print("="*80)
    print("\nCheck 'outputs/comparison_results.csv' for the final table.")

if __name__ == "__main__":
    main()
