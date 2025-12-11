#!/usr/bin/env python3
"""
Unified Comparison Script.
Evaluates all generated summaries (Extractive & Abstractive) against the Golden Sample.
"""

import sys
from pathlib import Path
import pandas as pd
import os

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import BiblicalDataLoader
from evaluator import SummarizationEvaluator

def compare_methods():
    print("TAEG - Unified Method Evaluation")
    print("="*80)
    
    # 1. Load Golden Sample
    loader = BiblicalDataLoader("data")
    try:
        golden_sample = loader.load_golden_sample()
        print(f"Loaded Golden Sample ({len(golden_sample)} chars)")
    except Exception as e:
        print(f"Error loading golden sample: {e}")
        return

    evaluator = SummarizationEvaluator()
    
    # 2. Define files to evaluate
    output_dir = Path("outputs")
    files_to_eval = [
        # Extractive
        ("TAEG-LexRank", output_dir / "taeg_summary_lexrank.txt"),
        ("TAEG-LexRank-TA", output_dir / "taeg_summary_lexrank-ta.txt"),
        
        # Abstractive TAEG
        ("TAEG-BART", output_dir / "taeg_bart.txt"),
        ("TAEG-PEGASUS", output_dir / "taeg_pegasus.txt"),
        ("TAEG-PRIMERA", output_dir / "taeg_primera.txt"),
        ("TAEG-GEMMA", output_dir / "taeg_gemma.txt"),
        
        # Abstractive Pure
        ("Pure-BART", output_dir / "pure_bart.txt"),
        ("Pure-PEGASUS", output_dir / "pure_pegasus.txt"),
        ("Pure-PRIMERA", output_dir / "pure_primera.txt"),
        ("Pure-GEMMA", output_dir / "pure_gemma.txt"),
    ]
    
    results = []
    
    for method_name, file_path in files_to_eval:
        if not file_path.exists():
            print(f"⚠️ Skipping {method_name}: File not found ({file_path})")
            continue
            
        print(f"\nEvaluating {method_name}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                summary_text = f.read()
                
            if not summary_text:
                print("  Empty file")
                continue
                
            # Determine if method is time-aware (TAEG or LexRank-TA)
            # This affects Kendall's Tau calculation (needs chronological chunks)
            is_temporal = "TAEG" in method_name or "TA" in method_name
            
            metrics = evaluator.evaluate_summary(summary_text, golden_sample, is_temporal_anchored=is_temporal)
            
            # Flatten metrics for table
            res = {
                "Method": method_name,
                "Length (chars)": len(summary_text),
                "ROUGE-1": metrics["rouge"]["rouge1"]["f1"],
                "ROUGE-2": metrics["rouge"]["rouge2"]["f1"],
                "ROUGE-L": metrics["rouge"]["rougeL"]["f1"],
                "BERTScore": metrics["bertscore"]["f1"],
                "METEOR": metrics["meteor"],
                "Kendall Tau": metrics["kendall_tau"]
            }
            results.append(res)
            print("  ✅ Done")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
    # 3. Print Comparison Table
    if results:
        df = pd.DataFrame(results)
        # Format floats
        float_cols = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "METEOR", "Kendall Tau"]
        for col in float_cols:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
            
        print("\n" + "="*80)
        print("FINAL RESULTS TABLE")
        print("="*80)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = output_dir / "comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved table to {csv_path}")
    else:
        print("\nNo results to show.")

if __name__ == "__main__":
    compare_methods()