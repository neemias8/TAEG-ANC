#!/usr/bin/env python3
"""
Pure Abstractive Runner (Baseline).
Concatenates all texts and summarizes globally.
"""

import sys
import argparse
from pathlib import Path
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import BiblicalDataLoader
from consolidators import (
    BartConsolidator, 
    PegasusConsolidator, 
    PrimeraConsolidator,
    GemmaOllamaConsolidator,
    BaseConsolidator
)

def get_consolidator(method: str) -> BaseConsolidator:
    method = method.lower()
    # For Pure Abstractive, we are summarizing the WHOLE set of gospels (huge input).
    # We should allow much longer outputs than the per-event summaries.
    # defaults were ~80-150. Here we want ~500-1000.
    
    LIMITS = {"max_length": 1024, "min_length": 100}
    
    if method == "bart":
        return BartConsolidator(**LIMITS)
    elif method == "pegasus":
        return PegasusConsolidator(**LIMITS)
    elif method == "primera":
        return PrimeraConsolidator(**LIMITS) # PRIMERA can handle it
    elif method == "gemma":
        return GemmaOllamaConsolidator() # Ollama handles context via prompt/options
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    parser = argparse.ArgumentParser(description="Pure Abstractive Summarization (Baseline)")
    parser.add_argument("--method", choices=["bart", "pegasus", "primera", "gemma"], required=True, help="Summarization model")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    print(f"🚀 Starting Pure-Abstractive with {args.method.upper()}...")
    
    # 1. Load Data
    print("Loading all gospel texts...")
    loader = BiblicalDataLoader()
    gospel_texts = loader.load_all_gospels()
    
    # Concatenate all to single string
    full_text = []
    total_chars = 0
    for gospel, text in gospel_texts.items():
        print(f"  📖 {gospel}: {len(text)} chars")
        full_text.append(text)
        total_chars += len(text)
        
    print(f"Total input size: {total_chars} characters")
    
    # 2. Initialize Consolidator
    try:
        consolidator = get_consolidator(args.method)
    except Exception as e:
        print(f"❌ Error initializing consolidator: {e}")
        sys.exit(1)
        
    # 3. Consolidate
    print("Summarizing (this may take a while)...")
    start_time = time.time()
    
    try:
        # Pass as a list of "documents" (gospels)
        # Consolidators handle concatenation if needed, or truncation in tokenizer
        summary = consolidator.consolidate(full_text)
        print("✅ Summarization complete")
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        sys.exit(1)
        
    total_time = time.time() - start_time
    print(f"✨ Processing completed in {total_time:.2f} seconds")
    
    # 4. Save Output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"pure_{args.method}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)
        
    print(f"💾 Saved output to {output_file}")

if __name__ == "__main__":
    main()
