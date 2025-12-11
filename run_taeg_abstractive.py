#!/usr/bin/env python3
"""
TAEG Abstractive Runner.
Uses the Temporal Graph to guide abstractive summarization (BART, PEGASUS, PRIMERA).
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from improved_graph_builder import ImprovedTemporalGraphBuilder
from data_loader import ChronologyLoader
from consolidators import (
    BartConsolidator, 
    PegasusConsolidator, 
    PrimeraConsolidator,
    GemmaOllamaConsolidator,
    BaseConsolidator
)

def get_consolidator(method: str) -> BaseConsolidator:
    method = method.lower()

    if method == "bart":
        return BartConsolidator()
    elif method == "pegasus":
        return PegasusConsolidator()
    elif method == "primera":
        return PrimeraConsolidator()
    elif method == "gemma":
        return GemmaOllamaConsolidator() # Default model defined in class
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    parser = argparse.ArgumentParser(description="TAEG Abstractive Summarization")
    parser.add_argument("--method", choices=["bart", "pegasus", "primera", "gemma"], required=True, help="Summarization model")
    parser.add_argument("--limit-events", type=int, default=None, help="Limit number of events for testing")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    print(f"🚀 Starting TAEG-Abstractive with {args.method.upper()}...")
    
    # 1. Build Graph
    print("Building temporal graph...")
    graph_builder = ImprovedTemporalGraphBuilder()
    graph = graph_builder.build_improved_temporal_graph()
    
    # 2. Prepare Event Groups
    event_nodes = {}
    for node_id, node_data in graph['nodes'].items():
        event_id = node_data['event_id']
        if event_id not in event_nodes:
            event_nodes[event_id] = []
        event_nodes[event_id].append(node_data)
        
    # 3. Initialize Consolidator
    try:
        consolidator = get_consolidator(args.method)
    except Exception as e:
        print(f"❌ Error initializing consolidator: {e}")
        sys.exit(1)
        
    # 4. Process Events
    loader = ChronologyLoader()
    events = loader.load_chronology()
    
    if args.limit_events:
        events = events[:args.limit_events]
        print(f"⚠️ Limiting to first {args.limit_events} events")
        
    generated_summaries = []
    
    print("\nProcessing events...")
    start_time = time.time()
    
    for i, event in enumerate(events):
        event_id = event['id']
        description = event['description']
        
        print(f"[{i+1}/{len(events)}] Processing Event {event_id}: {description}...", end="", flush=True)
        
        if event_id in event_nodes:
            # Collect texts from all gospel versions for this event
            node_datas = event_nodes[event_id]
            texts = [n['text'] for n in node_datas if n['text']]
            
            if texts:
                # Consolidate
                try:
                    summary = consolidator.consolidate(texts)
                    generated_summaries.append((event_id, summary))
                    print(f" ✅ ({len(texts)} sources)")
                except Exception as e:
                    print(f" ❌ Error: {e}")
                    generated_summaries.append((event_id, description)) # Fallback
            else:
                print(" ⚠️ No text found (using description)")
                generated_summaries.append((event_id, description))
        else:
            print(" ⚠️ No nodes found (using description)")
            generated_summaries.append((event_id, description))
            
    total_time = time.time() - start_time
    print(f"\n✨ Processing completed in {total_time:.2f} seconds")
    
    # 5. Save Output
    clean_summaries = []
    for event_id, s in generated_summaries:
        s = s.strip()
        if not s: continue
        # Ensure it ends with punctuation
        if s[-1] not in {'.', '!', '?', '"', "'"}:
            s += "."
        # Fix spacing after punctuation (e.g. "Priest.But")
        import re
        s = re.sub(r'([.!?])([A-Z])', r'\1 \2', s)
        
        # User reported: is near."I tell, away."Jesus, well!"Jesus
        # Fix missing space after closing quote + punctuation followed by a letter.
        # We target: [.!?]["']<letter>
        # e.g. ."I -> ." I
        s = re.sub(r'([.!?])([\'"])([a-zA-Z])', r'\1\2 \3', s)
        
        # Also handle cases where there might not be punctuation inside the quote but it acts as end?
        # But usually user said "depois de ." (after .).
        
        # Fix double punctuation
        s = s.replace('..', '.').replace('!!', '!').replace('??', '?')
        # Fix quote spacing if needed (e.g. me?"") - simplify slightly
        s = s.replace('""', '"')
        
        # Prepend Event ID
        s = f"[Event {event_id}] {s}"
        
        clean_summaries.append(s)
        
    final_text = "\n".join(clean_summaries)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"taeg_{args.method}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"💾 Saved output to {output_file}")

if __name__ == "__main__":
    main()
