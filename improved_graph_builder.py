#!/usr/bin/env python3
"""
Improved Graph Builder Module for TAEG - Better Node Structure
Creates separate nodes for each gospel version of an event.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import xml.etree.ElementTree as ET
import re

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import ChronologyLoader, BiblicalDataLoader


import pickle
import os

class ImprovedTemporalGraphBuilder:
    """Builder for temporal graphs with separate nodes for each gospel version."""

    def __init__(self, cache_dir: str = "outputs"):
        """Initialize the improved graph builder."""
        self.chrono_loader = ChronologyLoader()
        self.biblical_loader = BiblicalDataLoader()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_file = self.cache_dir / "taeg_graph_cache.pkl"

    def build_improved_temporal_graph(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Build temporal graph with separate nodes for each gospel version of events.
        
        Args:
            use_cache: If True, try to load from cache first.

        Returns:
            Dictionary representing the improved temporal graph
        """
        if use_cache and self.cache_file.exists():
            print(f"📦 Loading cached graph from {self.cache_file}...")
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load cache: {e}. Rebuilding...")

        print("🔍 Loading chronology data...")
        events = self.chrono_loader.load_chronology()
        print(f"✅ Loaded {len(events)} chronological events")

        graph = {
            'nodes': {},
            'edges': [],
            'statistics': {
                'total_events': len(events),
                'total_nodes': 0,
                'gospel_specific_nodes': 0,
                'combined_nodes': 0,
                'before_edges': 0,
                'same_event_edges': 0,
                'events_with_multiple_gospels': 0,
                'gospel_distribution': {'matthew': 0, 'mark': 0, 'luke': 0, 'john': 0}
            }
        }

        # Create nodes for each event
        print("🏗️ Creating gospel-specific nodes...")
        for event in events:
            event_id = event['id']

            # Get gospels that mention this event
            gospels_mentioning = []
            for gospel in ['matthew', 'mark', 'luke', 'john']:
                if event.get(gospel) and event[gospel].strip():
                    gospels_mentioning.append(gospel)
                    graph['statistics']['gospel_distribution'][gospel] += 1

            if len(gospels_mentioning) > 1:
                graph['statistics']['events_with_multiple_gospels'] += 1

            # Create individual nodes for each gospel version
            for gospel in gospels_mentioning:
                node_id = f"event_{event_id}_{gospel}"
                reference = event[gospel]

                # Extract specific verses for this gospel
                verse_text = self._extract_specific_verses(gospel, reference)

                graph['nodes'][node_id] = {
                    'event_id': event_id,
                    'gospel': gospel,
                    'reference': reference,
                    'description': event['description'],
                    'text': verse_text,
                    'text_length': len(verse_text) if verse_text else 0,
                    'event': event
                }
                graph['statistics']['gospel_specific_nodes'] += 1

                print(f"  📄 Created node {node_id}: '{event['description']}' ({gospel} {reference}) - {len(verse_text) if verse_text else 0} chars")

        graph['statistics']['total_nodes'] = len(graph['nodes'])
        print(f"✅ Created {len(graph['nodes'])} gospel-specific nodes")

        # Create BEFORE edges between consecutive events (across all gospels)
        print("🔗 Creating BEFORE edges...")
        self._create_before_edges(graph, events)
        print(f"✅ Created {graph['statistics']['before_edges']} BEFORE edges")

        # Create SAME_EVENT edges between different gospel versions of the same event
        print("🔗 Creating SAME_EVENT edges...")
        self._create_same_event_edges(graph, events)
        print(f"✅ Created {graph['statistics']['same_event_edges']} SAME_EVENT edges")

        # Print detailed statistics
        self._print_improved_graph_statistics(graph)

        # Save to cache
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(graph, f)
            print(f"💾 Graph saved to cache: {self.cache_file}")
        except Exception as e:
            print(f"⚠️ Could not save to cache: {e}")

        return graph

    def _extract_specific_verses(self, gospel: str, reference: str) -> str:
        """
        Extract specific verses from a gospel based on reference.
        Handles 'a'/'b' suffixes by splitting on strong punctuation.

        Args:
            gospel: Gospel name ('matthew', 'mark', 'luke', 'john')
            reference: Reference string like "26:6-13", "20:1a", "24:5-6a"

        Returns:
            Extracted verse text
        """
        try:
            # Parse reference (e.g., "26:6-13" -> chapter 26, verses 6-13)
            if ':' not in reference:
                return ""

            chapter_part, verse_part = reference.split(':', 1)
            chapter_num = int(chapter_part)

            start_suffix = ""
            end_suffix = ""

            # Parse verse range
            if '-' in verse_part:
                start_verse_str, end_verse_str = verse_part.split('-', 1)
                
                # Extract suffixes if present
                if start_verse_str[-1].isalpha():
                    start_suffix = start_verse_str[-1].lower()
                    start_verse = int(re.sub(r'[^\d]', '', start_verse_str))
                else:
                    start_verse = int(re.sub(r'[^\d]', '', start_verse_str))
                    
                if end_verse_str[-1].isalpha():
                    end_suffix = end_verse_str[-1].lower()
                    end_verse = int(re.sub(r'[^\d]', '', end_verse_str))
                else:
                    end_verse = int(re.sub(r'[^\d]', '', end_verse_str))
            else:
                # Single verse or verse part
                clean_verse = re.sub(r'[^\d]', '', verse_part)
                start_verse = end_verse = int(clean_verse)
                if verse_part[-1].isalpha():
                    start_suffix = end_suffix = verse_part[-1].lower()

            # Load gospel XML
            gospels_map = {
                'matthew': 'EnglishNIVMatthew40_PW.xml',
                'mark': 'EnglishNIVMark41_PW.xml',
                'luke': 'EnglishNIVLuke42_PW.xml',
                'john': 'EnglishNIVJohn43_PW.xml'
            }

            xml_file = Path("data") / gospels_map[gospel.lower()]
            if not xml_file.exists():
                return ""

            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find the specific chapter
            for chapter in root.findall('.//chapter'):
                if chapter.get('number') and int(chapter.get('number')) == chapter_num:
                    verses_text = []

                    # Extract specified verse range
                    for verse in chapter.findall('.//verse'):
                        verse_num = verse.get('number')
                        if verse_num and verse.text:
                            try:
                                v_num = int(verse_num)
                                if start_verse <= v_num <= end_verse:
                                    text = verse.text.strip()
                                    
                                    # Apply suffix logic for start verse
                                    if v_num == start_verse and start_suffix:
                                        text = self._apply_verse_suffix(text, start_suffix)
                                        
                                    # Apply suffix logic for end verse (if different or if range)
                                    if v_num == end_verse and end_suffix and start_verse != end_verse:
                                        text = self._apply_verse_suffix(text, end_suffix)

                                    if text:
                                        verses_text.append(text)
                            except ValueError:
                                continue

                    return ' '.join(verses_text)

        except Exception as e:
            print(f"Warning: Could not extract verses for {gospel} {reference}: {e}")
            return ""

        return ""

    def _apply_verse_suffix(self, text: str, suffix: str) -> str:
        """
        Split text based on suffix 'a' (first part) or 'b' (second part).
        Finds the split point closest to the middle of the text.
        """
        # Candidate separators: strong punctuation or ", and "
        # We look for indices of these separators
        separators = [";", ":", ".", "?", "!", ", and "]
        
        best_split_idx = -1
        min_dist_to_center = float('inf')
        center = len(text) / 2
        split_len = 0
        
        for sep in separators:
            # Find all occurrences of this separator
            start = 0
            while True:
                idx = text.find(sep, start)
                if idx == -1:
                    break
                
                # Calculate distance to center
                # We consider the split point to be AFTER the separator for clean cutting
                # "Clause A; Clause B" -> split at index of ';' + 1
                dist = abs(idx - center)
                
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_split_idx = idx
                    split_len = len(sep)
                
                start = idx + 1

        # If no separator found or text is too short, return full text
        if best_split_idx == -1:
            # Fallback: simple comma split closest to center
            # Text: "Early on ... dark, Mary ..."
            idx = text.find(", ", 0)
            while idx != -1:
                dist = abs(idx - center)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_split_idx = idx
                    split_len = 2 # length of ", "
                idx = text.find(", ", idx + 1)
        
        if best_split_idx == -1:
            return text
            
        # Split logic
        # For ", and " or ", " we typically want to remove the connector from the end of A
        # or keep it at start of B? 
        # Actually, usually "Clause A, and Clause B":
        # Part A = "Clause A"
        # Part B = "and Clause B" -> logic says 'b' usually keeps the connector if any
        # But if separator is ; or . we keep it in A.
        
        sep_used = text[best_split_idx:best_split_idx + split_len]
        
        if sep_used.strip() == "," or "and" in sep_used:
             # Separator is likely ", " or ", and "
             # Part A: everything before
             # Part B: separator (?) + everything after
             part_a = text[:best_split_idx].strip()
             # Optionally include "and" in B if desired, or exclude separator entirely
             part_b = text[best_split_idx + split_len:].strip()
             if "and" in sep_used:
                 part_b = "and " + part_b
        else:
            # Strong punctuation (., ; etc) -> Keep with A
            part_a = text[:best_split_idx + split_len].strip()
            part_b = text[best_split_idx + split_len:].strip()
        
        if suffix == 'a':
            return part_a
        elif suffix == 'b':
            return part_b
            
        return text

    def _create_before_edges(self, graph: Dict[str, Any], events: List[Dict]):
        """Create BEFORE edges between consecutive events."""
        # Group nodes by event_id
        event_nodes = {}
        for node_id, node_data in graph['nodes'].items():
            event_id = node_data['event_id']
            if event_id not in event_nodes:
                event_nodes[event_id] = []
            event_nodes[event_id].append(node_id)

        # Create BEFORE edges between consecutive events
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]

            current_event_id = current_event['id']
            next_event_id = next_event['id']

            # Connect all nodes of current event to all nodes of next event
            if current_event_id in event_nodes and next_event_id in event_nodes:
                for current_node in event_nodes[current_event_id]:
                    for next_node in event_nodes[next_event_id]:
                        graph['edges'].append({
                            'from': current_node,
                            'to': next_node,
                            'type': 'BEFORE',
                            'description': f"Event {current_event_id} → Event {next_event_id}"
                        })
                        graph['statistics']['before_edges'] += 1

    def _create_same_event_edges(self, graph: Dict[str, Any], events: List[Dict]):
        """Create SAME_EVENT edges between different gospel versions of the same event."""
        # Group nodes by event_id
        event_nodes = {}
        for node_id, node_data in graph['nodes'].items():
            event_id = node_data['event_id']
            if event_id not in event_nodes:
                event_nodes[event_id] = []
            event_nodes[event_id].append(node_id)

        # Create SAME_EVENT edges between different gospel versions of the same event
        for event_id, nodes in event_nodes.items():
            if len(nodes) > 1:
                # Connect all pairs of nodes for this event
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        node1 = nodes[i]
                        node2 = nodes[j]

                        gospel1 = graph['nodes'][node1]['gospel']
                        gospel2 = graph['nodes'][node2]['gospel']

                        graph['edges'].append({
                            'from': node1,
                            'to': node2,
                            'type': 'SAME_EVENT',
                            'description': f"Same event {event_id}: {gospel1} ↔ {gospel2}"
                        })
                        graph['statistics']['same_event_edges'] += 1

    def _print_improved_graph_statistics(self, graph: Dict[str, Any]):
        """Print detailed statistics about the improved graph."""
        stats = graph['statistics']

        print("\n" + "="*70)
        print("📊 IMPROVED GRAPH STATISTICS")
        print("="*70)

        print(f"📅 Total Events: {stats['total_events']}")
        print(f"🏗️ Total Nodes: {stats['total_nodes']}")
        print(f"📄 Gospel-Specific Nodes: {stats['gospel_specific_nodes']}")
        print(f"🔗 BEFORE Edges: {stats['before_edges']}")
        print(f"🔄 SAME_EVENT Edges: {stats['same_event_edges']}")
        print(f"📚 Events with Multiple Gospels: {stats['events_with_multiple_gospels']}")

        print("\n📖 Gospel Distribution:")
        for gospel, count in stats['gospel_distribution'].items():
            print(f"  {gospel.capitalize()}: {count} mentions")

        print("\n🔍 Sample Gospel-Specific Nodes:")
        sample_nodes = list(graph['nodes'].items())[:8]
        for node_id, node_data in sample_nodes:
            text_preview = node_data['text'][:100] + "..." if node_data['text'] and len(node_data['text']) > 100 else node_data['text']
            print(f"  {node_id}: '{node_data['description']}' ({node_data['gospel']} {node_data['reference']})")
            print(f"    Text: {text_preview}")

        print("\n🔗 Sample Edges:")
        for edge in graph['edges'][:10]:
            print(f"  {edge['from']} --[{edge['type']}]--> {edge['to']}")

        if len(graph['edges']) > 10:
            print(f"  ... and {len(graph['edges']) - 10} more edges")

        print("="*70)


def main():
    """Main function to build and analyze the improved temporal graph."""
    print("🚀 TAEG - Improved Temporal Graph Builder")
    print("="*70)

    builder = ImprovedTemporalGraphBuilder()
    graph = builder.build_improved_temporal_graph()

    print("\n✅ Improved graph construction completed!")
    print(f"📊 Graph has {graph['statistics']['total_nodes']} nodes and {len(graph['edges'])} edges")


if __name__ == "__main__":
    main()