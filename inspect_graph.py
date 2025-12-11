import pickle
import sys

try:
    with open('outputs/taeg_graph_cache.pkl', 'rb') as f:
        graph = pickle.load(f)
        
    nodes_to_check = ['event_146_john', 'event_148_john', 'event_155_matthew', 'event_156_john', 'event_157_john']
    
    print("--- INSPECTING SPECIFIC NODES ---")
    for node_id in nodes_to_check:
        if node_id in graph['nodes']:
            node = graph['nodes'][node_id]
            print(f"\nNode: {node_id}")
            print(f"Ref: {node['gospel']} {node['reference']}")
            print(f"Text ({len(node['text'])} chars): [{node['text']}]")
        else:
            print(f"\nNode {node_id} not found")

except Exception as e:
    print(e)
