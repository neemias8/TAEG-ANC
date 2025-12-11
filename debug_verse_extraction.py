
from improved_graph_builder import ImprovedTemporalGraphBuilder

def test_extraction():
    builder = ImprovedTemporalGraphBuilder()
    
    # Test Event 0: Anointing at Bethany
    # Expected: John 12:1-8, Matthew 26:6-13, Mark 14:3-9
    
    print("--- Testing John 12:1-8 ---")
    text_john = builder._extract_specific_verses("John", "12:1-8")
    print(f"Result length: {len(text_john)}")
    print(f"Result preview: {text_john[:100]}...")
    
    print("\n--- Testing Matthew 26:6-13 ---")
    text_matt = builder._extract_specific_verses("Matthew", "26:6-13")
    print(f"Result length: {len(text_matt)}")
    print(f"Result preview: {text_matt[:100]}...")

    # Test single verse with suffix if applicable (though these ranges don't explicitly have suffixes in standard ref, maybe chronology does?)
    # Let's test a known suffix case if we can find one, or just the ranges.
    
    # Also test the specific result found in the output: "I tell you the truth..." which is Matt 26:13 / Mark 14:9
    print("\n--- Testing Matthew 26:13 (Single) ---")
    text_matt_13 = builder._extract_specific_verses("Matthew", "26:13")
    print(f"Result: {text_matt_13}")

if __name__ == "__main__":
    test_extraction()
