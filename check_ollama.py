import requests
import sys

def check_ollama(model_name="gemma3:4b"):
    url = "http://localhost:11434/api/tags"
    try:
        print(f"📡 Connecting to Ollama at {url}...")
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        models = [m['name'] for m in data.get('models', [])]
        print(f"✅ Ollama is RUNNING.")
        print("📋 Installed Models:")
        for m in models:
            print(f"  - {m}")
            
        print("-" * 40)
        
        # Check for specific model match (loose match)
        if any(model_name in m for m in models):
             print(f"✅ Target model '{model_name}' found!")
             print("🚀 You are ready to run: python run_taeg_abstractive.py --method gemma")
        else:
             print(f"⚠️ Target model '{model_name}' NOT found in list.")
             print(f"👉 Please run: ollama pull {model_name}")
             print("(Or edit src/consolidators.py to match an installed model name)")
             
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama.")
        print("Make sure Ollama is installed and running (check taskbar or run 'ollama serve').")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_ollama()
