"""
Consolidators module for TAEG using Strategy Pattern.
Defines abstract base class and concrete implementations for different summarization models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import nltk

class BaseConsolidator(ABC):
    """Abstract base class for all consolidators."""
    
    @abstractmethod
    def consolidate(self, texts: List[str]) -> str:
        """
        Consolidate a list of texts into a single summary.
        
        Args:
            texts: List of input text segments.
            
        Returns:
            The consolidated summary string.
        """
        pass

class AbstractiveConsolidator(BaseConsolidator):
    """Base class for HF Transformer models."""
    
    def __init__(self, model_name: str, max_length: int = 150, min_length: int = 40, generation_kwargs: Dict[str, Any] = None):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        self.device = "cpu" # Enforced by requirements
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.min_length = min_length
        self.generation_kwargs = generation_kwargs or {}

        # Determine max input length from model config or tokenizer
        if hasattr(self.model.config, 'max_position_embeddings'):
            self.max_input_length = self.model.config.max_position_embeddings
        elif hasattr(self.tokenizer, 'model_max_length'):
            self.max_input_length = self.tokenizer.model_max_length
        else:
            self.max_input_length = 1024 # Fallback
            
        print(f"Model {model_name} initialized with max_input_length={self.max_input_length}")
        
    def consolidate(self, texts: List[str]) -> str:
        if not texts:
            return ""
            
        # Concatenate inputs
        input_text = " ".join(texts)
        
        # Tokenize (truncate mostly for start, we might lose end context but it's a heuristic)
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=self.max_input_length, 
            truncation=True
        ).to(self.device)
        
        # Define default args
        gen_args = {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "length_penalty": 2.0,
            "num_beams": 4,
            "early_stopping": True
        }
        # Override with custom kwargs
        gen_args.update(self.generation_kwargs)
        
        # Generate
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            **gen_args
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class BartConsolidator(AbstractiveConsolidator):
    def __init__(self, max_length: int = 150, min_length: int = 40):
        super().__init__("facebook/bart-large-cnn", max_length=max_length, min_length=min_length)

class PegasusConsolidator(AbstractiveConsolidator):
    def __init__(self, max_length: int = 80, min_length: int = 10):
        # Switching to cnn_dailymail as multi_news was hallucinatory (biased to news style).
        # CNN/DM is more standard.
        params = {
            "length_penalty": 1.0, 
            "num_beams": 4, # Back to beam search
            "repetition_penalty": 1.2
        }
        super().__init__("google/pegasus-cnn_dailymail", max_length=max_length, min_length=min_length, generation_kwargs=params)

    # Removed prompts as they didn't help with multi_news and might confuse cnn_dailymail
    def consolidate(self, texts: List[str]) -> str:
        summary = super().consolidate(texts)
        # cnn_dailymail model uses <n> as a sentence separator. We replace it with space.
        return summary.replace("<n>", " ")

class PrimeraConsolidator(AbstractiveConsolidator):
    def __init__(self, max_length: int = 256, min_length: int = 50):
        super().__init__("allenai/PRIMERA", max_length=max_length, min_length=min_length)

    def consolidate(self, texts: List[str]) -> str:
        if not texts:
            return ""
            
        # PRIMERA uses <doc-sep> usually, but space is fine for now
        return super().consolidate(texts)

class GemmaOllamaConsolidator(BaseConsolidator):
    """
    Consolidator using a local Ollama instance (Gemma 3).
    Requires Ollama running at localhost:11434.
    """
    def __init__(self, model_name: str = "gemma3:4b", timeout: int = 300):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.timeout = timeout
        print(f"initialized GemmaOllamaConsolidator with model={model_name}")

    def consolidate(self, texts: List[str]) -> str:
        if not texts:
            return ""
            
        import requests
        import json
        
        # Concatenate inputs for the prompt context
        combined_text = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(texts)])
        
        # User requested specific prompt: CONSOLIDATE (keep details, eliminate redundancy) 
        # instead of SUMMARIZE.
        prompt = (
            "You are an expert narrative consolidator. Your task is to merge the following versions "
            "of the same event into a single, cohesive narrative.\n\n"
            "Rules:\n"
            "1. PRESERVE unique details from ALL sources. If one source mentions a detail the others miss, INCLUDE it.\n"
            "2. ELIMINATE redundancy. Do not repeat the same fact twice.\n"
            "3. MAINTAIN chronological flow and grammatical coherence.\n"
            "4. DO NOT summarize or shorten unnecessarily. The goal is consolidation, not compression.\n\n"
            "Input Texts:\n"
            f"{combined_text}\n\n"
            "Consolidated Narrative:"
        )
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3, # Low temp for factual consistency
                "num_ctx": 4096     # Ensure enough context window
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            return f"[Error generating consolidation with {self.model_name}]"
