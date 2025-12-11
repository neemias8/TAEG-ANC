"""
Consolidators module for TAEG using Strategy Pattern.
Defines abstract base class and concrete implementations for different summarization models.
"""

from abc import ABC, abstractmethod
from typing import List
import nltk

# Transformers imports will be inside classes to avoid importing if not used/installed yet
# but for type hinting we might need them or just use "Any"

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
    
    def __init__(self, model_name: str, max_length: int = 150, min_length: int = 40):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        self.device = "cpu" # Enforced by requirements
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
        self.min_length = min_length

        # Determine max input length from model config or tokenizer
        if hasattr(self.model.config, 'max_position_embeddings'):
            self.max_input_length = self.model.config.max_position_embeddings
        elif hasattr(self.tokenizer, 'model_max_length'):
            self.max_input_length = self.tokenizer.model_max_length
        else:
            self.max_input_length = 1024 # Fallback
            
        # Adjust for safety (some models have huge limits but we might want to cap processing)
        # But for Pegasus (512) vs BART (1024) vs PRIMERA (4096), we typically want the model's limit.
        print(f"Model {model_name} initialized with max_input_length={self.max_input_length}")
        
    def consolidate(self, texts: List[str]) -> str:
        if not texts:
            return ""
            
        # Concatenate inputs
        input_text = " ".join(texts)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=self.max_input_length, 
            truncation=True
        ).to(self.device)
        
        # Generate
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            max_length=self.max_length, 
            min_length=self.min_length, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class BartConsolidator(AbstractiveConsolidator):
    def __init__(self):
        super().__init__("facebook/bart-large-cnn")

class PegasusConsolidator(AbstractiveConsolidator):
    def __init__(self):
        # Switching to multi_news as it is specifically trained for MDS
        # giving it an advantage over cnn_dailymail for combining gospel versions
        super().__init__("google/pegasus-multi_news")

class PrimeraConsolidator(AbstractiveConsolidator):
    def __init__(self):
        # PRIMERA is designed for MDS, handling longer contexts (4096)
        # However, for consistency in class verification locally, we stick to the pattern
        # Be aware PRIMERA might need specific special tokens for separating documents if used in pure mode
        # But here we just concat.
        super().__init__("allenai/PRIMERA", max_length=256) # PRIMERA produces longer summaries usually

    def consolidate(self, texts: List[str]) -> str:
        # PRIMERA handles multi-doc explicitly with special separator tokens usually, 
        # but for simple usage we can concat with " <doc-sep> " if the tokenizer supports it, 
        # or just space. Let's start with space concatenation as in the base class for simplicity
        # unless specific PRIMERA usage requires otherwise.
        # Checking PRIMERA docs: it uses <doc-sep>
        
        if not texts:
            return ""
            
        # Try to use doc separator if token exists, else space
        separator = " <doc-sep> " 
        # But to be safe and avoid token errors if not added, we stick to space for now 
        # or check tokenizer.
        
        return super().consolidate(texts)

