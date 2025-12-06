import json
import os
from typing import List
import re

# ==================== Custom Tokenizer ====================
class CustomTokenizer:
    def __init__(self, vocab_path: str = None, max_length: int = 512):
        """Custom tokenizer with thinking/reasoning tokens"""
        self.max_length = max_length
        self.thinking_tokens = {
            '[THINK_START]': 0,
            '[THINK_END]': 1,
            '[REASON_START]': 2,
            '[REASON_END]': 3,
            '[PAD]': 4,
            '[UNK]': 5,
            '[CLS]': 6,
            '[SEP]': 7,
            '[IMG]': 8,
        }
        
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)
        else:
            # Start with thinking tokens
            self.vocab = self.thinking_tokens.copy()
            self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            words = self._preprocess(text).split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add words meeting frequency threshold
        idx = len(self.vocab)
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = idx
                idx += 1
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"✓ Vocabulary built with {len(self.vocab)} tokens")
    
    def _preprocess(self, text: str) -> str:
        """Preprocess text"""
        text = text.lower()
        # Add space before punctuation for better tokenization
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        words = self._preprocess(text).split()
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.vocab['[CLS]'])
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Try subword tokenization
                subwords = self._subword_tokenize(word)
                tokens.extend([self.vocab.get(sw, self.vocab['[UNK]']) for sw in subwords])
        
        if add_special_tokens:
            tokens.append(self.vocab['[SEP]'])
        
        # Truncate/pad
        tokens = tokens[:self.max_length]
        tokens += [self.vocab['[PAD]']] * (self.max_length - len(tokens))
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in tokens:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token.startswith('[') and token.endswith(']'):
                    continue  # Skip special tokens
                words.append(token)
        return ' '.join(words)
    
    def add_thinking_tokens(self, tokens: List[int]) -> List[int]:
        """Add thinking tokens around sequence"""
        return ([self.vocab['[THINK_START]']] + tokens + 
                [self.vocab['[THINK_END]']] +
                [self.vocab['[REASON_START]']] + 
                tokens +  # Same tokens for reasoning
                [self.vocab['[REASON_END]']])
    
    def _subword_tokenize(self, word: str) -> List[str]:
        """Simple subword tokenization"""
        subwords = []
        for i in range(len(word)):
            subwords.append(word[:i+1])
        return subwords[:3]  # Return at most 3 subwords
    
    def save(self, path: str):
        """Save tokenizer"""
        data = {
            'vocab': self.vocab,
            'max_length': self.max_length,
            'thinking_tokens': self.thinking_tokens
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"✓ Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load tokenizer"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = {k: int(v) for k, v in data['vocab'].items()}
        self.max_length = data['max_length']
        self.thinking_tokens = data['thinking_tokens']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"✓ Tokenizer loaded from {path}")
