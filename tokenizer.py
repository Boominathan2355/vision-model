import json
import os
from typing import List
import re

try:
    from tokenizers import Tokenizer, models as tk_models, trainers, pre_tokenizers, decoders, processors
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

# ==================== Claude-Style BPE Tokenizer ====================
class CustomTokenizer:
    def __init__(self, vocab_path: str = None, max_length: int = 4096, vocab_size: int = 32000):
        """
        Claude-style Byte-Level BPE Tokenizer.
        If tokenizers library is missing, falls back to basic word-level.
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.special_tokens = [
            '[THINK_START]', '[THINK_END]',
            '[REASON_START]', '[REASON_END]',
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[IMG]'
        ]
        
        if TOKENIZERS_AVAILABLE:
            if vocab_path and os.path.exists(vocab_path):
                self.tokenizer = Tokenizer.from_file(vocab_path)
            else:
                self.tokenizer = Tokenizer(tk_models.BPE(unk_token="[UNK]"))
                self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                self.tokenizer.decoder = decoders.ByteLevel()
                self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
                # Initial state, needs training
        else:
            print("⚠ 'tokenizers' library not found. Falling back to basic word-level.")
            self.tokenizer = None
            
        # Legacy vocab for compatibility
        self.vocab = {}
        self.id_to_token = {}
        
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """Train BPE on provided texts"""
        if TOKENIZERS_AVAILABLE:
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_freq,
                special_tokens=self.special_tokens
            )
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
            
            # Sync legacy vocab
            self.vocab = self.tokenizer.get_vocab()
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            print(f"✓ BPE Tokenizer trained with {len(self.vocab)} tokens")
        else:
            # Simple word-level fallback training
            word_freq = {}
            for text in texts:
                words = re.sub(r'([.,!?;:])', r' \1 ', text).split()
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            self.vocab = {t: i for i, t in enumerate(self.special_tokens)}
            idx = len(self.vocab)
            for word, freq in word_freq.items():
                if freq >= min_freq and word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            print(f"✓ Fallback vocabulary built with {len(self.vocab)} tokens")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            output = self.tokenizer.encode(text)
            tokens = output.ids
            
            # Add CLS/SEP manually if needed (BPE usually handles this in post-processor, but we match old logic)
            if add_special_tokens:
                cls_id = self.vocab.get('[CLS]', 6)
                sep_id = self.vocab.get('[SEP]', 7)
                tokens = [cls_id] + tokens + [sep_id]
        else:
            # Word-level fallback
            words = re.sub(r'([.,!?;:])', r' \1 ', text).split()
            tokens = []
            if add_special_tokens: tokens.append(self.vocab.get('[CLS]', 6))
            for word in words:
                tokens.append(self.vocab.get(word, self.vocab.get('[UNK]', 5)))
            if add_special_tokens: tokens.append(self.vocab.get('[SEP]', 7))

        # Truncate/pad
        pad_id = self.vocab.get('[PAD]', 4)
        tokens = tokens[:self.max_length]
        tokens += [pad_id] * (self.max_length - len(tokens))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        # Filter special tokens
        clean_tokens = [t for t in tokens if t in self.id_to_token and not (self.id_to_token[t].startswith('[') and self.id_to_token[t].endswith(']'))]
        
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            return self.tokenizer.decode(clean_tokens)
        else:
            return ' '.join([self.id_to_token.get(t, '') for t in clean_tokens])

    def save(self, path: str):
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            self.tokenizer.save(path)
            # Also save legacy vocab mapping for resume-without-library if needed
            meta_path = path + ".meta"
            with open(meta_path, 'w') as f:
                json.dump({'vocab': self.vocab, 'max_length': self.max_length}, f)
        else:
            with open(path, 'w') as f:
                json.dump({'vocab': self.vocab, 'max_length': self.max_length}, f)
        print(f"✓ Tokenizer saved to {path}")

    def load(self, path: str):
        if TOKENIZERS_AVAILABLE:
            try:
                self.tokenizer = Tokenizer.from_file(path)
                self.vocab = self.tokenizer.get_vocab()
                self.id_to_token = {v: k for k, v in self.vocab.items()}
            except:
                # Try loading from legacy JSON
                with open(path, 'r') as f:
                    data = json.load(f)
                self.vocab = {k: int(v) for k, v in data['vocab'].items()}
                self.id_to_token = {v: k for k, v in self.vocab.items()}
        else:
            with open(path, 'r') as f:
                data = json.load(f)
            self.vocab = {k: int(v) for k, v in data['vocab'].items()}
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"✓ Tokenizer loaded from {path}")
