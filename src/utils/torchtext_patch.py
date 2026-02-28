"""
TorchText Patch Utility
Fixes torchtext loading issues for scGPT
"""

import os
import sys
from types import ModuleType


def patch_torchtext():
    """
    Apply patches to fix torchtext loading issues
    
    This prevents errors when loading scGPT models due to 
    missing libtorchtext library
    """
    try:
        import torch
        import collections
        
        # 1. Mock torchtext._extension to prevent loading .so library
        mock_extension = ModuleType("torchtext._extension")
        mock_extension._init_extension = lambda: None
        sys.modules["torchtext._extension"] = mock_extension
        
        # 2. Intercept torch.ops.load_library
        orig_load_library = torch.ops.load_library
        def mocked_load(path):
            if "libtorchtext" in path:
                return
            return orig_load_library(path)
        torch.ops.load_library = mocked_load
        
        # 3. Construct mock Vocab class and module
        class MockVocab:
            def __init__(self, vocab):
                self.vocab = vocab
                self.itos = list(vocab.keys()) if isinstance(vocab, dict) else []
            def __len__(self): 
                return len(self.itos)
        
        # Create mock torchtext.vocab module
        mt_vocab = ModuleType("torchtext.vocab")
        mt_vocab.Vocab = MockVocab
        sys.modules["torchtext.vocab"] = mt_vocab
        
        # Create mock torchtext top-level module
        mt_root = ModuleType("torchtext")
        mt_root.vocab = mt_vocab
        sys.modules["torchtext"] = mt_root
        
        print("--- torchtext patch applied successfully ---")
        return True
        
    except Exception as e:
        print(f"--- torchtext patch failed: {e} ---")
        return False


# Apply patch when module is imported
patch_torchtext()
