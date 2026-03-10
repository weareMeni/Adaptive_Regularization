import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# ==========================================
# 1. DYCK EXTRAPOLATION (The State Test)
# ==========================================
class DyckDataset(Dataset):
    def __init__(self, num_samples, seq_len):
        self.samples = []
        self.labels = []
        
        for _ in range(num_samples):
            is_balanced = random.random() > 0.5
            seq = self.generate_sequence(is_balanced, seq_len)
            self.samples.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(1 if is_balanced else 0)

    def _check_balanced(self, seq):
        count = 0
        for token in seq:
            if token == 1: count += 1
            elif token == 2: count -= 1
            if count < 0: return False
        return count == 0

    def generate_sequence(self, is_balanced, length):
        base_seq = [1] * (length // 2) + [2] * (length // 2)
        if is_balanced:
            seq = []
            open_count = 0
            for i in range(length):
                if open_count > 0 and (open_count == length - i or random.random() > 0.5):
                    seq.append(2)
                    open_count -= 1
                else:
                    seq.append(1)
                    open_count += 1
            return seq
        else:
            seq = base_seq.copy()
            while True:
                random.shuffle(seq)
                if seq[0] == 1 and not self._check_balanced(seq):
                    return seq

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

def get_dyck_extrapolation_loaders(batch_size, train_len=30, test_len=120):
    trainloader = DataLoader(DyckDataset(8000, train_len), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(DyckDataset(2000, test_len), batch_size=batch_size, shuffle=False)
    return trainloader, testloader, 3, 2, train_len, 15

# ==========================================
# 2. ASSOCIATIVE RECALL (The Lookup Test)
# ==========================================
class AssociativeRecallDataset(Dataset):
    def __init__(self, num_samples, num_pairs):
        self.samples = []
        self.labels = []
        for _ in range(num_samples):
            keys = random.sample(range(0, 10), num_pairs)
            values = random.sample(range(10, 20), num_pairs)
            seq = []
            for k, v in zip(keys, values):
                seq.extend([k, v])
            query_idx = random.randint(0, num_pairs - 1)
            seq.append(keys[query_idx])
            self.samples.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(values[query_idx] - 10) 

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

def get_associative_recall_loaders(batch_size):
    num_pairs = 6
    seq_len = (num_pairs * 2) + 1
    trainloader = DataLoader(AssociativeRecallDataset(8000, num_pairs), batch_size=batch_size, shuffle=True)
    testloader = DataLoader(AssociativeRecallDataset(2000, num_pairs), batch_size=batch_size, shuffle=False)
    return trainloader, testloader, 20, 10, seq_len, 10

# ==========================================
# 3. SYNTHETIC LISTOPS (The Boss Fight)
# ==========================================
class ListOpsDataset(Dataset):
    """
    Generates deeply nested math operations.
    Vocab: 0: PAD, 1-10: Digits 0-9
    11: [MIN, 12: [MAX, 13: [MED, 14: [SUM_MOD, 15: ]
    """
    def __init__(self, num_samples, max_seq_len=128, max_depth=4):
        self.max_seq_len = max_seq_len
        self.max_depth = max_depth
        self.samples = []
        self.labels = []
        
        for _ in range(num_samples):
            tokens, label = self._generate_tree(depth=0)
            # Truncate or Pad to exact length
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
                # If we truncated, the math is technically corrupted for the label, 
                # but it serves as a strict max-length bound for the GPU
            else:
                tokens = tokens + [0] * (max_seq_len - len(tokens))
                
            self.samples.append(torch.tensor(tokens, dtype=torch.long))
            self.labels.append(label)

    def _generate_tree(self, depth):
        if depth >= self.max_depth or random.random() < 0.2:
            val = random.randint(0, 9)
            return [val + 1], val # Digits are shifted by +1 in vocab
            
        op = random.choice([11, 12, 13, 14])
        num_args = random.randint(2, 4)
        
        tokens = [op]
        args_vals = []
        for _ in range(num_args):
            arg_tokens, arg_val = self._generate_tree(depth + 1)
            tokens.extend(arg_tokens)
            args_vals.append(arg_val)
            
        tokens.append(15) # The closing bracket ']'
        
        # Calculate ground truth
        if op == 11: label = min(args_vals)
        elif op == 12: label = max(args_vals)
        elif op == 13: label = int(np.median(args_vals))
        elif op == 14: label = sum(args_vals) % 10
            
        return tokens, label

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx], self.labels[idx]

def get_listops_loaders(batch_size):
    seq_len = 128
    # Train strictly on shallow trees
    train_dataset = ListOpsDataset(8000, max_seq_len=seq_len, max_depth=4)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Test strictly on deep trees (OOD)
    test_dataset = ListOpsDataset(2000, max_seq_len=seq_len, max_depth=8)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    vocab_size = 17 
    num_classes = 10 
    return trainloader, testloader, vocab_size, num_classes, seq_len, 100