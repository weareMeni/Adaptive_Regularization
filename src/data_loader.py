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
    return trainloader, testloader, 3, 2, train_len, 30

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
    return trainloader, testloader, 20, 10, seq_len, 100

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
    return trainloader, testloader, vocab_size, num_classes, seq_len, 20

class CoTListOpsDataset(Dataset):
    def __init__(self, num_samples, max_depth=3, max_seq_len=256):
        """
        Generates ListOps sequences with full step-by-step Chain of Thought traces.
        Vocab: 
        0: <PAD>, 1-9: Digits
        10: MIN, 11: MAX, 12: MED, 13: SM (Sum Modulo 10)
        14: '[', 15: ']'
        16: '=', 17: '<EOS>'
        """
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.max_seq_len = max_seq_len
        self.vocab_size = 18
        
        self.ops = {10: "MIN", 11: "MAX", 12: "MED", 13: "SM"}
        self.op_keys = list(self.ops.keys())
        
        self.data = []
        for _ in range(num_samples):
            prompt, trace = self._generate_sample(self.max_depth)
            full_seq = prompt + trace
            
            # Pad or truncate
            if len(full_seq) > max_seq_len:
                full_seq = full_seq[:max_seq_len]
            else:
                full_seq = full_seq + [0] * (max_seq_len - len(full_seq))
                
            self.data.append(full_seq)
            
        self.data = torch.tensor(self.data, dtype=torch.long)

    def _generate_tree(self, depth):
        """Recursively generates the initial ListOps tree."""
        if depth == 0 or random.random() < 0.2:
            return [random.randint(0, 9)]
        
        op = random.choice(self.op_keys)
        length = random.randint(2, 4)
        
        tokens = [14, op] # '[', OP
        for _ in range(length):
            tokens.extend(self._generate_tree(depth - 1))
        tokens.append(15) # ']'
        return tokens

    def _resolve_innermost_step(self, tokens):
        """Finds the first innermost bracket, evaluates it, and returns the new sequence."""
        start_idx, end_idx = -1, -1
        for i, t in enumerate(tokens):
            if t == 14: # '['
                start_idx = i
            elif t == 15 and start_idx != -1: # ']'
                end_idx = i
                break
        
        # If no brackets are left, we are done
        if start_idx == -1:
            return tokens, False
            
        op = tokens[start_idx + 1]
        nums = tokens[start_idx + 2 : end_idx]
        
        if op == 10: res = min(nums)
        elif op == 11: res = max(nums)
        elif op == 12: res = int(np.median(nums))
        elif op == 13: res = sum(nums) % 10
        else: res = 0 # Fallback
        
        # Substitute the evaluated bracket with the result
        new_tokens = tokens[:start_idx] + [res] + tokens[end_idx + 1:]
        return new_tokens, True

    def _generate_sample(self, depth):
        """Generates the prompt and the sequential CoT trace."""
        tree = self._generate_tree(depth)
        while len(tree) == 1: # Ensure it's not just a trivial single digit
            tree = self._generate_tree(depth)
            
        prompt = tree.copy()
        trace = []
        
        # Iteratively reduce the tree until only one number remains
        current_state = tree.copy()
        while True:
            current_state, changed = self._resolve_innermost_step(current_state)
            if not changed:
                break
            trace.append(16) # '='
            trace.extend(current_state)
            
        trace.append(17) # '<EOS>'
        return prompt, trace

    def __getitem__(self, idx):
        # Shift targets for autoregressive next-token prediction
        seq = self.data[idx]
        x = seq[:-1]
        y = seq[1:]
        return x, y

    def __len__(self):
        return self.num_samples

def get_cot_listops_loaders(batch_size, train_samples=10000, test_samples=1000, train_depth=3, test_depth=5):
    """Returns dataloaders for the CoT ListOps task."""
    # Note: Sequences get very long when generating traces. 
    # Max seq len of 256 is safe for depth-3, but depth-5 may need 512.
    train_dataset = CoTListOpsDataset(train_samples, max_depth=train_depth, max_seq_len=256)
    test_dataset = CoTListOpsDataset(test_samples, max_depth=test_depth, max_seq_len=512)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset.vocab_size, train_dataset.vocab_size, 512, 100