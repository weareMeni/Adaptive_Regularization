import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

# Import all architectures
from src.models import IndustryStandardLLM, RecurrentLLM, CausalRecurrentLLM, UniversalLLM

# Import your dataloaders
from src.data_loader import (
    get_dyck_extrapolation_loaders, 
    get_associative_recall_loaders, 
    get_listops_loaders, 
    get_cot_listops_loaders
)

def evaluate(model, testloader, criterion, device, task):
    """Dynamic evaluation loop supporting classification and strict CoT."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            out_tensor = output[0] if isinstance(output, tuple) else output
                
            if task == "COT_LISTOPS":
                # GENERATION LOSS
                loss_sum += criterion(out_tensor.view(-1, out_tensor.size(-1)), y.view(-1)).item()
                preds = out_tensor.argmax(dim=-1)
                
                # STRICT ACCURACY: Grade the actual mathematical answer
                eos_mask = (y == 18)
                batch_idx, seq_idx = eos_mask.nonzero(as_tuple=True)
                
                if len(batch_idx) > 0:
                    ans_seq_idx = seq_idx - 1
                    batch_correct = (preds[batch_idx, ans_seq_idx] == y[batch_idx, ans_seq_idx]).sum().item()
                    correct += batch_correct
                    total += len(batch_idx) # Only count untruncated sequences
                
            else:
                # CLASSIFICATION: Dynamic sequence length slicing
                lengths = (x != 0).sum(dim=1).clamp(min=1) - 1
                logits = out_tensor[torch.arange(x.size(0)), lengths, :]
                
                loss_sum += criterion(logits, y).item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
            
    # Protect against division by zero if all test sequences were truncated
    eval_acc = (100 * correct / total) if total > 0 else 0.0
    return loss_sum / len(testloader), eval_acc

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # --- THE GAUNTLET ---
    TASKS = ["COT_LISTOPS"] # ["DYCK_EXTRAPOLATE", "COT_LISTOPS", "RECALL", "LISTOPS"]
    MODELS = ["CAUSAL_LLM"] # ["CAUSAL_LLM", "RECURRENT_LLM", "INDUSTRY_LLM". "Universal_LLM"]
    
    NUM_LAYERS = 4 
    BATCH_SIZE = 64
    DIM = 256
    
    os.makedirs("results", exist_ok=True)
    master_results = []

    print("\n" + "="*70)
    print(" THE ULTIMATE TRANSFORMER BENCHMARK SUITE")
    print("="*70)

    for task in TASKS:
        print(f"\n>>> INITIALIZING TASK: {task}")
        
        # 1. Load Data
        if task == "RECALL":
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_associative_recall_loaders(BATCH_SIZE)
        elif task == "DYCK_EXTRAPOLATE":
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_dyck_extrapolation_loaders(BATCH_SIZE, train_len=30, test_len=120)
        elif task == "LISTOPS":
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_listops_loaders(BATCH_SIZE)
        elif task == "COT_LISTOPS":
            # Make sure your src/data_loader.py has the updated CoTListOpsDataset with vocab shift and larger seq_len!
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_cot_listops_loaders(BATCH_SIZE, train_depth=3, test_depth=5)
            
        print(f"    Vocab: {vocab_size} | Classes: {num_classes} | Max SeqLen: {seq_len} | Epochs: {epochs}")
        
        for model_name in MODELS:
            if task == "COT_LISTOPS" and model_name in ["RECURRENT_LLM"]:
                print(f"Skipping {model_name} on {task}: Bidirectional models violate causality on CoT.")
                continue

            ema_grads = {}
            grokfast_alpha = 0.98
            grokfast_lamb = 2.0
            grokfast_enabled = False

            print(f"\n--- Running {model_name} on {task} ---")
            
            # 2. Instantiate Model
            if model_name == "RECURRENT_LLM":
                model = RecurrentLLM(vocab_size, num_classes, DIM, nhead=4, num_layers=NUM_LAYERS, max_seq_len=5000).to(device)
            elif model_name == "CAUSAL_LLM":
                model = CausalRecurrentLLM(vocab_size, num_classes, DIM, nhead=4, num_layers=NUM_LAYERS, max_seq_len=5000).to(device)
            elif model_name == "INDUSTRY_LLM":
                model = IndustryStandardLLM(vocab_size, num_classes, DIM, nhead=4, num_layers=NUM_LAYERS, max_seq_len=5000).to(device)
            elif model_name == "Universal_LLM":
                model = UniversalLLM(vocab_size, num_classes, DIM, nhead=4, max_steps=20, max_seq_len=5000).to(device)

            if task == "COT_LISTOPS":
                criterion = nn.CrossEntropyLoss(ignore_index=0)
            else:
                criterion = nn.CrossEntropyLoss()

            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-1)

            for epoch in range(epochs):
                model.train()
                correct, total, total_steps = 0, 0, 0.0
                
                current_ponder_weight = 0.0
                if model_name == "Universal_LLM" and epoch >= 30:
                    current_ponder_weight = min(0.01, 0.01 * ((epoch - 30) / 25.0))
                
                for x, y in trainloader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    
                    output = model(x)
                    out_tensor, ponder_cost = (output[0], output[1]) if isinstance(output, tuple) else (output, torch.tensor(0.0, device=device))

                    if task == "COT_LISTOPS":
                        loss = criterion(out_tensor.view(-1, out_tensor.size(-1)), y.view(-1))
                        preds = out_tensor.argmax(dim=-1)
                        
                        # STRICT METRIC FOR TRAINING: Only measure the final answer
                        eos_mask = (y == 18)
                        batch_idx, seq_idx = eos_mask.nonzero(as_tuple=True)
                        
                        if len(batch_idx) > 0: 
                            ans_seq_idx = seq_idx - 1
                            batch_correct = (preds[batch_idx, ans_seq_idx] == y[batch_idx, ans_seq_idx]).sum().item()
                            batch_total = len(batch_idx) 
                        else:
                            batch_correct = 0
                            batch_total = 0
                    else:
                        # CLASSIFICATION: Dynamic indexing for training
                        lengths = (x != 0).sum(dim=1).clamp(min=1) - 1
                        logits = out_tensor[torch.arange(x.size(0)), lengths, :]
                        loss = criterion(logits, y)
                        batch_correct = (logits.argmax(1) == y).sum().item()
                        batch_total = y.size(0)

                    total_loss = loss + (current_ponder_weight * ponder_cost)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if grokfast_enabled:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if param.grad is None:
                                    continue
                                    
                                if name not in ema_grads:
                                    ema_grads[name] = param.grad.clone().detach()
                                else:
                                    ema_grads[name].mul_(grokfast_alpha).add_(param.grad, alpha=1 - grokfast_alpha)
                                    
                                param.grad.add_(ema_grads[name], alpha=grokfast_lamb)

                    optimizer.step()
                    
                    correct += batch_correct
                    total += batch_total
                    total_steps += ponder_cost.item()
                
                if (epoch + 1) % 1 == 0:
                    _, test_acc = evaluate(model, testloader, criterion, device, task)
                    train_acc_display = (100*correct/total) if total > 0 else 0.0
                    print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc_display:.1f}% | Test Acc: {test_acc:.1f}%")

                if train_acc_display >= 90.0:
                    if not grokfast_enabled:
                        print("\n>>> THRESHOLD REACHED: Enabling Grokfast EMA... <<<\n")
                        grokfast_enabled = True
                else:
                    if grokfast_enabled:
                        print("\n>>> INSTABILITY DETECTED: Disabling Grokfast EMA... <<<\n")
                        grokfast_enabled = False
                        ema_grads = {}

            master_results.append({"Task": task, "Model": model_name, "Final_Test_Acc": test_acc})
            
    print("\n" + "="*70 + "\n FINAL STANDINGS: \n" + "="*70)
    df = pd.DataFrame(master_results)
    if not df.empty:
        print(df.pivot(index="Task", columns="Model", values="Final_Test_Acc").to_string())
    else:
        print("No valid experiments were run. Check your TASKS and MODELS combinations.")

if __name__ == "__main__":
    main()