import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

# Import all architectures (Ensure CausalRecurrentLLM is added to your src.models)
from src.models import IndustryStandardLLM, RecurrentLLM, CausalRecurrentLLM, UniversalLLM
from src.data_loader import (
    get_dyck_extrapolation_loaders, 
    get_associative_recall_loaders, 
    get_listops_loaders, 
    get_cot_listops_loaders
)

def evaluate(model, testloader, criterion, device, task):
    """Dynamic evaluation loop supporting classification and generation."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            out_tensor = output[0] if isinstance(output, tuple) else output
                
            if task == "COT_LISTOPS":
                # GENERATION: Accuracy across all non-padding tokens
                loss_sum += criterion(out_tensor.view(-1, out_tensor.size(-1)), y.view(-1)).item()
                preds = out_tensor.argmax(dim=-1)
                mask = (y != 0) 
                correct += ((preds == y) & mask).sum().item()
                total += mask.sum().item()
            else:
                # CLASSIFICATION: Accuracy on the absolute final token
                logits = out_tensor[:, -1, :]
                loss_sum += criterion(logits, y).item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
            
    return loss_sum / len(testloader), 100 * correct / total

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # --- THE GAUNTLET ---
    TASKS = ["COT_LISTOPS"] # ["RECALL", "DYCK_EXTRAPOLATE", "LISTOPS", "COT_LISTOPS"]
    MODELS = ["INDUSTRY_LLM"] # ["CAUSAL_LLM", "RECURRENT_LLM", "INDUSTRY_LLM", "Universal_LLM"]
    
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
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_cot_listops_loaders(BATCH_SIZE, train_depth=3, test_depth=5)
            
        print(f"    Vocab: {vocab_size} | Classes: {num_classes} | Max SeqLen: {seq_len} | Epochs: {epochs}")
        
        for model_name in MODELS:
            # CAUSALITY GUARDRAIL
            if task == "COT_LISTOPS" and model_name in ["RECURRENT_LLM", "INDUSTRY_LLM"]:
                print(f"Skipping {model_name} on {task}: Bidirectional models violate causality on CoT.")
                continue

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

            # 3. Task-Specific Criterion
            if task == "COT_LISTOPS":
                criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding
            else:
                criterion = nn.CrossEntropyLoss() # 0 is a valid class for RECALL

            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-1)

            for epoch in range(epochs):
                model.train()
                correct, total, total_steps = 0, 0, 0.0
                
                # Ponder schedule for Universal LLM
                current_ponder_weight = 0.0
                if model_name == "Universal_LLM" and epoch >= 30:
                    current_ponder_weight = min(0.01, 0.01 * ((epoch - 30) / 25.0))
                
                for x, y in trainloader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    
                    output = model(x)
                    out_tensor, ponder_cost = (output[0], output[1]) if isinstance(output, tuple) else (output, torch.tensor(0.0, device=device))

                    # 4. Task-Specific Loss & Acc Logic
                    if task == "COT_LISTOPS":
                        loss = criterion(out_tensor.view(-1, out_tensor.size(-1)), y.view(-1))
                        preds = out_tensor.argmax(dim=-1)
                        mask = (y != 0)
                        batch_correct = ((preds == y) & mask).sum().item()
                        batch_total = mask.sum().item()
                    else:
                        logits = out_tensor[:, -1, :] # Slicing for classification
                        loss = criterion(logits, y)
                        batch_correct = (logits.argmax(1) == y).sum().item()
                        batch_total = y.size(0)

                    total_loss = loss + (current_ponder_weight * ponder_cost)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    correct += batch_correct
                    total += batch_total
                    total_steps += ponder_cost.item()
                
                if (epoch + 1) % 1 == 0:
                    _, test_acc = evaluate(model, testloader, criterion, device, task)
                    print(f"Epoch {epoch+1:02d} | Train Acc: {100*correct/total:.1f}% | Test Acc: {test_acc:.1f}%")

            master_results.append({"Task": task, "Model": model_name, "Final_Test_Acc": test_acc})
            
    print("\n" + "="*70 + "\n FINAL STANDINGS: \n" + "="*70)
    df = pd.DataFrame(master_results)
    print(df.pivot(index="Task", columns="Model", values="Final_Test_Acc").to_string())

if __name__ == "__main__":
    main()