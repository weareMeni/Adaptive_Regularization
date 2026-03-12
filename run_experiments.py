import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os

# Import all three distinct architectures
from src.models import IndustryStandardLLM, RecurrentLLM, UniversalLLM
from src.data_loader import get_dyck_extrapolation_loaders, get_associative_recall_loaders, get_listops_loaders

def evaluate(model, testloader, criterion, device):
    """Simplified evaluation loop for the master script."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            
            # Handle models that return residuals or not
            out = model(x)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out
                
            loss_sum += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            
    return loss_sum / len(testloader), 100 * correct / total

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # --- THE GAUNTLET ---
    TASKS = ["LISTOPS", "DYCK_EXTRAPOLATE"] # ["RECALL", "DYCK_EXTRAPOLATE", "LISTOPS"]
    MODELS = ["Universal_LLM", "RECURRENT_LLM"] 
    
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
        
        # Load the appropriate dataset
        if task == "RECALL":
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_associative_recall_loaders(BATCH_SIZE)
        elif task == "DYCK_EXTRAPOLATE":
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_dyck_extrapolation_loaders(BATCH_SIZE, train_len=30, test_len=120)
        elif task == "LISTOPS":
            trainloader, testloader, vocab_size, num_classes, seq_len, epochs = get_listops_loaders(BATCH_SIZE)
            
        print(f"    Vocab: {vocab_size} | Classes: {num_classes} | Max SeqLen: {seq_len} | Epochs: {epochs}")
        
        for model_name in MODELS:
            print(f"\n--- Running {model_name} on {task} ---")
            
            # Dynamically instantiate the correct model architecture
            if model_name == "RECURRENT_LLM":
                model = RecurrentLLM(vocab_size, num_classes, DIM, nhead=4, num_layers=NUM_LAYERS, max_seq_len=5000).to(device)
            elif model_name == "INDUSTRY_LLM":
                model = IndustryStandardLLM(vocab_size, num_classes, DIM, nhead=4, num_layers=NUM_LAYERS, max_seq_len=5000).to(device)
            elif model_name == "Universal_LLM":
                model = UniversalLLM(vocab_size, num_classes, DIM, nhead=4, max_steps=20, max_seq_len=5000).to(device)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
                
            if device.type == "cuda":
                print("Compiling model for CUDA...")

                cache_dir = "/iopsstor/scratch/cscs/course_00151/torch_cache"
                os.makedirs(cache_dir, exist_ok=True)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
                
                model = torch.compile(model, mode="reduce-overhead")
            else:
                print("Running locally on Mac/CPU. Skipping torch.compile.")
            
            optimizer = optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-1)
            criterion = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                model.train()
                correct, total = 0, 0
                
                for x, y in trainloader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    
                    out, ponder_cost = model(x)
                    loss = criterion(out, y)

                    target_penalty = 0.01
                    ponder_weight = min(target_penalty, target_penalty * (epoch / 25.0))

                    total_loss = loss + (ponder_weight * ponder_cost)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    correct += (out.argmax(1) == y).sum().item() 
                    total += y.size(0)
                
                # Print stats every epoch, or adjust modulo to print less frequently
                if (epoch + 1) % 1 == 0 or epoch == epochs - 1 or epoch == 0:
                    test_loss, test_acc = evaluate(model, testloader, criterion, device)
                    print(f"Epoch {epoch+1:02d} | Train Acc: {100*correct/total:.1f}% | Test Acc: {test_acc:.1f}%")

            master_results.append({
                "Task": task,
                "Model": model_name,
                "Final_Test_Acc": test_acc
            })
            
    print("\n" + "="*70)
    print(" SUITE COMPLETE. FINAL STANDINGS:")
    print("="*70)
    
    df = pd.DataFrame(master_results)
    # Pivot for a beautiful side-by-side comparison table
    pivot_df = df.pivot(index="Task", columns="Model", values="Final_Test_Acc")
    
    # Reorder columns logically for display
    available_models = [m for m in MODELS if m in pivot_df.columns]
    pivot_df = pivot_df[available_models]
    print(pivot_df.to_string())
    
    df.to_csv("results/ultimate_benchmark_suite.csv", index=False)
    print("\nResults saved to results/ultimate_benchmark_suite.csv")

if __name__ == "__main__":
    main()