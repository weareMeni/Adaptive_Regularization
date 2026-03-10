import torch
import torch.nn as nn
from src.metrics import compute_residual_covariance_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    return total_loss / total, 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_residuals = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, residuals = model(inputs, return_residual=True)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_residuals.append(residuals.cpu())
            
    # Compute HDS metrics on the collected test-set residuals
    full_residual_tensor = torch.cat(all_residuals, dim=0)
    min_eig, cond_num = compute_residual_covariance_metrics(full_residual_tensor)
            
    return total_loss / total, 100. * correct / total, min_eig, cond_num