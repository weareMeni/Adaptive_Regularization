import torch

def compute_residual_covariance_metrics(residual_tensor):
    if residual_tensor.dim() == 3:
        residual_tensor = residual_tensor.view(-1, residual_tensor.size(-1))
        
    n = residual_tensor.size(0)
    centered = residual_tensor - residual_tensor.mean(dim=0, keepdim=True)
    
    # Compute the Covariance Matrix
    cov_matrix = (centered.t() @ centered) / (n - 1)
    
    # Extract Eigenvalues
    eigenvalues = torch.linalg.eigvalsh(cov_matrix)
    
    # CLAMP THE MINIMUM EIGENVALUE
    # If the manifold collapses and floats drift negative, bound it to 1e-9
    min_eig = max(eigenvalues[0].item(), 1e-9)
    max_eig = eigenvalues[-1].item()
    
    cond_num = max_eig / min_eig
    
    return min_eig, cond_num