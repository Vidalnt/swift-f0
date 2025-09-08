import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiftF0Loss(nn.Module):
    """
    Combined loss function for SwiftF0 training.
    
    Combines classification loss (cross-entropy) and regression loss (L1 on log frequencies).
    """
    
    def __init__(self, classification_weight: float = 1.0, regression_weight: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            classification_weight: Weight for classification loss
            regression_weight: Weight for regression loss
        """
        super(SwiftF0Loss, self).__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                classification_logits: torch.Tensor, 
                regression_output: torch.Tensor,
                classification_targets: torch.Tensor,
                regression_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the combined loss.
        
        Args:
            classification_logits: Output logits from classification head (batch_size, n_bins)
            regression_output: Output from regression head (batch_size, 1)
            classification_targets: Target probability distributions (batch_size, n_bins)
            regression_targets: Target log frequencies (batch_size, 1)
            
        Returns:
            Combined loss value
        """
        # Classification loss (KL divergence or cross-entropy with soft targets)
        # Convert soft targets to hard targets for cross-entropy
        # Find the index of the maximum probability in each target distribution
        hard_targets = torch.argmax(classification_targets, dim=1)
        classification_loss = self.ce_loss(classification_logits, hard_targets)
        
        # Alternative: Use KL divergence for soft targets
        # classification_loss = F.kl_div(F.log_softmax(classification_logits, dim=1), 
        #                               classification_targets, reduction='batchmean')
        
        # Regression loss (L1 on log frequencies)
        regression_loss = F.l1_loss(regression_output, regression_targets)
        
        # Combined loss
        total_loss = (self.classification_weight * classification_loss + 
                     self.regression_weight * regression_loss)
        
        return total_loss, classification_loss, regression_loss

# Function to create loss instance
def create_loss(**kwargs):
    return SwiftF0Loss(**kwargs)