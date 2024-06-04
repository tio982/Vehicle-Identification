import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, label_smooth=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # Always initialize label_dist but adjust its use based on label smoothing
        self.register_buffer('label_dist', torch.ones(num_classes) * (self.epsilon / (num_classes - 1)))
        if not label_smooth:
            self.epsilon = 0  # Set epsilon to 0 if not using label smoothing

    def forward(self, inputs, targets):
        assert inputs.dim() == 2 and inputs.size(1) == self.num_classes, "Inputs must be [batch_size, num_classes]"
    
    # Ensure inputs and targets are on the correct device
        inputs, targets = inputs.to(self.device), targets.to(self.device)
    
    # Apply log softmax
        inputs = self.logsoftmax(inputs)
    
    # Create a full distribution for each target with smoothing
        true_dist = self.label_dist.clone().detach().repeat(targets.size(0), 1) # use cloned buffer to avoid in-place operations
        true_dist.scatter_(1, targets.unsqueeze(1).long(), 1.0 - self.epsilon)
    
    # Compute the cross-entropy loss
        loss = -torch.sum(true_dist * inputs, dim=1).mean()
    
        return loss



