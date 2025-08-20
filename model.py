import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for Focal Loss
import torchvision.models as models
import timm
# Using specific imports from config
from config import MODEL_ARCHITECTURE, NUM_CLASSES, DEVICE, CLASS_WEIGHTS, LEARNING_RATE, LOSS_FUNCTION_TYPE, FOCAL_LOSS_ALPHA, FOCAL_LOSS_GAMMA, FOCAL_LOSS_REDUCTION

# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss from Lin et al. (2017)
    It measures the loss between raw logits (input) and binary targets.
    """
    def __init__(self, alpha=0.864, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # Probability of the true class
        
        # Calculate a_t for alpha factor
        # alpha_t = alpha if target = 1, (1-alpha) if target = 0
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_factor * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

# --- Metadata Fusion Model ---
class MetadataMelanomaModel(nn.Module):
    """Model combining image features (CNN) and tabular metadata (MLP)."""
    def __init__(self, num_metadata_features, pretrained=True, 
                 cnn_dropout=0.0, mlp_hidden_dims=[128, 64], mlp_dropout=0.3):
        super().__init__()
        
        # Image Branch - uses MODEL_ARCHITECTURE from config
        self.image_backbone = timm.create_model(MODEL_ARCHITECTURE, pretrained=pretrained, num_classes=0)
        self.num_image_features = self.image_backbone.num_features
        self.cnn_dropout = nn.Dropout(cnn_dropout) if cnn_dropout > 0 else nn.Identity()

        # Metadata Branch MLP
        layers = []
        input_dim = num_metadata_features
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # Apply dropout to all but last hidden layer output
            if i < len(mlp_hidden_dims) - 1: 
                layers.append(nn.Dropout(mlp_dropout))
            input_dim = hidden_dim # Output dim becomes input for next layer
        self.metadata_mlp = nn.Sequential(*layers)
        self.num_mlp_output_features = input_dim 

        # Final Classifier (on concatenated features)
        classifier_input_features = self.num_image_features + self.num_mlp_output_features
        self.final_classifier = nn.Linear(classifier_input_features, NUM_CLASSES)
        
    def forward(self, image_input, metadata_input):
        # Process image
        image_features = self.image_backbone(image_input)
        image_features = self.cnn_dropout(image_features)
        
        # Process metadata
        metadata_features = self.metadata_mlp(metadata_input)
        
        # Concatenate features
        combined_features = torch.cat((image_features, metadata_features), dim=1)
        
        # Final classification
        output = self.final_classifier(combined_features)
        return output

# --- Model Instantiation Function ---
def get_model(num_metadata_features):
    """Instantiates the metadata fusion model.
    Args:
        num_metadata_features (int): Number of features in the metadata input.
    Returns:
        torch.nn.Module: The instantiated model, moved to DEVICE.
    """
    print(f"Creating metadata fusion model '{MODEL_ARCHITECTURE}' with {num_metadata_features} metadata features.")
    model = MetadataMelanomaModel(num_metadata_features=num_metadata_features)
    model = model.to(DEVICE)
    return model

# --- Criterion and Optimizer Functions ---
def get_criterion(loss_type=LOSS_FUNCTION_TYPE):
    # malignant proportion = 0.136, so alpha_pos=0.864
    return FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, reduction=FOCAL_LOSS_REDUCTION)

def get_optimizer(model, learning_rate=LEARNING_RATE):
    return torch.optim.Adam(model.parameters(), lr=learning_rate) 