import torch
import torch.nn as nn
import torchvision.models as models
import timm
# Using specific imports from config
from configs.config import MODEL_ARCHITECTURE, NUM_CLASSES, DEVICE, LEARNING_RATE

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

# --- Optimizer Function ---
def get_optimizer(model, learning_rate=LEARNING_RATE):
    return torch.optim.Adam(model.parameters(), lr=learning_rate) 