import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


from configs.config import DEVICE # Evaluate function will need DEVICE for moving data
from data.dataset import get_image_transforms # For val_transforms
from utils.tta import get_basic_tensor_tta_transforms
from utils.activations import sigmoid_prob, binarize_probs
val_transforms = get_image_transforms(train=False) # Get validation transforms once

def evaluate(model, val_loader, criterion, use_tta=False):
    model.eval()
    total_loss = 0.0
    all_preds_final = []
    all_labels_final = []
    all_probs_final = []

    with torch.no_grad():
        for images_batch_tensor, metadata_batch, labels_batch in tqdm(val_loader, desc="Evaluating"):
            metadata_batch = metadata_batch.to(DEVICE)
            labels_batch_cpu = labels_batch.cpu().numpy().flatten()
            all_labels_final.extend(labels_batch_cpu)

            for i in range(images_batch_tensor.size(0)):
                current_metadata_single = metadata_batch[i:i+1]
                original_image_tensor_single = images_batch_tensor[i]

                if use_tta:
                    tta_probs_for_image = []
                    outputs = None
                    for tta_fn in get_basic_tensor_tta_transforms():
                        aug_t = tta_fn(original_image_tensor_single)
                        out = model(aug_t.unsqueeze(0).to(DEVICE), current_metadata_single)
                        # Keep first output for loss calculation consistency
                        if outputs is None:
                            outputs = out
                        tta_probs_for_image.append(sigmoid_prob(out))
                    final_prob = float(np.mean(tta_probs_for_image))
                    loss = criterion(outputs, labels_batch[i:i+1].unsqueeze(1).float().to(DEVICE))
                else:
                    outputs = model(original_image_tensor_single.unsqueeze(0).to(DEVICE), current_metadata_single)
                    loss = criterion(outputs, labels_batch[i:i+1].unsqueeze(1).float().to(DEVICE))
                    final_prob = sigmoid_prob(outputs)
                
                total_loss += loss.item()
                all_probs_final.append(final_prob)
                all_preds_final.append(binarize_probs(final_prob, threshold=0.5))

    avg_loss = total_loss / len(all_labels_final) if len(all_labels_final) > 0 else 0
    return avg_loss, all_preds_final, all_labels_final, all_probs_final

# --- Plotting Utilities ---
def plot_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png') 
    plt.close()
    print("ROC curve saved to roc_curve.png")

def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to confusion_matrix.png") 