import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import glob
import random
import shutil
from tqdm import tqdm # For a nice progress bar

# Import metrics and plotting libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

print(f"PyTorch Version: {torch.__version__}")

# ==============================================================================
# 1. CONFIGURATION: UPDATE THESE PATHS
# ==============================================================================

# ---> STEP 1: Update this path to the folder containing your 5 class folders.
DATA_DIRECTORY = 'C:/Users/T8605/Desktop/Blood cancer/blood_cell_images'

# ---> STEP 2: Update this path to your trained ResNet-18 .pth or .pt file.
RESNET_WEIGHTS_PATH = 'C:/Users/T8605/Downloads/resnet18_blood_cell_best.pth'

# ---> STEP 3: Update this path to your trained MobileNetV2 .pth or .pt file.
MOBILENET_WEIGHTS_PATH = 'C:/Users/T8605/Downloads/mobilenet_v2_best.pth'

# --- Dataset specific parameters ---
CLASS_NAMES = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']
NUM_CLASSES = len(CLASS_NAMES)
class_map = {i: name for i, name in enumerate(CLASS_NAMES)}

# ==============================================================================
# 2. SETUP (Test Set Creation, Model Loading, Augmentations)
# ==============================================================================

# --- Helper functions from previous steps ---
def create_test_split(base_dir, class_names, test_split_ratio=0.2):
    test_dir = os.path.join(base_dir, 'test')
    if os.path.exists(test_dir): return test_dir
    print(f"Creating a new test directory...")
    os.makedirs(test_dir)
    for class_name in class_names:
        source_class_dir = os.path.join(base_dir, class_name)
        dest_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(dest_class_dir)
        images = glob.glob(os.path.join(source_class_dir, '*.jpg'))
        if not images: continue
        random.shuffle(images)
        test_images = images[:int(len(images) * test_split_ratio)]
        for img_path in test_images: shutil.move(img_path, dest_class_dir)
    print("Test set created.")
    return test_dir

def get_your_models(num_classes, resnet_path, mobilenet_path):
    print("\nLoading your trained models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ResNet-18
    resnet18 = models.resnet18(weights=None)
    resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)
    try:
        resnet18.load_state_dict(torch.load(resnet_path, map_location=device))
        print(f"OK: ResNet-18 weights loaded.")
    except Exception as e: print(f"ERROR loading ResNet-18: {e}")
    # MobileNetV2
    mobilenet_v2 = models.mobilenet_v2(weights=None)
    mobilenet_v2.classifier[1] = nn.Linear(mobilenet_v2.classifier[1].in_features, num_classes)
    try:
        mobilenet_v2.load_state_dict(torch.load(mobilenet_path, map_location=device))
        print(f"OK: MobileNetV2 weights loaded.")
    except Exception as e: print(f"ERROR loading MobileNetV2: {e}")

    resnet18.to(device).eval()
    mobilenet_v2.to(device).eval()
    return resnet18, mobilenet_v2, device

# --- Define Augmentations ---
test_time_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45), transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
original_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# ==============================================================================
# 3. FULL EVALUATION LOOP
# ==============================================================================

def perform_tta_inference(model, image_path, device, num_augmentations=4):
    """Slightly modified TTA function to return probabilities."""
    image = Image.open(image_path).convert("RGB")
    all_predictions = []
    with torch.no_grad():
        all_predictions.append(torch.nn.functional.softmax(model(original_transform(image).unsqueeze(0).to(device)), dim=1))
        for _ in range(num_augmentations - 1):
            all_predictions.append(torch.nn.functional.softmax(model(test_time_transforms(image).unsqueeze(0).to(device)), dim=1))
    avg_probabilities = torch.mean(torch.cat(all_predictions, dim=0), dim=0)
    return avg_probabilities.cpu().numpy()

def evaluate_models_on_test_set(test_dir, resnet, mobilenet, device, class_names):
    """Iterates through the entire test set to gather predictions."""
    print("\nStarting evaluation on the full test set...")
    all_image_paths = glob.glob(os.path.join(test_dir, '**', '*.jpg'), recursive=True)
    
    true_labels = []
    resnet_preds, mobilenet_preds = [], []
    resnet_probs, mobilenet_probs = [], []
    
    # Create a mapping from class name to index
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for img_path in tqdm(all_image_paths, desc="Evaluating Images"):
        true_label_name = os.path.basename(os.path.dirname(img_path))
        true_labels.append(class_to_idx[true_label_name])
        
        # Get predictions for both models
        res_prob = perform_tta_inference(resnet, img_path, device)
        mob_prob = perform_tta_inference(mobilenet, img_path, device)
        
        resnet_probs.append(res_prob)
        mobilenet_probs.append(mob_prob)
        
        resnet_preds.append(np.argmax(res_prob))
        mobilenet_preds.append(np.argmax(mob_prob))

    return np.array(true_labels), np.array(resnet_preds), np.array(mobilenet_preds), np.array(resnet_probs), np.array(mobilenet_probs)

# ==============================================================================
# 4. METRICS AND VISUALIZATION FUNCTIONS
# ==============================================================================

def display_metrics(true_labels, pred_labels, pred_probs, class_names, model_name="Model"):
    """Calculates and prints all key metrics."""
    print(f"\n{'='*30}\n📊 Metrics for {model_name}\n{'='*30}")
    
    # Accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"Accuracy: {accuracy:.4f}\n")
    
    # Classification Report (Precision, Recall, F1-Score)
    print("Classification Report:")
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    print(report)
    
    # ROC AUC Score
    roc_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr')
    print(f"ROC AUC Score (One-vs-Rest): {roc_auc:.4f}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_metrics_comparison(true_labels, resnet_preds, mobilenet_preds, class_names):
    """Creates a bar chart comparing the two models."""
    metrics = {'Model': [], 'Accuracy': [], 'F1-Score (Macro)': [], 'Recall (Macro)': []}
    
    for model_name, preds in [('ResNet-18', resnet_preds), ('MobileNetV2', mobilenet_preds)]:
        report = classification_report(true_labels, preds, target_names=class_names, output_dict=True)
        metrics['Model'].append(model_name)
        metrics['Accuracy'].append(report['accuracy'])
        metrics['F1-Score (Macro)'].append(report['macro avg']['f1-score'])
        metrics['Recall (Macro)'].append(report['macro avg']['recall'])
        
    df = pd.DataFrame(metrics)
    df.plot(x='Model', y=['Accuracy', 'F1-Score (Macro)', 'Recall (Macro)'], kind='bar', figsize=(12, 7), rot=0)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_roc_curves(true_labels, resnet_probs, mobilenet_probs, class_names):
    """Plots ROC curves for each class for both models."""
    true_labels_bin = label_binarize(true_labels, classes=range(len(class_names)))
    
    plt.figure(figsize=(12, 9))
    
    # Plot for ResNet
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], resnet_probs[:, i])
        plt.plot(fpr, tpr, linestyle='-', label=f'ResNet-18 - {class_name}')

    # Plot for MobileNet
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_labels_bin[:, i], mobilenet_probs[:, i])
        plt.plot(fpr, tpr, linestyle='--', label=f'MobileNetV2 - {class_name}')

    plt.plot([0, 1], [0, 1], 'k:', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Multi-Class ROC Curves', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    if any(p.startswith('path/to/') for p in [DATA_DIRECTORY, RESNET_WEIGHTS_PATH, MOBILENET_WEIGHTS_PATH]):
        print("ERROR: Please update the placeholder paths in the 'CONFIGURATION' section before running.")
    else:
        # Step 1: Setup environment
        test_dir = create_test_split(DATA_DIRECTORY, CLASS_NAMES)
        resnet_model, mobilenet_model, device = get_your_models(NUM_CLASSES, RESNET_WEIGHTS_PATH, MOBILENET_WEIGHTS_PATH)
        
        # Step 2: Run evaluation on the entire test set
        true_labels, resnet_preds, mobilenet_preds, resnet_probs, mobilenet_probs = evaluate_models_on_test_set(
            test_dir, resnet_model, mobilenet_model, device, CLASS_NAMES
        )
        
        # Step 3: Display detailed metrics and confusion matrix for each model
        display_metrics(true_labels, resnet_preds, resnet_probs, CLASS_NAMES, model_name="ResNet-18")
        display_metrics(true_labels, mobilenet_preds, mobilenet_probs, CLASS_NAMES, model_name="MobileNetV2")
        
        # Step 4: Display comparison visualizations
        plot_metrics_comparison(true_labels, resnet_preds, mobilenet_preds, CLASS_NAMES)
        plot_roc_curves(true_labels, resnet_probs, mobilenet_probs, CLASS_NAMES)
        
        print("\nEvaluation complete.")