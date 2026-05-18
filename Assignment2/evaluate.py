import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from dataset import MEGDataset
from eegnet import EEGNet

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained EEGNet")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .pth model')
    parser.add_argument('--test-path', type=str, required=True, help='Path to test data folder (or one of the test folders)')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Determine all test folders to evaluate
    # If test-path is a specific test folder (e.g., .../Cross/test1 or .../Intra/test), check its parent directory
    parent_dir = os.path.dirname(args.test_path)
    if os.path.exists(parent_dir):
        test_folders = sorted([os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
                               if d.startswith('test') and os.path.isdir(os.path.join(parent_dir, d))])
    else:
        test_folders = [args.test_path]
        
    # Fallback if empty
    if not test_folders:
        test_folders = [args.test_path]

    model = EEGNet(num_classes=4, channels=248, samples=3562).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    
    model_name = os.path.basename(args.model_path).replace('.pth', '')
    overall_results = {
        "model_path": args.model_path,
        "results": {}
    }
    
    target_names = ["Rest (0)", "Math (1)", "Memory (2)", "Motor (3)"]

    for test_folder in test_folders:
        folder_name = os.path.basename(test_folder)
        print("\n" + "="*50)
        print(f"🚀 Evaluating on test folder: {folder_name} ({test_folder})")
        print("="*50)
        
        dataset = MEGDataset(folder_path=test_folder, downsample_factor=10)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                
        # Calculate detailed metrics
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        print(f"Accuracy:  {acc*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
        
        # Generate and save Confusion Matrix plot
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {model_name} ({folder_name})')
        
        # Save the plot based on the model name and folder name
        if len(test_folders) == 1:
            plot_filename = f"{model_name}_confusion_matrix.png"
        else:
            plot_filename = f"{model_name}_{folder_name}_confusion_matrix.png"
            
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix plot saved as: {plot_filename}")
        
        # Store in JSON results dictionary
        report_dict = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
        overall_results["results"][folder_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "raw_predictions": all_preds,
            "raw_labels": all_labels
        }
        
    # Save overall JSON file
    json_filename = f"{model_name}_evaluation_results.json"
    with open(json_filename, 'w') as f:
        json.dump(overall_results, f, indent=4)
    print("\n" + "="*50)
    print(f"📁 All raw evaluation results saved to: {json_filename}")
    print("="*50)

if __name__ == '__main__':
    main()
