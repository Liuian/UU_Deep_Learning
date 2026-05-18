import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from dataset import MEGDataset
from eegnet import EEGNet

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained EEGNet")
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained .pth model')
    parser.add_argument('--test-path', type=str, required=True, help='Path to test data folder')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"Loading testing data from: {args.test_path}")
    dataset = MEGDataset(folder_path=args.test_path, downsample_factor=10)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    model = EEGNet(num_classes=4, channels=248, samples=3562).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Calculate detailed metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print("\n" + "="*45)
    print("🏆 PERFORMANCE RESULTS")
    print("="*45)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Rest (0)", "Math (1)", "Memory (2)", "Motor (3)"], zero_division=0))

if __name__ == '__main__':
    main()
