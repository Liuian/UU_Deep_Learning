import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MEGDataset
from eegnet import EEGNet

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / len(dataloader), 100 * correct / total

def main():
    parser = argparse.ArgumentParser(description="Train EEGNet on MEG Data")
    parser.add_argument('--train-path', type=str, default='Final Project data/Intra/train',
                        help='Path to the training folder containing .h5 files')
    parser.add_argument('--test-path', type=str, default='',
                        help='Path to the testing folder (optional)')
    parser.add_argument('--epochs', type=int, default=1, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"🚀 Using device: {device}")
    
    # 2. Load Datasets
    print(f"Loading training data from: {args.train_path}")
    train_dataset = MEGDataset(folder_path=args.train_path, downsample_factor=10)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_loader = None
    if args.test_path:
        print(f"Loading testing data from: {args.test_path}")
        test_dataset = MEGDataset(folder_path=args.test_path, downsample_factor=10)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Setup Model, Loss, Optimizer
    model = EEGNet(num_classes=4, channels=248, samples=3562).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    print("\nStarting Training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            print(f"  Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}")
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"🎯 Epoch {epoch+1} Summary | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # 5. Evaluation Loop
        if test_loader:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"📊 Epoch {epoch+1} Eval    | Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print("-" * 50)
        
    torch.save(model.state_dict(), "eegnet_trained.pth")
    print("✅ Training complete! Model saved to eegnet_trained.pth")

if __name__ == '__main__':
    main()
