# Training and Evaluation Instructions

Always start by running the **Intra-subject** pipeline first. Intra-subject classification (training and testing on the exact same person's brain) acts as a sanity check. If the model achieves good accuracy here, it means the preprocessing and neural network are functioning correctly.

*Cross-subject* classification is naturally much harder because every human brain has a slightly different shape and neural mapping, so you expect the accuracy to drop when testing on unseen subjects.

Follow this exact workflow when you connect to your RunPod instance:

### Step 1: Train your Intra Model
Run the training script pointing to the Intra train folder. 
```bash
python train.py --train-path "/workspace/Final Project data/Intra/train" --epochs 30
```
*After this finishes, it will save the model as `eegnet_trained.pth`. Rename it so you don't accidentally overwrite it during cross-subject training!*
```bash
mv eegnet_trained.pth intra_model.pth
```

### Step 2: Evaluate your Intra Model
Use the evaluation script to calculate detailed metrics (Accuracy, Precision, Recall, F1) on the Intra test dataset.
```bash
python evaluate.py --model-path "intra_model.pth" --test-path "/workspace/Final Project data/Intra/test"
```

### Step 3: Train your Cross Model
Now, train a new model on the Cross-subject data.
```bash
python train.py --train-path "/workspace/Final Project data/Cross/train" --epochs 30
```
*Rename the output file again:*
```bash
mv eegnet_trained.pth cross_model.pth
```

### Step 4: Evaluate your Cross Model (on unseen subjects)
You can evaluate this model on the unseen subjects in `test1`, `test2`, or `test3`.
```bash
python evaluate.py --model-path "cross_model.pth" --test-path "/workspace/Final Project data/Cross/test1"
```

*This script will print out a detailed table showing exactly how well the model predicted Rest vs Math vs Memory vs Motor tasks, providing all the metrics you need for your final report!*
