

#### **Overview**
This project demonstrates how to train a modified ResNet-50 model on the CIFAR-100 dataset using PyTorch. The workflow includes dataset preparation, model customization, training, evaluation, and saving/loading the trained model.

---

#### **Project Features**
1. **Dataset**: Uses CIFAR-100, a dataset of 100 classes of objects.
2. **Model**: Modified ResNet-50 with a custom fully connected layer.
3. **Training**: Includes custom training and evaluation loops.
4. **Metrics**: Computes accuracy and a detailed classification report using `sklearn`.
5. **Model Persistence**: Saves and reloads the trained model.

---

#### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/cifar100-resnet.git
   cd cifar100-resnet
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn
   ```

---

#### **Files**

- `main.py`: Contains the main implementation.
- `cifar100_resnet50.pth`: Saved model weights (after training).

---

#### **Usage**

1. **Run the training script**:
   ```bash
   python main.py
   ```

2. **Dataset Preparation**:
   - Downloads the CIFAR-100 dataset automatically.
   - Normalizes images to `[0, 1]` range.

3. **Model Details**:
   - Pretrained ResNet-50 from PyTorch's `torchvision.models`.
   - Adds a custom head with:
     - Linear layer (512 units).
     - ReLU activation.
     - Dropout (50%).
     - Final Linear layer (100 classes).

4. **Training**:
   - Configurable with a default of 5 epochs.
   - Uses Adam optimizer and CrossEntropyLoss.

5. **Evaluation**:
   - Computes accuracy and generates a classification report.

6. **Save and Load**:
   - Save model with:
     ```python
     torch.save(model.state_dict(), "cifar100_resnet50.pth")
     ```
   - Load model later with:
     ```python
     model.load_state_dict(torch.load("cifar100_resnet50.pth"))
     ```

---

#### **Results**
1. **Accuracy**:
   - Test accuracy displayed after evaluation.

2. **Metrics**:
   - Classification report generated using `sklearn`.

---

#### **Customizing the Code**

1. **Adjust Batch Size**:
   Change in `DataLoader`:
   ```python
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   ```

2. **Change Epochs**:
   Modify in the `train_model` function call:
   ```python
   train_model(model, train_loader, criterion, optimizer, epochs=10)
   ```

3. **Add Metrics**:
   Use additional metrics from `sklearn` or other libraries for detailed evaluation.

---

#### **Dependencies**
- Python 3.8+
- PyTorch
- torchvision
- scikit-learn

---

#### **Acknowledgments**
- **PyTorch**: For providing pretrained ResNet-50 and CIFAR-100 dataset utilities.
- **scikit-learn**: For evaluation metrics.
