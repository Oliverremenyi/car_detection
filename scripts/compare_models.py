import numpy as np
import matplotlib.pyplot as plt

# Load training histories
models = ["CNN", "ResNet50", "GoogLeNet"]
histories = {}

for model_name in models:
    history_path = f"outputs/{model_name}_history.npy"
    histories[model_name] = np.load(history_path, allow_pickle=True).item()

# Plot training/validation accuracy and loss
plt.figure(figsize=(15, 10))

# Plot Training Accuracy
plt.subplot(2, 2, 1)
for model_name in models:
    plt.plot(histories[model_name]['accuracy'], label=model_name)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Validation Accuracy
plt.subplot(2, 2, 2)
for model_name in models:
    plt.plot(histories[model_name]['val_accuracy'], label=model_name)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training Loss
plt.subplot(2, 2, 3)
for model_name in models:
    plt.plot(histories[model_name]['loss'], label=model_name)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Loss
plt.subplot(2, 2, 4)
for model_name in models:
    plt.plot(histories[model_name]['val_loss'], label=model_name)
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.show()

# Generate summary table
print("\nModel Comparison Summary:")
print("{:<10} {:<15} {:<15} {:<15}".format("Model", "Max Val Acc", "Min Val Loss", "Epochs to Converge"))
for model_name in models:
    val_acc = max(histories[model_name]['val_accuracy'])
    val_loss = min(histories[model_name]['val_loss'])
    epochs = len(histories[model_name]['val_accuracy'])
    print("{:<10} {:<15.4f} {:<15.4f} {:<15}".format(model_name, val_acc, val_loss, epochs))