from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

MODELS = ["CNN", "ResNet50", "GoogLeNet"]

# Configure test generator
test_datagen = ImageDataGenerator().flow_from_directory(
    directory="../data/test_processed_small",
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Dictionary to store all histories (if available)
histories = {}


def evaluate_model(model_name):
    # Load model
    model_path = f"outputs/{model_name}_weights.h5"
    model = load_model(model_path)

    # Evaluate
    test_loss, test_acc = model.evaluate(test_datagen)
    print(f"\n{'=' * 40}")
    print(f"Evaluation results for {model_name}:")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Predict
    y_pred = (model.predict(test_datagen) > 0.5).astype(int)
    y_true = test_datagen.classes

    # Generate metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save metrics to file
    with open(f"outputs/{model_name}_metrics.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(y_true, y_pred)))


if __name__ == "__main__":
    # Create outputs directory if needed
    os.makedirs("outputs", exist_ok=True)

    # Evaluate all models
    for model_name in MODELS:
        evaluate_model(model_name)

    plt.figure(figsize=(10, 6))
    for model_name in MODELS:
        history = np.load(f"outputs/{model_name}_history.npy", allow_pickle=True).item()
        plt.plot(history['val_accuracy'], label=f'{model_name} Val Accuracy')
    plt.title('Model Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("outputs/accuracy_comparison.png")