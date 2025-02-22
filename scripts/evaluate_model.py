from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from train_model import history

model = load_model("outputs/model_weights.h5")

test_datagen = ImageDataGenerator().flow_from_directory(
    directory="../data/test_processed",
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# evaluate
test_loss, test_acc = model.evaluate(test_datagen)
print(f"Test accuracy: {test_acc}")

# predict
y_pred = (model.predict(test_datagen) > 0.5).astype(int)
y_true = test_datagen.classes

# metrics
print(classification_report(y_true, y_pred))
print("Confusion matrix:")
conf_matrix = confusion_matrix(y_true, y_pred)
print(conf_matrix)

# plot confusion matrix
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("outputs/accuracy.png")