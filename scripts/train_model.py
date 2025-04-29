import argparse
from model import build_cnn, build_resnet, build_googlenet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import os

parser = argparse.ArgumentParser(description='Train parking spot classification model')
parser.add_argument('--model', type=str, required=True,
                    choices=['CNN', 'ResNet50', 'GoogLeNet'])
args = parser.parse_args()


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    directory="../data/train_processed_small",
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

valid_generator = ImageDataGenerator().flow_from_directory(
    directory="../data/valid_processed_small",
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Model selection
if args.model == "CNN":
    model = build_cnn()
elif "ResNet" in args.model:
    model = build_resnet()
elif args.model == "GoogLeNet":
    model = build_googlenet()

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Training
checkpoint = ModelCheckpoint(
    f"outputs/{args.model}_weights.h5",
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=[checkpoint, reduce_lr]
)

os.makedirs("outputs", exist_ok=True)
np.save(f"outputs/{args.model}_history.npy", history.history)