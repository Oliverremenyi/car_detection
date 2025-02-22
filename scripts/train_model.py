from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(256, 256, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    directory="../data/train_processed",
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

valid_generator = ImageDataGenerator().flow_from_directory(
    directory="../data/valid_processed",
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train
checkpoint = ModelCheckpoint(
    "outputs/model_weights.h5",
#    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max'
)

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    callbacks=[checkpoint]
)