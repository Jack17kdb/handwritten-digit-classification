"""
Standalone training script — run from the project root:
    python train.py
Trains a CNN on MNIST and saves the model to models_registry/best_model.keras
"""
import os
os.makedirs("models_registry", exist_ok=True)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.tensorflow

print("Loading MNIST dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Data augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
)

# Build CNN
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

mlflow.set_experiment("mnist_cnn")

with mlflow.start_run():
    mlflow.log_params({"epochs": 10, "batch_size": 128, "optimizer": "adam"})

    model = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"\n✅ Test accuracy: {accuracy:.4f}")

    model.save("models_registry/best_model.keras")
    print("✅ Model saved to models_registry/best_model.keras")
