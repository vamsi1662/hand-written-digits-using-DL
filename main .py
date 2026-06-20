# main.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

def train_and_save_model(save_path: str = "digit_model.keras"):
    print("📦 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train[..., np.newaxis] / 255.0
    x_test  = x_test[...,  np.newaxis] / 255.0

    print("🏗️  Building model...")

    # ── Augmentation (closes domain gap) ────────────────────────
    augment = keras.Sequential([
        layers.RandomRotation(0.1),             # handles tilted drawings
        layers.RandomTranslation(0.1, 0.1),     # handles off-center digits
        layers.RandomZoom(0.15),                # handles size variation
        layers.GaussianNoise(0.02),             # handles canvas noise
    ], name="augmentation")

    inputs = keras.Input(shape=(28, 28, 1))
    x = augment(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="mnist_cnn")
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=2, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5,
                restore_best_weights=True, verbose=1
            ),
        ],
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy : {acc*100:.2f}%")
    print(f"   Test Loss    : {loss:.4f}")

    model.save(save_path)
    print(f"💾 Model saved → {save_path}")


if __name__ == "__main__":
    train_and_save_model()