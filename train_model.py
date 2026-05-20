# train_model.py
# Train model ANN nhận diện số viết tay MNIST
# Chạy: python train_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# 1. Load dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Đưa ảnh 28x28 về vector 784 giống logic Colab
x_train = x_train.reshape((60000, 28 * 28)).astype("float32") / 255.0
x_test = x_test.reshape((10000, 28 * 28)).astype("float32") / 255.0

# 3. One-hot label
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Model ANN mạnh hơn bản cũ nhưng vẫn là ANN, không dùng CNN
model = Sequential([
    Dense(512, activation="relu", input_shape=(28 * 28,)),
    BatchNormalization(),
    Dropout(0.25),

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.25),

    Dense(128, activation="relu"),
    Dropout(0.15),

    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Callback lưu model tốt nhất
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ModelCheckpoint("train.keras", monitor="val_accuracy", save_best_only=True)
]

# 6. Train
history = model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks
)

# 7. Đánh giá
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# 8. Lưu thêm dạng .h5 nếu anh muốn dùng
model.save("train.h5")
print("Đã lưu model: train.keras và train.h5")