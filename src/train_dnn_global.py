import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

DATA_PATH = "datasets/global_dataset.npz"
RESULT_DIR = "results"
EPOCHS = 30
BATCH_SIZE = 256

data = np.load(DATA_PATH)

X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]

model = models.Sequential([
    layers.Input(shape=(26,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

test_loss, test_mae = model.evaluate(X_test, y_test)
print("Test MAE:", test_mae)

os.makedirs(RESULT_DIR, exist_ok=True)
model.save(os.path.join(RESULT_DIR, "global_dnn_model"))