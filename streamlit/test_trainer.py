import os
import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Create dummy data
X = np.random.random((100, 300))
y = np.random.randint(0, 10, 100)

print("🧪 Testing basic model training...")

# Simple model
inputs = Input(shape=(300,))
x = Dense(512, activation="relu")(inputs)
x = Dropout(0.5)(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train briefly
history = model.fit(X, y, epochs=1, batch_size=32, verbose=1)

# Save model
os.makedirs("artifacts", exist_ok=True)
model_path = "artifacts/neural_network_model.h5"
model.save(model_path)

print(f"✅ Model saved to: {model_path}")
print(f"✅ File exists: {os.path.exists(model_path)}")