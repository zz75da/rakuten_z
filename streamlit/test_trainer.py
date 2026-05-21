"""
Smoke test for the Keras model architecture inside the predict-api container.
Verifies all three encoder input shapes build and save correctly.
Run: docker exec predict-api python /app/test_trainer.py
"""
import os
import numpy as np
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

os.makedirs("artifacts", exist_ok=True)

CONFIGS = [
    # (name,       input_dim, save_path)
    ("CV",     896, "artifacts/neural_network_model.keras"),
    ("CLIP",   896, "artifacts/neural_network_model_clip.keras"),
    ("MiniLM", 768, "artifacts/neural_network_model_minilm.keras"),
]

N_CLASSES = 27

for name, input_dim, path in CONFIGS:
    print(f"\nTesting {name} model (input_dim={input_dim}, classes={N_CLASSES}) ...")
    X = np.random.random((100, input_dim)).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, 100)

    inp = Input(shape=(input_dim,))
    h = Dense(512, activation="relu")(inp)
    h = Dropout(0.45)(h)
    h = Dense(256, activation="relu")(h)
    h = Dropout(0.35)(h)
    out = Dense(N_CLASSES, activation="softmax")(h)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(X, y, epochs=1, batch_size=32, verbose=0)
    model.save(path)

    assert os.path.exists(path), f"Model file not found: {path}"
    assert model.input_shape == (None, input_dim)
    assert model.output_shape == (None, N_CLASSES)
    print(f"  OK  {path}  input={input_dim}  output={N_CLASSES}")

print("\nAll 3 encoder smoke tests passed.")
