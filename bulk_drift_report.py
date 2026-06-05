#!/usr/bin/env python3
"""
Bulk drift report: batch-predicts all X_test rows using the CV (TF-IDF+PCA+Keras) model,
then generates an Evidently drift report comparing test predictions vs training reference.
"""
import os, sys, gc
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime, timezone

ARTIFACTS  = Path("/app/data/artifacts")
REPORT_DIR = ARTIFACTS / "drift_reports"
REF_PATH   = ARTIFACTS / "drift_reference.csv"
TEST_PATH  = Path("/app/data/X_test_update.csv")


def _load_model(keras_path):
    """Load the late-fusion model safely: weights-only loader first, standard fallback."""
    import zipfile, json as _json, h5py, io
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (Dense, Dropout, Concatenate, Multiply,
                                         Add, Softmax, LayerNormalization, Lambda)
    from tensorflow.keras.regularizers import l2 as keras_l2

    try:
        with zipfile.ZipFile(str(keras_path)) as zf:
            config   = _json.loads(zf.read('config.json'))
            w_bytes  = zf.read('model.weights.h5')

        def _find(n):
            for l in config.get('config', {}).get('layers', []):
                if l.get('config', {}).get('name') == n:
                    return l
            return None

        def _units(n): l = _find(n); return l['config']['units'] if l else None
        def _rate(n):  l = _find(n); return l['config']['rate']  if l else 0.0
        def _has(n):   return _find(n) is not None
        def _l2(n):
            l = _find(n)
            if not l: return 0.0
            kr = l['config'].get('kernel_regularizer')
            return kr.get('config', {}).get('l2', 0.0) if kr else 0.0
        def _in_shape(n):
            l = _find(n)
            if not l: return None
            bs = l.get('build_config', {}).get('input_shape')
            return bs[-1] if bs else None

        input_dim = config['config']['layers'][0]['config']['batch_input_shape'][-1]
        text_dim  = _in_shape('text_dense1') or (input_dim - 256)
        img_dim   = input_dim - text_dim
        hidden_1  = _units('text_dense1')  or 512
        hidden_2  = _units('image_dense1') or 256
        n_classes = _units('text_logits')  or 27
        drop_1    = _rate('text_drop1')
        drop_2    = _rate('image_drop1')
        l2_val    = _l2('text_dense1')
        use_ln    = _has('text_ln1')
        _reg      = keras_l2(l2_val) if l2_val > 0 else None

        _td  = int(text_dim)
        inp      = Input(shape=(input_dim,), name='input_1')
        text_inp  = Lambda(lambda x, d=_td: x[:, :d],  output_shape=(text_dim,),  name='text_split')(inp)
        image_inp = Lambda(lambda x, d=_td: x[:, d:],  output_shape=(img_dim,),   name='image_split')(inp)

        t = Dense(hidden_1, activation='relu', kernel_regularizer=_reg, name='text_dense1')(text_inp)
        if use_ln: t = LayerNormalization(name='text_ln1')(t)
        t = Dropout(drop_1, name='text_drop1')(t)
        t_logits = Dense(n_classes, name='text_logits')(t)

        i = Dense(hidden_2, activation='relu', kernel_regularizer=_reg, name='image_dense1')(image_inp)
        if use_ln: i = LayerNormalization(name='image_ln1')(i)
        i = Dropout(drop_2, name='image_drop1')(i)
        i_logits = Dense(n_classes, name='image_logits')(i)

        t_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True), output_shape=(1,), name='t_mean')(t_logits)
        i_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True), output_shape=(1,), name='i_mean')(i_logits)
        branch  = Concatenate(name='branch_summary')([t_mean, i_mean])
        alpha   = Dense(1, activation='sigmoid', kernel_initializer='zeros', name='fusion_alpha')(branch)

        t_probs   = Softmax(name='text_softmax')(t_logits)
        i_probs   = Softmax(name='image_softmax')(i_logits)
        alpha_inv = Lambda(lambda x: 1.0 - x, output_shape=(1,), name='alpha_inv')(alpha)
        out = Add(name='fused_probs')([
            Multiply(name='alpha_text') ([alpha,     t_probs]),
            Multiply(name='alpha_image')([alpha_inv, i_probs]),
        ])
        model = Model(inp, out)

        with h5py.File(io.BytesIO(w_bytes), 'r') as hf:
            h5_by_key = {}
            if 'layers' in hf:
                for lkey in hf['layers']:
                    grp = hf[f'layers/{lkey}']
                    if 'vars' in grp:
                        h5_by_key[lkey] = [grp[f'vars/{vk}'][:] for vk in sorted(grp['vars'].keys())]

            type_counters = {}
            name_to_h5key = {}
            for l in config.get('config', {}).get('layers', []):
                cls   = l.get('class_name', '')
                lname = l.get('config', {}).get('name', '')
                if cls == 'Dense':                prefix = 'dense'
                elif cls == 'LayerNormalization': prefix = 'layer_normalization'
                else:                             continue
                cnt   = type_counters.get(prefix, 0)
                h5key = prefix if cnt == 0 else f'{prefix}_{cnt}'
                type_counters[prefix] = cnt + 1
                if h5key in h5_by_key:
                    name_to_h5key[lname] = h5key

            for layer in model.layers:
                if layer.name in name_to_h5key:
                    w = h5_by_key[name_to_h5key[layer.name]]
                    if len(w) == len(layer.get_weights()):
                        layer.set_weights(w)

        print("  Loaded via weights-only loader")
        return model, int(text_dim)

    except Exception as e1:
        print(f"  weights-only loader failed ({e1}), trying standard load...")
        try:
            import keras as _keras
            _keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
        import tensorflow as tf
        m = tf.keras.models.load_model(str(keras_path), compile=False)
        text_dim = m.input_shape[1] - 256  # fallback estimate
        print("  Loaded via standard load")
        return m, int(text_dim)


def main():
    if not TEST_PATH.exists():
        sys.exit(f"ERROR: {TEST_PATH} not found — copy it into the container first")
    if not REF_PATH.exists():
        sys.exit(f"ERROR: {REF_PATH} not found")

    print("Loading artifacts...")
    label_encoder   = pickle.load(open(ARTIFACTS / "label_encoder.pkl",   "rb"))
    pca_image       = pickle.load(open(ARTIFACTS / "pca_image.pkl",       "rb"))
    text_vectorizer = pickle.load(open(ARTIFACTS / "text_vectorizer.pkl", "rb"))
    pca_text        = pickle.load(open(ARTIFACTS / "pca_text.pkl",        "rb"))
    print(f"  pca_text components: {pca_text.n_components_}  |  pca_image components: {pca_image.n_components_}")

    keras_path = ARTIFACTS / "neural_network_model.keras"
    if not keras_path.exists():
        keras_path = ARTIFACTS / "neural_network_model.h5"
    print(f"Loading model from {keras_path.name} ...")
    model, text_dim = _load_model(keras_path)
    input_dim = model.input_shape[1]
    n_img     = pca_image.n_components_
    print(f"  input_dim={input_dim}  text_dim={text_dim}  n_img={n_img}")

    print(f"Reading {TEST_PATH} ...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"  {len(test_df)} rows")

    print("Vectorising text ...")
    desig = test_df["designation"].fillna("").apply(lambda t: " ".join(t.lower().split()))
    bow   = text_vectorizer.transform(desig.tolist())
    text_feats = pca_text.transform(bow.toarray()).astype(np.float32)
    img_feats  = np.zeros((len(test_df), n_img), dtype=np.float32)
    combined   = np.hstack([text_feats, img_feats])
    print(f"  combined shape: {combined.shape}")

    del bow, text_feats, img_feats
    gc.collect()

    print("Running batch predictions ...")
    probs  = model.predict(combined, batch_size=512, verbose=1)
    preds  = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(preds).astype(int)
    del combined, probs
    gc.collect()

    current_df = pd.DataFrame({
        "designation": test_df["designation"].fillna(""),
        "prdtypecode":  labels,
    })
    print(f"  Class distribution (top 10):\n{current_df['prdtypecode'].value_counts().head(10)}")

    # Save predictions CSV alongside the report so it can be reused
    pred_csv = ARTIFACTS / "bulk_test_predictions.csv"
    current_df.to_csv(pred_csv, index=False)
    print(f"  Predictions saved: {pred_csv}")

    print("Loading reference dataset ...")
    reference_df = pd.read_csv(REF_PATH)
    reference_df = reference_df[["designation", "prdtypecode"]].copy()
    reference_df["prdtypecode"] = reference_df["prdtypecode"].astype(int)

    print(f"  Reference rows: {len(reference_df)}  |  Current rows: {len(current_df)}")

    # Cast prdtypecode to str so Evidently treats it as categorical (not continuous numerical).
    for df in (reference_df, current_df):
        df["prdtypecode"] = df["prdtypecode"].astype(str)
        # Derived text features: give Evidently meaningful numerical distributions for
        # designation instead of "count 1 everywhere" (each designation is unique text).
        df["designation_word_count"] = df["designation"].fillna("").str.split().str.len()
        df["designation_char_count"] = df["designation"].fillna("").str.len()

    print("Running Evidently DataDriftPreset ...")
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    column_mapping = ColumnMapping(
        target=None,
        numerical_features=["designation_word_count", "designation_char_count"],
        categorical_features=["prdtypecode"],
        text_features=["designation"],
    )
    all_cols = ["designation", "prdtypecode", "designation_word_count", "designation_char_count"]
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[all_cols],
        current_data=current_df[all_cols],
        column_mapping=column_mapping,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = REPORT_DIR / f"drift_bulk_test_{ts}.html"
    report.save_html(str(path))
    size_kb = path.stat().st_size // 1024
    print(f"\nDrift report saved: {path}  ({size_kb} KB)")
    return str(path)


if __name__ == "__main__":
    result = main()
    print(f"\nDone: {result}")
