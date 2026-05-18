"""
Standalone PCA reduction script — invoked as a subprocess by run_full_pipeline.py.

Writes output paths to a JSON file rather than stdout so that any logging
output from pca_reducer.py (or imported libraries) cannot corrupt the result.

Usage:
    python run_pca.py <text_cache> <image_cache> <text_encoder> <output_json_path>
"""
import sys
import json
sys.path.insert(0, "/app")

from services.pca_reducer import reduce_features

text_cache, image_cache, text_encoder, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pca_components        = int(sys.argv[5]) if len(sys.argv) > 5 else 300
n_text_pca_components = int(sys.argv[6]) if len(sys.argv) > 6 else None

result = reduce_features(
    text_features_path=text_cache,
    image_features_path=image_cache,
    text_encoder=text_encoder,
    n_components_img=pca_components,
    n_components_text=n_text_pca_components,
)

with open(out_path, "w") as fh:
    json.dump(list(result), fh)
