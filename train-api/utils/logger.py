# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : configuration minimale du logging partagee par
# train-api (app.py et services/*).
#
# Fonctions principales :
#   - get_logger() -> Logger : configure logging.basicConfig
#     (niveau INFO, format avec timestamp) et retourne le
#     logger nomme "train-api"
#
# Dependances externes : logging (stdlib)
# ============================================================
import logging

def get_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger("train-api")
