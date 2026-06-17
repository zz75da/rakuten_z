# gate-api/app.py
# ============================================================
# RESUME DU MODULE
# ------------------------------------------------------------
# Role : microservice FastAPI d'authentification centralise
# (JWT) utilise par Airflow (DAG train_dag_v7), train-api et
# predict-api pour proteger leurs endpoints sensibles.
#
# Fonctions principales :
#   - verify_token(authorization) -> dict : dependance FastAPI
#     qui decode/valide un JWT "Bearer <token>" (HS256)
#   - POST /login : verifie username/password contre USERS_DB et
#     retourne un JWT signe (role + expiration)
#   - GET / : info basique sur l'API (version, endpoints)
#   - GET /health : etat du service + nb d'utilisateurs configures
#   - POST /validate-token : valide un JWT et renvoie son contenu
#     (username, role, expiration) - utilise par les autres services
#   - GET /user-info : retourne les infos de l'utilisateur courant
#     (protege par verify_token)
#
# Variables / constantes importantes :
#   - SECRET_KEY (env SECRET_KEY, sinon genere aleatoirement -
#     avertissement si non defini)
#   - ALGORITHM = "HS256"
#   - TOKEN_EXPIRE_MINUTES (env TOKEN_EXPIRE_MINUTES, def. 180)
#   - USERS_DB : utilisateurs "admin"/"user" avec mots de passe
#     definis via ADMIN_PASSWORD / USER_PASSWORD
#
# En docker-compose, SECRET_KEY / ADMIN_PASSWORD / USER_PASSWORD
# proviennent de .env (JWT_SECRET_KEY, GATE_API_ADMIN_PASSWORD,
# GATE_API_USER_PASSWORD - non commite). Les valeurs "default_secret"
# / "admin_pass" / "user_pass" ci-dessous ne sont que des valeurs
# de repli pour les tests/dev hors docker.
#
# Dependances externes : fastapi, pydantic, prometheus_fastapi_instrumentator,
# pyjwt
# ============================================================

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import jwt
import os
import sys
from datetime import datetime, timedelta
import secrets
import bcrypt

app = FastAPI(title="Gate API", version="0.1")
Instrumentator().instrument(app).expose(app)

# --- Security Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    print("WARNING: SECRET_KEY not set — using a random key. Tokens will not survive restarts. Set JWT_SECRET_KEY in .env!", file=sys.stderr)
elif SECRET_KEY == "default_secret":
    print("CRITICAL: SECRET_KEY is the insecure default 'default_secret'. Set a strong JWT_SECRET_KEY in .env!", file=sys.stderr)

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "180"))

# --- Users Config via environment variables ---
# Passwords are bcrypt-hashed at startup; plaintext never stored after init.
# Fallback values are for dev/test only — override via ADMIN_PASSWORD / USER_PASSWORD in .env.
def _hash(plaintext: str) -> bytes:
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt())

def _verify(plaintext: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plaintext.encode(), hashed)

USERS_DB = {
    "admin": {
        "password_hash": _hash(os.getenv("ADMIN_PASSWORD", "admin_pass")),
        "role": "admin"
    },
    "user": {
        "password_hash": _hash(os.getenv("USER_PASSWORD", "user_pass")),
        "role": "user"
    }
}

# --- Pydantic models ---
class LoginRequest(BaseModel):
    username: str
    password: str

# --- JWT token verification dependency ---
async def verify_token(authorization: str = Header(...)):
    """Dependency to verify JWT token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Login endpoint ---
@app.post("/login")
def login(request: LoginRequest):
    if not request.username or not request.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    
    user = USERS_DB.get(request.username)
    if not user or not _verify(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    expires = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": request.username,
        "username": request.username,
        "role": user["role"],
        "exp": expires,
        "iat": datetime.utcnow()
    }

    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except Exception:
        raise HTTPException(status_code=500, detail="Token generation failed")

    return {
        "token": encoded_jwt,
        "role": user["role"],
        "expires_in": TOKEN_EXPIRE_MINUTES * 60,
        "token_type": "bearer"
    }

# --- Root and health endpoints ---
@app.get("/")
def root():
    return {
        "status": "API up and running",
        "version": "0.2",
        "endpoints": ["/login", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "gate-api",
        "authentication": "jwt",
        "users_configured": len(USERS_DB)
    }

# --- Token validation endpoint for other services ---
@app.post("/validate-token")
async def validate_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "valid": True,
            "username": payload["username"],
            "role": payload["role"],
            "expires": payload["exp"]
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Token validation failed")

# --- User info endpoint ---
@app.get("/user-info")
async def get_user_info(user: dict = Depends(verify_token)):
    return {
        "username": user["username"],
        "role": user["role"],
        "authenticated": True
    }
