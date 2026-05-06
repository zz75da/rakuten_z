# gate-api/app.py

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import jwt
import os
from datetime import datetime, timedelta
import secrets

app = FastAPI(title="Gate API", version="0.1")
Instrumentator().instrument(app).expose(app)

# --- Security Configuration ---
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    print(f"WARNING: Using auto-generated SECRET_KEY. Set SECRET_KEY env var in production!")

ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "180"))

# --- Users Config via environment variables ---
# Default credentials if env vars not set
DEFAULT_ADMIN_PASS = "admin_pass"
DEFAULT_USER_PASS = "user_pass"

USERS_DB = {
    "admin": {
        "password": os.getenv("ADMIN_PASSWORD", DEFAULT_ADMIN_PASS),
        "role": "admin"
    },
    "user": {
        "password": os.getenv("USER_PASSWORD", DEFAULT_USER_PASS),
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
    if not user or user["password"] != request.password:
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
