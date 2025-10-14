from fastapi import HTTPException

def verify_token(token: str, admin_required: bool = False):
    ADMIN_TOKEN = "admin_secret_token"
    USER_TOKEN = "user_secret_token"

    if token not in [ADMIN_TOKEN, USER_TOKEN]:
        raise HTTPException(status_code=403, detail="Invalid token")
    if admin_required and token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin token required")
    return True
