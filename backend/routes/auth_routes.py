"""Auth routes: signup, login, refresh, Email OTP verification, AngelOne credentials."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import random

from database import get_db
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user
)
from encryption import encrypt, decrypt
from email_utils import generate_otp, send_otp_email
import models
import schemas

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/signup", response_model=schemas.TokenResponse)
def signup(data: schemas.UserSignup, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        email=data.email,
        full_name=data.full_name,
        phone=data.phone,
        hashed_password=hash_password(data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return schemas.TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=schemas.UserProfile(
            id=user.id, email=user.email, full_name=user.full_name,
            phone=user.phone, is_verified=user.is_verified,
            is_active=user.is_active, has_angelone=False, created_at=user.created_at,
        ),
    )


@router.post("/login", response_model=schemas.TokenResponse)
def login(data: schemas.UserLogin, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    has_angelone = user.angelone_creds is not None and user.angelone_creds.is_configured

    return schemas.TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=schemas.UserProfile(
            id=user.id, email=user.email, full_name=user.full_name,
            phone=user.phone, is_verified=user.is_verified,
            is_active=user.is_active, has_angelone=has_angelone, created_at=user.created_at,
        ),
    )


@router.post("/refresh", response_model=schemas.TokenResponse)
def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    payload = decode_token(refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = db.query(models.User).filter(models.User.id == int(payload["sub"])).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    has_angelone = user.angelone_creds is not None and user.angelone_creds.is_configured

    return schemas.TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        user=schemas.UserProfile(
            id=user.id, email=user.email, full_name=user.full_name,
            phone=user.phone, is_verified=user.is_verified,
            is_active=user.is_active, has_angelone=has_angelone, created_at=user.created_at,
        ),
    )


@router.get("/me", response_model=schemas.UserProfile)
def get_me(user: models.User = Depends(get_current_user)):
    has_angelone = user.angelone_creds is not None and user.angelone_creds.is_configured
    return schemas.UserProfile(
        id=user.id, email=user.email, full_name=user.full_name,
        phone=user.phone, is_verified=user.is_verified,
        is_active=user.is_active, has_angelone=has_angelone, created_at=user.created_at,
    )


# ─────────── Email OTP Verification ───────────

@router.post("/email/send-otp")
def send_email_otp(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a 6-digit OTP to the user's registered email."""
    if user.is_verified:
        return {"message": "Email already verified", "already_verified": True}

    otp = generate_otp()
    user.email_otp_code = otp
    user.email_otp_expires = datetime.utcnow() + timedelta(minutes=5)
    db.commit()

    email_sent = send_otp_email(user.email, otp, user.full_name)

    response = {
        "message": f"OTP sent to {user.email}",
        "expires_in_seconds": 300,
        "email_sent": email_sent,
    }

    # If SMTP not configured, return OTP for testing (remove in production)
    if not email_sent:
        response["fallback_otp"] = otp
        response["note"] = "SMTP not configured — showing OTP for testing"

    return response


@router.post("/email/verify-otp")
def verify_email_otp(
    data: schemas.EmailVerifyOTP,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Verify the OTP sent to user's email."""
    if user.is_verified:
        return {"message": "Email already verified", "is_verified": True}

    if not user.email_otp_code:
        raise HTTPException(status_code=400, detail="No OTP requested. Send OTP first.")

    if user.email_otp_expires and datetime.utcnow() > user.email_otp_expires:
        raise HTTPException(status_code=400, detail="OTP expired. Request a new one.")

    if user.email_otp_code != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP. Please try again.")

    user.is_verified = True
    user.email_verified_at = datetime.utcnow()
    user.email_otp_code = None
    user.email_otp_expires = None
    db.commit()

    return {"message": "Email verified successfully!", "is_verified": True}


@router.get("/email/status")
def get_email_status(user: models.User = Depends(get_current_user)):
    """Check whether email is verified."""
    return {
        "is_verified": user.is_verified,
        "email": user.email,
        "verified_at": user.email_verified_at,
    }


# ─────────── AngelOne Credentials ───────────

@router.post("/angelone/credentials")
def save_angelone_creds(
    data: schemas.AngelOneCredsInput,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Verify your email first before connecting AngelOne.")

    # ─── Validate credentials by testing real login ───
    try:
        import pyotp
        from SmartApi import SmartConnect
        totp = pyotp.TOTP(data.totp_secret).now()
        smart_api = SmartConnect(data.api_key)
        login_resp = smart_api.generateSession(data.client_id, data.password, totp)

        if login_resp is None or (isinstance(login_resp, dict) and login_resp.get("status") is False):
            error_msg = "Login failed"
            if isinstance(login_resp, dict):
                error_msg = login_resp.get("message", error_msg)
            raise HTTPException(status_code=400, detail=f"AngelOne login failed: {error_msg}")

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e)
        if "Invalid" in error_str or "invalid" in error_str:
            raise HTTPException(status_code=400, detail=f"Invalid credentials: {error_str}")
        elif "TOTP" in error_str or "totp" in error_str:
            raise HTTPException(status_code=400, detail=f"TOTP secret error: {error_str}")
        else:
            raise HTTPException(status_code=400, detail=f"AngelOne connection failed: {error_str}")

    # ─── Credentials validated — save encrypted ───
    creds = db.query(models.AngelOneCredential).filter_by(user_id=user.id).first()
    if creds:
        creds.api_key_enc = encrypt(data.api_key)
        creds.client_id_enc = encrypt(data.client_id)
        creds.password_enc = encrypt(data.password)
        creds.totp_secret_enc = encrypt(data.totp_secret)
        creds.updated_at = datetime.utcnow()
    else:
        creds = models.AngelOneCredential(
            user_id=user.id,
            api_key_enc=encrypt(data.api_key),
            client_id_enc=encrypt(data.client_id),
            password_enc=encrypt(data.password),
            totp_secret_enc=encrypt(data.totp_secret),
        )
        db.add(creds)

    db.commit()
    return {"message": "AngelOne credentials verified and saved successfully!", "is_configured": True}


@router.get("/angelone/status", response_model=schemas.AngelOneCredsStatus)
def get_angelone_status(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    creds = db.query(models.AngelOneCredential).filter_by(user_id=user.id).first()
    if not creds:
        return schemas.AngelOneCredsStatus(is_configured=False)

    client_id = decrypt(creds.client_id_enc)
    masked = client_id[:2] + "****" + client_id[-2:] if len(client_id) > 4 else "****"

    return schemas.AngelOneCredsStatus(
        is_configured=creds.is_configured, client_id_masked=masked, updated_at=creds.updated_at,
    )
