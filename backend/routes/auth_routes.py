"""Auth routes: signup, login, refresh, Aadhaar verification, AngelOne credentials."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
import random
import hashlib

from database import get_db
from auth import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user
)
from encryption import encrypt, decrypt
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


# ─────────── Aadhaar Verification ───────────

@router.post("/aadhaar/send-otp")
def send_aadhaar_otp(
    data: schemas.AadhaarSendOTP,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    aadhaar_hash = hashlib.sha256(data.aadhaar_number.encode()).hexdigest()
    otp = f"{random.randint(100000, 999999)}"
    expires = datetime.utcnow() + timedelta(minutes=5)

    aadhaar = db.query(models.AadhaarVerification).filter_by(user_id=user.id).first()
    if aadhaar:
        aadhaar.aadhaar_number_hash = aadhaar_hash
        aadhaar.aadhaar_last4 = data.aadhaar_number[-4:]
        aadhaar.otp_code = otp
        aadhaar.otp_expires = expires
        aadhaar.is_verified = False
    else:
        aadhaar = models.AadhaarVerification(
            user_id=user.id,
            aadhaar_number_hash=aadhaar_hash,
            aadhaar_last4=data.aadhaar_number[-4:],
            otp_code=otp,
            otp_expires=expires,
        )
        db.add(aadhaar)

    db.commit()
    return {
        "message": "OTP sent to Aadhaar-linked mobile number",
        "otp_hint": otp,
        "expires_in_seconds": 300,
    }


@router.post("/aadhaar/verify-otp", response_model=schemas.AadhaarStatus)
def verify_aadhaar_otp(
    data: schemas.AadhaarVerifyOTP,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    aadhaar = db.query(models.AadhaarVerification).filter_by(user_id=user.id).first()
    if not aadhaar:
        raise HTTPException(status_code=400, detail="No OTP request found. Send OTP first.")

    if aadhaar.otp_expires and datetime.utcnow() > aadhaar.otp_expires:
        raise HTTPException(status_code=400, detail="OTP expired. Request a new one.")

    if aadhaar.otp_code != data.otp:
        raise HTTPException(status_code=400, detail="Invalid OTP")

    aadhaar.is_verified = True
    aadhaar.verified_at = datetime.utcnow()
    aadhaar.otp_code = None
    user.is_verified = True
    db.commit()

    return schemas.AadhaarStatus(
        is_verified=True, last4=aadhaar.aadhaar_last4, verified_at=aadhaar.verified_at,
    )


@router.get("/aadhaar/status", response_model=schemas.AadhaarStatus)
def get_aadhaar_status(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    aadhaar = db.query(models.AadhaarVerification).filter_by(user_id=user.id).first()
    if not aadhaar:
        return schemas.AadhaarStatus(is_verified=False)
    return schemas.AadhaarStatus(
        is_verified=aadhaar.is_verified, last4=aadhaar.aadhaar_last4, verified_at=aadhaar.verified_at,
    )


# ─────────── AngelOne Credentials ───────────

@router.post("/angelone/credentials")
def save_angelone_creds(
    data: schemas.AngelOneCredsInput,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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
    return {"message": "AngelOne credentials saved successfully", "is_configured": True}


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
