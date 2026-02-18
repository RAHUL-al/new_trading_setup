"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional, List
from datetime import datetime
import re


class UserSignup(BaseModel):
    email: EmailStr
    full_name: str
    phone: Optional[str] = None
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserProfile"


class UserProfile(BaseModel):
    id: int
    email: str
    full_name: str
    phone: Optional[str]
    is_verified: bool
    is_active: bool
    has_angelone: bool = False
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AadhaarSendOTP(BaseModel):
    aadhaar_number: str

    @field_validator("aadhaar_number")
    @classmethod
    def validate_aadhaar(cls, v):
        cleaned = re.sub(r"\s+", "", v)
        if not re.match(r"^\d{12}$", cleaned):
            raise ValueError("Aadhaar number must be exactly 12 digits")
        return cleaned


class AadhaarVerifyOTP(BaseModel):
    otp: str

    @field_validator("otp")
    @classmethod
    def validate_otp(cls, v):
        if not re.match(r"^\d{6}$", v):
            raise ValueError("OTP must be exactly 6 digits")
        return v


class AadhaarStatus(BaseModel):
    is_verified: bool
    last4: Optional[str] = None
    verified_at: Optional[datetime] = None


class AngelOneCredsInput(BaseModel):
    api_key: str
    client_id: str
    password: str
    totp_secret: str


class AngelOneCredsStatus(BaseModel):
    is_configured: bool
    client_id_masked: Optional[str] = None
    updated_at: Optional[datetime] = None


class TradeResponse(BaseModel):
    id: int
    token: str
    trading_symbol: Optional[str]
    option_type: str
    position_type: str
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    entry_time: str
    exit_time: Optional[str]
    stop_loss: Optional[float]
    pnl: float
    close_reason: Optional[str]
    trade_date: Optional[str]

    class Config:
        from_attributes = True


class PortfolioResponse(BaseModel):
    total_trades: int
    today_trades: int
    today_pnl: float
    total_pnl: float
    win_rate: float
    open_position: Optional[dict] = None
    bot_status: str = "stopped"


class BotControlRequest(BaseModel):
    action: str


class BotStatusResponse(BaseModel):
    status: str
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None


class UserSettingsInput(BaseModel):
    default_quantity: Optional[int] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    trading_start_time: Optional[str] = None
    trading_end_time: Optional[str] = None
    square_off_time: Optional[str] = None


class UserSettingsResponse(BaseModel):
    default_quantity: int
    price_min: float
    price_max: float
    trading_start_time: str
    trading_end_time: str
    square_off_time: str

    class Config:
        from_attributes = True


class WSMessage(BaseModel):
    type: str
    data: dict


TokenResponse.model_rebuild()
