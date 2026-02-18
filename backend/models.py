"""SQLAlchemy ORM models."""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    aadhaar = relationship("AadhaarVerification", back_populates="user", uselist=False)
    angelone_creds = relationship("AngelOneCredential", back_populates="user", uselist=False)
    trades = relationship("TradeRecord", back_populates="user", order_by="TradeRecord.exit_time.desc()")
    bot_sessions = relationship("BotSession", back_populates="user")


class AadhaarVerification(Base):
    __tablename__ = "aadhaar_verifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    aadhaar_number_hash = Column(String, nullable=False)
    aadhaar_last4 = Column(String(4), nullable=False)
    is_verified = Column(Boolean, default=False)
    otp_code = Column(String, nullable=True)
    otp_expires = Column(DateTime, nullable=True)
    verified_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="aadhaar")


class AngelOneCredential(Base):
    __tablename__ = "angelone_credentials"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    api_key_enc = Column(String, nullable=False)
    client_id_enc = Column(String, nullable=False)
    password_enc = Column(String, nullable=False)
    totp_secret_enc = Column(String, nullable=False)
    is_configured = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="angelone_creds")


class TradeRecord(Base):
    __tablename__ = "trade_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String, nullable=False)
    trading_symbol = Column(String, nullable=True)
    option_type = Column(String, nullable=False)
    position_type = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Integer, default=1)
    entry_time = Column(String, nullable=False)
    exit_time = Column(String, nullable=True)
    stop_loss = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)
    close_reason = Column(String, nullable=True)
    trade_date = Column(String, nullable=True)

    user = relationship("User", back_populates="trades")


class BotSession(Base):
    __tablename__ = "bot_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String, default="stopped")
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    pid_csv = Column(Integer, nullable=True)
    pid_symbol = Column(Integer, nullable=True)
    pid_websocket = Column(Integer, nullable=True)
    pid_poshandle = Column(Integer, nullable=True)

    user = relationship("User", back_populates="bot_sessions")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    default_quantity = Column(Integer, default=1)
    price_min = Column(Float, default=110.0)
    price_max = Column(Float, default=150.0)
    trading_start_time = Column(String, default="12:30")
    trading_end_time = Column(String, default="15:10")
    square_off_time = Column(String, default="15:24")
