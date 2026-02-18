'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { api, clearTokens, type UserProfile } from '@/lib/api';

export default function SetupPage() {
    const router = useRouter();
    const [user, setUser] = useState<UserProfile | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [actionLoading, setActionLoading] = useState(false);

    // Email OTP
    const [otp, setOtp] = useState('');
    const [otpSent, setOtpSent] = useState(false);
    const [fallbackOtp, setFallbackOtp] = useState('');
    const [emailVerified, setEmailVerified] = useState(false);

    // AngelOne
    const [apiKey, setApiKey] = useState('');
    const [clientId, setClientId] = useState('');
    const [angelPass, setAngelPass] = useState('');
    const [totpSecret, setTotpSecret] = useState('');
    const [angelOneConfigured, setAngelOneConfigured] = useState(false);

    useEffect(() => {
        api.getMe().then(u => {
            setUser(u);
            setEmailVerified(u.is_verified);
            setAngelOneConfigured(u.has_angelone);
            if (u.is_verified && u.has_angelone) {
                router.push('/dashboard');
            }
            setLoading(false);
        }).catch(() => {
            clearTokens();
            router.push('/auth/login');
        });
    }, [router]);

    const handleSendOTP = async () => {
        setError(''); setActionLoading(true);
        try {
            const res = await api.sendEmailOTP();
            setOtpSent(true);
            if (res.fallback_otp) {
                setFallbackOtp(res.fallback_otp);
            }
        } catch (e: any) { setError(e.message); }
        setActionLoading(false);
    };

    const handleVerifyOTP = async () => {
        setError(''); setActionLoading(true);
        try {
            await api.verifyEmailOTP(otp);
            setEmailVerified(true);
            setFallbackOtp('');
            setError('');
        } catch (e: any) { setError(e.message); }
        setActionLoading(false);
    };

    const handleAngelOne = async () => {
        setError(''); setActionLoading(true);
        try {
            await api.saveAngelOneCreds({ api_key: apiKey, client_id: clientId, password: angelPass, totp_secret: totpSecret });
            setAngelOneConfigured(true);
            setError('');
        } catch (e: any) { setError(e.message); }
        setActionLoading(false);
    };

    if (loading) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', gap: '12px' }}>
                <span className="spinner" /> <span style={{ color: 'var(--text-secondary)' }}>Loading...</span>
            </div>
        );
    }

    const allDone = emailVerified && angelOneConfigured;

    return (
        <div className="page-container" style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '40px 24px' }}>
            <motion.div
                style={{ width: '100%', maxWidth: 560 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
            >
                {/* Header */}
                <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                    <span style={{ fontSize: '40px', display: 'block', marginBottom: '12px' }}>📈</span>
                    <h1 style={{ fontSize: '28px', marginBottom: '8px' }}>
                        Welcome, <span style={{ color: 'var(--accent-green)' }}>{user?.full_name?.split(' ')[0]}</span>!
                    </h1>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '15px' }}>
                        Complete these steps to start trading
                    </p>
                </div>

                {error && <div className="alert alert-error" style={{ marginBottom: '20px' }}>{error}</div>}

                {/* Step 1: Email Verification */}
                <motion.div
                    className="glass"
                    style={{ padding: '28px', marginBottom: '16px', border: emailVerified ? '1px solid rgba(0,255,136,0.3)' : undefined }}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: emailVerified ? '0' : '20px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <span style={{
                                width: '36px', height: '36px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: emailVerified ? 'var(--accent-green-dim)' : 'rgba(255,255,255,0.05)', fontSize: '18px',
                            }}>
                                {emailVerified ? '✅' : '1'}
                            </span>
                            <div>
                                <div style={{ fontWeight: 700, fontSize: '16px' }}>Email Verification</div>
                                <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                                    {emailVerified ? 'Verified successfully' : `We'll send a code to ${user?.email}`}
                                </div>
                            </div>
                        </div>
                        {emailVerified && <span style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '14px' }}>DONE</span>}
                    </div>

                    {!emailVerified && (
                        <div style={{ marginTop: '16px' }}>
                            {!otpSent ? (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    <div className="glass" style={{ padding: '16px', textAlign: 'center', background: 'rgba(255,255,255,0.02)' }}>
                                        <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginBottom: '4px' }}>Sending OTP to</div>
                                        <div className="mono" style={{ fontSize: '16px', fontWeight: 600, color: 'var(--accent-blue)' }}>{user?.email}</div>
                                    </div>
                                    <button className="btn btn-primary" onClick={handleSendOTP} disabled={actionLoading}>
                                        {actionLoading ? <span className="spinner" /> : '📧 Send Verification Code'}
                                    </button>
                                </div>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                    <div className="alert alert-success" style={{ marginBottom: '0', textAlign: 'center' }}>
                                        📧 OTP sent to <strong>{user?.email}</strong>
                                        <br /><span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Check your inbox (and spam folder)</span>
                                    </div>
                                    {fallbackOtp && (
                                        <div className="alert" style={{ background: 'rgba(255,165,0,0.1)', border: '1px solid rgba(255,165,0,0.3)', color: '#ffa500', fontSize: '13px', textAlign: 'center', marginBottom: '0' }}>
                                            ⚠️ SMTP not configured — Test OTP: <strong className="mono">{fallbackOtp}</strong>
                                        </div>
                                    )}
                                    <div className="input-group">
                                        <label>Enter 6-digit Code</label>
                                        <input className="input mono" placeholder="000000" maxLength={6} value={otp}
                                            onChange={e => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
                                            style={{ fontSize: '24px', letterSpacing: '8px', textAlign: 'center' }} />
                                    </div>
                                    <button className="btn btn-primary" onClick={handleVerifyOTP} disabled={actionLoading || otp.length !== 6}>
                                        {actionLoading ? <span className="spinner" /> : '✓ Verify Code'}
                                    </button>
                                    <button className="btn btn-outline btn-sm" onClick={() => { setOtpSent(false); setOtp(''); setFallbackOtp(''); }} style={{ alignSelf: 'center' }}>
                                        ← Resend Code
                                    </button>
                                </div>
                            )}
                        </div>
                    )}
                </motion.div>

                {/* Step 2: AngelOne Configuration */}
                <motion.div
                    className="glass"
                    style={{
                        padding: '28px', marginBottom: '24px',
                        border: angelOneConfigured ? '1px solid rgba(0,255,136,0.3)' : undefined,
                        opacity: emailVerified ? 1 : 0.4,
                        pointerEvents: emailVerified ? 'auto' : 'none',
                    }}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: emailVerified ? 1 : 0.4, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: angelOneConfigured ? '0' : '20px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                            <span style={{
                                width: '36px', height: '36px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                background: angelOneConfigured ? 'var(--accent-green-dim)' : 'rgba(255,255,255,0.05)', fontSize: '18px',
                            }}>
                                {angelOneConfigured ? '✅' : '2'}
                            </span>
                            <div>
                                <div style={{ fontWeight: 700, fontSize: '16px' }}>AngelOne Account</div>
                                <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                                    {angelOneConfigured ? 'Connected successfully' : 'Connect your trading account'}
                                </div>
                            </div>
                        </div>
                        {angelOneConfigured && <span style={{ color: 'var(--accent-green)', fontWeight: 700, fontSize: '14px' }}>DONE</span>}
                        {!emailVerified && <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>🔒 Complete Step 1 first</span>}
                    </div>

                    {emailVerified && !angelOneConfigured && (
                        <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                            <div className="alert" style={{ background: 'var(--accent-blue-dim)', border: '1px solid rgba(0,212,255,0.3)', color: 'var(--accent-blue)', fontSize: '13px', marginBottom: '0' }}>
                                🔒 Your credentials are encrypted with bank-grade Fernet encryption
                            </div>
                            <div className="input-group">
                                <label>API Key</label>
                                <input className="input mono" placeholder="Your SmartAPI key" value={apiKey} onChange={e => setApiKey(e.target.value)} />
                            </div>
                            <div className="input-group">
                                <label>Client ID</label>
                                <input className="input mono" placeholder="e.g. R865920" value={clientId} onChange={e => setClientId(e.target.value)} />
                            </div>
                            <div className="input-group">
                                <label>Password</label>
                                <input className="input mono" type="password" placeholder="Trading password" value={angelPass} onChange={e => setAngelPass(e.target.value)} />
                            </div>
                            <div className="input-group">
                                <label>TOTP Secret</label>
                                <input className="input mono" placeholder="32-char TOTP secret" value={totpSecret} onChange={e => setTotpSecret(e.target.value)} />
                            </div>
                            <button className="btn btn-primary" onClick={handleAngelOne}
                                disabled={actionLoading || !apiKey || !clientId || !angelPass || !totpSecret}>
                                {actionLoading ? <span className="spinner" /> : '🔗 Connect AngelOne'}
                            </button>
                        </div>
                    )}
                </motion.div>

                {/* Continue */}
                {allDone && (
                    <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring' }} style={{ textAlign: 'center' }}>
                        <div className="alert alert-success" style={{ marginBottom: '16px', textAlign: 'center', fontSize: '16px' }}>
                            🎉 All set! You&apos;re ready to trade.
                        </div>
                        <button className="btn btn-primary btn-lg" onClick={() => router.push('/dashboard')} style={{ width: '100%' }}>
                            🚀 Go to Dashboard
                        </button>
                    </motion.div>
                )}

                <div style={{ textAlign: 'center', marginTop: '24px' }}>
                    <button className="btn btn-outline btn-sm" onClick={() => { clearTokens(); router.push('/auth/login'); }}>
                        ← Logout
                    </button>
                </div>
            </motion.div>
        </div>
    );
}
