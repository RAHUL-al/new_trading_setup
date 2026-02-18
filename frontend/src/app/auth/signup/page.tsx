'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { api, setTokens } from '@/lib/api';

export default function SignupPage() {
    const router = useRouter();
    const [step, setStep] = useState(1);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [phone, setPhone] = useState('');
    const [password, setPassword] = useState('');

    const [aadhaar, setAadhaar] = useState('');
    const [otp, setOtp] = useState('');
    const [otpSent, setOtpSent] = useState(false);
    const [otpHint, setOtpHint] = useState('');

    const [apiKey, setApiKey] = useState('');
    const [clientId, setClientId] = useState('');
    const [angelPass, setAngelPass] = useState('');
    const [totpSecret, setTotpSecret] = useState('');

    const handleSignup = async () => {
        setError(''); setLoading(true);
        try {
            const res = await api.signup({ email, full_name: fullName, phone: phone || undefined, password });
            setTokens(res.access_token, res.refresh_token);
            setStep(2);
        } catch (e: any) { setError(e.message); }
        setLoading(false);
    };

    const handleSendOTP = async () => {
        setError(''); setLoading(true);
        try {
            const res = await api.sendAadhaarOTP(aadhaar);
            setOtpSent(true);
            setOtpHint(res.otp_hint);
        } catch (e: any) { setError(e.message); }
        setLoading(false);
    };

    const handleVerifyOTP = async () => {
        setError(''); setLoading(true);
        try {
            await api.verifyAadhaarOTP(otp);
            setStep(3);
        } catch (e: any) { setError(e.message); }
        setLoading(false);
    };

    const handleAngelOne = async () => {
        setError(''); setLoading(true);
        try {
            await api.saveAngelOneCreds({ api_key: apiKey, client_id: clientId, password: angelPass, totp_secret: totpSecret });
            router.push('/dashboard');
        } catch (e: any) { setError(e.message); }
        setLoading(false);
    };

    const skipToNext = () => {
        if (step === 2) setStep(3);
        else if (step === 3) router.push('/dashboard');
    };

    return (
        <div className="page-container" style={{ display: 'flex', minHeight: '100vh', alignItems: 'center', justifyContent: 'center' }}>
            <motion.div
                className="glass gradient-border"
                style={{ width: '100%', maxWidth: 480, padding: '48px 40px' }}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
            >
                <div style={{ textAlign: 'center', marginBottom: '32px' }}>
                    <Link href="/" style={{ textDecoration: 'none' }}>
                        <span style={{ fontSize: '32px' }}>📈</span>
                        <h1 style={{ fontSize: '24px', marginTop: '8px' }}>
                            Trade<span style={{ color: 'var(--accent-green)' }}>Pulse</span>
                        </h1>
                    </Link>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '8px' }}>
                        {step === 1 && 'Create your trading account'}
                        {step === 2 && 'Verify your identity with Aadhaar'}
                        {step === 3 && 'Connect your AngelOne account'}
                    </p>
                    <div style={{ display: 'flex', justifyContent: 'center', gap: '8px', marginTop: '16px' }}>
                        {[1, 2, 3].map(s => (
                            <div key={s} style={{
                                width: s === step ? '32px' : '10px', height: '4px', borderRadius: '2px',
                                background: s <= step ? 'var(--accent-green)' : 'var(--border-glass)',
                                transition: 'all 0.3s ease',
                            }} />
                        ))}
                    </div>
                </div>

                {error && <div className="alert alert-error">{error}</div>}

                {step === 1 && (
                    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        <div className="input-group">
                            <label>Full Name</label>
                            <input className="input" placeholder="Rahul Sharma" value={fullName} onChange={e => setFullName(e.target.value)} />
                        </div>
                        <div className="input-group">
                            <label>Email Address</label>
                            <input className="input" type="email" placeholder="rahul@example.com" value={email} onChange={e => setEmail(e.target.value)} />
                        </div>
                        <div className="input-group">
                            <label>Phone (optional)</label>
                            <input className="input" placeholder="+91 98765 43210" value={phone} onChange={e => setPhone(e.target.value)} />
                        </div>
                        <div className="input-group">
                            <label>Password</label>
                            <input className="input" type="password" placeholder="Min 6 characters" value={password} onChange={e => setPassword(e.target.value)} />
                        </div>
                        <button className="btn btn-primary btn-lg" onClick={handleSignup} disabled={loading || !fullName || !email || !password} style={{ marginTop: '8px' }}>
                            {loading ? <span className="spinner" /> : '→ Create Account'}
                        </button>
                    </motion.div>
                )}

                {step === 2 && (
                    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        {!otpSent ? (
                            <>
                                <div className="input-group">
                                    <label>Aadhaar Number</label>
                                    <input className="input mono" placeholder="XXXX XXXX XXXX" maxLength={14} value={aadhaar}
                                        onChange={e => setAadhaar(e.target.value.replace(/\D/g, '').slice(0, 12))} />
                                </div>
                                <button className="btn btn-primary btn-lg" onClick={handleSendOTP} disabled={loading || aadhaar.length !== 12}>
                                    {loading ? <span className="spinner" /> : '📱 Send OTP'}
                                </button>
                            </>
                        ) : (
                            <>
                                {otpHint && <div className="alert alert-success">🔑 Mock OTP: <strong className="mono">{otpHint}</strong> (demo mode)</div>}
                                <div className="input-group">
                                    <label>Enter OTP</label>
                                    <input className="input mono" placeholder="6-digit OTP" maxLength={6} value={otp}
                                        onChange={e => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))} style={{ fontSize: '24px', letterSpacing: '8px', textAlign: 'center' }} />
                                </div>
                                <button className="btn btn-primary btn-lg" onClick={handleVerifyOTP} disabled={loading || otp.length !== 6}>
                                    {loading ? <span className="spinner" /> : '✓ Verify'}
                                </button>
                            </>
                        )}
                        <button className="btn btn-outline btn-sm" onClick={skipToNext} style={{ alignSelf: 'center' }}>Skip for later →</button>
                    </motion.div>
                )}

                {step === 3 && (
                    <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                        <div className="alert" style={{ background: 'var(--accent-blue-dim)', border: '1px solid rgba(0,212,255,0.3)', color: 'var(--accent-blue)', fontSize: '13px' }}>
                            🔒 Credentials are encrypted with bank-grade Fernet encryption
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
                        <button className="btn btn-primary btn-lg" onClick={handleAngelOne}
                            disabled={loading || !apiKey || !clientId || !angelPass || !totpSecret} style={{ marginTop: '8px' }}>
                            {loading ? <span className="spinner" /> : '🔗 Connect & Start'}
                        </button>
                        <button className="btn btn-outline btn-sm" onClick={skipToNext} style={{ alignSelf: 'center' }}>Setup later →</button>
                    </motion.div>
                )}

                <p style={{ textAlign: 'center', marginTop: '24px', color: 'var(--text-secondary)', fontSize: '14px' }}>
                    Already have an account?{' '}
                    <Link href="/auth/login" style={{ color: 'var(--accent-green)', textDecoration: 'none', fontWeight: 600 }}>Login →</Link>
                </p>
            </motion.div>
        </div>
    );
}
