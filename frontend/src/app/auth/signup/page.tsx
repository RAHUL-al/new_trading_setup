'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { api, setTokens } from '@/lib/api';

export default function SignupPage() {
    const router = useRouter();
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [phone, setPhone] = useState('');
    const [password, setPassword] = useState('');

    const handleSignup = async () => {
        setError(''); setLoading(true);
        try {
            const res = await api.signup({ email, full_name: fullName, phone: phone || undefined, password });
            setTokens(res.access_token, res.refresh_token);
            router.push('/setup'); // → Mandatory verification
        } catch (e: any) { setError(e.message); }
        setLoading(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && fullName && email && password) handleSignup();
    };

    return (
        <div className="page-container" style={{ display: 'flex', minHeight: '100vh', alignItems: 'center', justifyContent: 'center' }}>
            <motion.div
                className="glass gradient-border"
                style={{ width: '100%', maxWidth: 440, padding: '48px 40px' }}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
            >
                <div style={{ textAlign: 'center', marginBottom: '32px' }}>
                    <Link href="/" style={{ textDecoration: 'none' }}>
                        <span style={{ fontSize: '40px' }}>📈</span>
                        <h1 style={{ fontSize: '24px', marginTop: '8px' }}>
                            Trade<span style={{ color: 'var(--accent-green)' }}>Pulse</span>
                        </h1>
                    </Link>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '14px', marginTop: '8px' }}>
                        Create your trading account
                    </p>
                </div>

                {error && <div className="alert alert-error">{error}</div>}

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div className="input-group">
                        <label>Full Name</label>
                        <input className="input" placeholder="Rahul Sharma" value={fullName}
                            onChange={e => setFullName(e.target.value)} onKeyDown={handleKeyDown} />
                    </div>
                    <div className="input-group">
                        <label>Email Address</label>
                        <input className="input" type="email" placeholder="rahul@example.com" value={email}
                            onChange={e => setEmail(e.target.value)} onKeyDown={handleKeyDown} />
                    </div>
                    <div className="input-group">
                        <label>Phone (optional)</label>
                        <input className="input" placeholder="+91 98765 43210" value={phone}
                            onChange={e => setPhone(e.target.value)} onKeyDown={handleKeyDown} />
                    </div>
                    <div className="input-group">
                        <label>Password</label>
                        <input className="input" type="password" placeholder="Min 6 characters" value={password}
                            onChange={e => setPassword(e.target.value)} onKeyDown={handleKeyDown} />
                    </div>
                    <button className="btn btn-primary btn-lg" onClick={handleSignup}
                        disabled={loading || !fullName || !email || !password} style={{ marginTop: '8px' }}>
                        {loading ? <span className="spinner" /> : '→ Create Account'}
                    </button>
                </div>

                <p style={{ textAlign: 'center', marginTop: '24px', color: 'var(--text-secondary)', fontSize: '14px' }}>
                    Already have an account?{' '}
                    <Link href="/auth/login" style={{ color: 'var(--accent-green)', textDecoration: 'none', fontWeight: 600 }}>Login →</Link>
                </p>
            </motion.div>
        </div>
    );
}
