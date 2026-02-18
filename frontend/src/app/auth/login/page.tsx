'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { api, setTokens } from '@/lib/api';

export default function LoginPage() {
    const router = useRouter();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleLogin = async () => {
        setError(''); setLoading(true);
        try {
            const res = await api.login({ email, password });
            setTokens(res.access_token, res.refresh_token);
            router.push('/dashboard');
        } catch (e: any) { setError(e.message); }
        setLoading(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && email && password) handleLogin();
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
                        Welcome back. Enter your credentials.
                    </p>
                </div>

                {error && <div className="alert alert-error">{error}</div>}

                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div className="input-group">
                        <label>Email Address</label>
                        <input className="input" type="email" placeholder="rahul@example.com"
                            value={email} onChange={e => setEmail(e.target.value)} onKeyDown={handleKeyDown} />
                    </div>
                    <div className="input-group">
                        <label>Password</label>
                        <input className="input" type="password" placeholder="••••••••"
                            value={password} onChange={e => setPassword(e.target.value)} onKeyDown={handleKeyDown} />
                    </div>
                    <button className="btn btn-primary btn-lg" onClick={handleLogin}
                        disabled={loading || !email || !password} style={{ marginTop: '8px' }}>
                        {loading ? <span className="spinner" /> : '⚡ Login'}
                    </button>
                </div>

                <p style={{ textAlign: 'center', marginTop: '24px', color: 'var(--text-secondary)', fontSize: '14px' }}>
                    Don&apos;t have an account?{' '}
                    <Link href="/auth/signup" style={{ color: 'var(--accent-green)', textDecoration: 'none', fontWeight: 600 }}>
                        Sign up free →
                    </Link>
                </p>
            </motion.div>
        </div>
    );
}
