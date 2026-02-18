'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { useEffect, useState } from 'react';

const tickerItems = [
    { sym: 'NIFTY 50', val: '24,850.25', chg: '+125.40', pct: '+0.51%', up: true },
    { sym: 'BANKNIFTY', val: '52,340.10', chg: '-210.50', pct: '-0.40%', up: false },
    { sym: 'SENSEX', val: '81,920.40', chg: '+380.20', pct: '+0.47%', up: true },
    { sym: 'NIFTY IT', val: '34,560.80', chg: '+89.30', pct: '+0.26%', up: true },
    { sym: 'RELIANCE', val: '₹2,450.30', chg: '-12.60', pct: '-0.51%', up: false },
    { sym: 'TCS', val: '₹3,890.50', chg: '+45.20', pct: '+1.18%', up: true },
    { sym: 'INFY', val: '₹1,560.75', chg: '+22.40', pct: '+1.46%', up: true },
    { sym: 'HDFCBANK', val: '₹1,780.20', chg: '-8.90', pct: '-0.50%', up: false },
];

const features = [
    { icon: '🤖', title: 'AI Signal Engine', desc: 'UT Bot trailing stop loss strategy with real-time NIFTY analysis' },
    { icon: '⚡', title: 'Sub-Second Execution', desc: 'WebSocket-powered data streaming with Redis-backed infrastructure' },
    { icon: '🛡️', title: 'Aadhaar Verified', desc: 'KYC-compliant user verification with Aadhaar-based authentication' },
    { icon: '📊', title: 'Live Dashboard', desc: 'Real-time P&L, positions, trade history synced via WebSocket' },
    { icon: '🔒', title: 'Encrypted Storage', desc: 'Bank-grade Fernet encryption for all API credentials' },
    { icon: '⚙️', title: 'Configurable', desc: 'Custom quantity, price ranges, trading windows per user' },
];

// Candlestick animation data
const candles = Array.from({ length: 50 }, (_, i) => ({
    left: i * 2 + Math.random() * 2,
    height: 20 + Math.random() * 60,
    wickTop: 5 + Math.random() * 15,
    wickBottom: 5 + Math.random() * 15,
    green: Math.random() > 0.45,
    delay: Math.random() * 3,
}));

export default function Home() {
    const [mounted, setMounted] = useState(false);
    useEffect(() => setMounted(true), []);

    return (
        <div className="page-container">
            {/* Candlestick Background */}
            {mounted && (
                <div className="candle-bg">
                    {candles.map((c, i) => (
                        <motion.div
                            key={i}
                            className={`candle ${c.green ? 'green' : 'red'}`}
                            style={{ left: `${c.left}%`, height: c.height }}
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: c.height, opacity: 1 }}
                            transition={{ duration: 1.5, delay: c.delay, ease: 'easeOut' }}
                        />
                    ))}
                </div>
            )}

            {/* Ticker Tape */}
            <div className="ticker-tape">
                <div className="ticker-content">
                    {[...tickerItems, ...tickerItems].map((item, i) => (
                        <span key={i} style={{ color: item.up ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                            {item.sym} {item.val} {item.chg} ({item.pct})
                            <span style={{ color: 'var(--text-muted)', margin: '0 8px' }}>│</span>
                        </span>
                    ))}
                </div>
            </div>

            {/* Hero */}
            <div style={{ position: 'relative', zIndex: 1 }}>
                <nav className="navbar">
                    <div className="navbar-inner">
                        <Link href="/" className="navbar-brand">
                            <span className="logo-icon">📈</span>
                            <span>Trade<span style={{ color: 'var(--accent-green)' }}>Pulse</span></span>
                        </Link>
                        <div className="navbar-links">
                            <Link href="/auth/login"><button className="btn btn-outline btn-sm">Login</button></Link>
                            <Link href="/auth/signup"><button className="btn btn-primary btn-sm">Get Started</button></Link>
                        </div>
                    </div>
                </nav>

                <section style={{ textAlign: 'center', padding: '120px 24px 80px' }}>
                    <motion.div
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                    >
                        <div style={{
                            display: 'inline-block',
                            padding: '6px 16px',
                            borderRadius: '20px',
                            background: 'var(--accent-green-dim)',
                            border: '1px solid rgba(0,255,136,0.2)',
                            fontSize: '13px',
                            fontWeight: 600,
                            color: 'var(--accent-green)',
                            marginBottom: '24px',
                        }}>
                            ⚡ Powered by UT Bot Signals + AngelOne SmartAPI
                        </div>
                        <h1 style={{
                            fontSize: 'clamp(40px, 6vw, 72px)',
                            fontWeight: 900,
                            lineHeight: 1.1,
                            letterSpacing: '-2px',
                            marginBottom: '24px',
                        }}>
                            Trade NIFTY Options
                            <br />
                            <span style={{
                                background: 'linear-gradient(135deg, var(--accent-green), var(--accent-blue))',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                            }}>
                                Like a Pro
                            </span>
                        </h1>
                        <p style={{
                            fontSize: '18px',
                            color: 'var(--text-secondary)',
                            maxWidth: '600px',
                            margin: '0 auto 40px',
                            lineHeight: 1.7,
                        }}>
                            Automated CE/PE options trading with real-time signals, trailing stop loss, and a premium dashboard. Connect your AngelOne account and let AI handle the rest.
                        </p>
                        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                            <Link href="/auth/signup">
                                <button className="btn btn-primary btn-lg">
                                    🚀 Start Trading Now
                                </button>
                            </Link>
                            <Link href="/auth/login">
                                <button className="btn btn-outline btn-lg">
                                    Login →
                                </button>
                            </Link>
                        </div>
                    </motion.div>
                </section>

                {/* Stats Strip */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="container"
                    style={{ marginBottom: '80px' }}
                >
                    <div className="grid-4" style={{ textAlign: 'center' }}>
                        {[
                            { label: 'Active Users', value: '500+', icon: '👥' },
                            { label: 'Trades Executed', value: '12,400+', icon: '📊' },
                            { label: 'Avg Win Rate', value: '68%', icon: '🎯' },
                            { label: 'Uptime', value: '99.9%', icon: '⚡' },
                        ].map((stat, i) => (
                            <motion.div
                                key={stat.label}
                                className="glass stat-card"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.7 + i * 0.1 }}
                            >
                                <span style={{ fontSize: '32px' }}>{stat.icon}</span>
                                <div className="stat-value">{stat.value}</div>
                                <div className="stat-label">{stat.label}</div>
                            </motion.div>
                        ))}
                    </div>
                </motion.div>

                {/* Features */}
                <section className="container" style={{ paddingBottom: '100px' }}>
                    <h2 style={{ textAlign: 'center', fontSize: '36px', marginBottom: '48px' }}>
                        Why <span style={{ color: 'var(--accent-green)' }}>TradePulse</span>?
                    </h2>
                    <div className="grid-3">
                        {features.map((f, i) => (
                            <motion.div
                                key={f.title}
                                className="glass glass-hover"
                                style={{ padding: '32px' }}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.3 + i * 0.1 }}
                            >
                                <span style={{ fontSize: '40px', display: 'block', marginBottom: '16px' }}>{f.icon}</span>
                                <h3 style={{ fontSize: '18px', marginBottom: '8px' }}>{f.title}</h3>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>{f.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </section>

                {/* Footer */}
                <footer style={{
                    textAlign: 'center',
                    padding: '32px',
                    borderTop: '1px solid var(--border-glass)',
                    color: 'var(--text-muted)',
                    fontSize: '13px',
                }}>
                    © 2026 TradePulse. Built with AngelOne SmartAPI.
                    <br />
                    <span style={{ color: 'var(--text-secondary)' }}>
                        Trading involves risk. Past performance is not indicative of future results.
                    </span>
                </footer>
            </div>
        </div>
    );
}
