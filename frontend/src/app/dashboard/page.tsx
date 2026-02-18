'use client';

import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { api, clearTokens, createWSConnection, type PortfolioData, type TradeRecord, type UserProfile, type UserSettings } from '@/lib/api';

type Tab = 'overview' | 'trades' | 'settings';

export default function DashboardPage() {
    const router = useRouter();
    const [user, setUser] = useState<UserProfile | null>(null);
    const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
    const [trades, setTrades] = useState<TradeRecord[]>([]);
    const [settings, setSettings] = useState<UserSettings | null>(null);
    const [tab, setTab] = useState<Tab>('overview');
    const [loading, setLoading] = useState(true);
    const [botLoading, setBotLoading] = useState(false);
    const [settingsSaving, setSettingsSaving] = useState(false);
    const [message, setMessage] = useState('');
    const [wsData, setWsData] = useState<any>(null);
    const [niftyPrice, setNiftyPrice] = useState<number>(0);

    const [qty, setQty] = useState(1);
    const [priceMin, setPriceMin] = useState(110);
    const [priceMax, setPriceMax] = useState(150);

    const loadData = useCallback(async () => {
        try {
            const [u, p, t, s] = await Promise.all([
                api.getMe(), api.getPortfolio(), api.getTrades(), api.getSettings(),
            ]);
            setUser(u); setPortfolio(p); setTrades(t); setSettings(s);
            setQty(s.default_quantity); setPriceMin(s.price_min); setPriceMax(s.price_max);
        } catch {
            clearTokens(); router.push('/auth/login');
        }
        setLoading(false);
    }, [router]);

    useEffect(() => {
        loadData();
        const refreshInterval = setInterval(() => {
            api.getPortfolio().then(setPortfolio).catch(() => { });
        }, 5000);
        return () => clearInterval(refreshInterval);
    }, [loadData]);

    useEffect(() => {
        const ws = createWSConnection((msg) => {
            if (msg.type === 'update') {
                setWsData(msg.data);
                if (msg.data.nifty_price) setNiftyPrice(msg.data.nifty_price);
                if (msg.data.today_pnl !== undefined && portfolio) {
                    setPortfolio(prev => prev ? { ...prev, today_pnl: msg.data.today_pnl, today_trades: msg.data.today_trades || prev.today_trades } : prev);
                }
            }
        });
        return () => { ws?.close(); };
    }, [portfolio]);

    const handleBotControl = async (action: string) => {
        setBotLoading(true); setMessage('');
        try {
            const res = await api.controlBot(action);
            setMessage(`Bot ${res.status}`);
            setTimeout(loadData, 1000);
        } catch (e: any) { setMessage(e.message); }
        setBotLoading(false);
    };

    const handleSaveSettings = async () => {
        setSettingsSaving(true);
        try {
            await api.updateSettings({ default_quantity: qty, price_min: priceMin, price_max: priceMax });
            setMessage('Settings saved!');
            setTimeout(() => setMessage(''), 3000);
        } catch (e: any) { setMessage(e.message); }
        setSettingsSaving(false);
    };

    const logout = () => { clearTokens(); router.push('/auth/login'); };

    if (loading) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', gap: '12px' }}>
                <span className="spinner" /> <span style={{ color: 'var(--text-secondary)' }}>Loading dashboard...</span>
            </div>
        );
    }

    const pos = wsData?.position || portfolio?.open_position;
    const botStatus = portfolio?.bot_status || 'stopped';
    const isBotRunning = botStatus === 'running';

    return (
        <div className="page-container">
            <nav className="navbar">
                <div className="navbar-inner">
                    <Link href="/dashboard" className="navbar-brand">
                        <span className="logo-icon">📈</span>
                        <span>Trade<span style={{ color: 'var(--accent-green)' }}>Pulse</span></span>
                    </Link>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                        {niftyPrice > 0 && (
                            <div className="mono" style={{ fontSize: '14px', color: 'var(--accent-blue)' }}>
                                NIFTY {niftyPrice.toFixed(2)}
                            </div>
                        )}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className={`status-dot ${isBotRunning ? 'live' : 'offline'}`} />
                            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                                {isBotRunning ? 'Live' : 'Offline'}
                            </span>
                        </div>
                        <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
                            {user?.full_name}
                            {user?.is_verified && <span title="Aadhaar Verified" style={{ marginLeft: '6px' }}>✅</span>}
                        </span>
                        <button className="btn btn-outline btn-sm" onClick={logout}>Logout</button>
                    </div>
                </div>
            </nav>

            <div className="container" style={{ marginTop: '24px' }}>
                <div style={{ display: 'flex', gap: '4px', marginBottom: '24px' }}>
                    {(['overview', 'trades', 'settings'] as Tab[]).map(t => (
                        <button key={t} className={`btn btn-sm ${tab === t ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setTab(t)} style={{ textTransform: 'capitalize' }}>
                            {t === 'overview' ? '📊 Overview' : t === 'trades' ? '📜 Trade History' : '⚙️ Settings'}
                        </button>
                    ))}
                </div>

                {message && (
                    <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                        className={`alert ${message.includes('error') || message.includes('Error') ? 'alert-error' : 'alert-success'}`}>
                        {message}
                    </motion.div>
                )}

                <AnimatePresence mode="wait">
                    {tab === 'overview' && (
                        <motion.div key="overview" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                            <div className="grid-4" style={{ marginBottom: '24px' }}>
                                <motion.div className="glass stat-card" whileHover={{ scale: 1.02 }}>
                                    <div className="stat-label">Today P&L</div>
                                    <div className={`stat-value ${(portfolio?.today_pnl || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                                        ₹{(portfolio?.today_pnl || 0).toFixed(2)}
                                    </div>
                                    <div className="stat-sub">{portfolio?.today_trades || 0} trades today</div>
                                </motion.div>
                                <motion.div className="glass stat-card" whileHover={{ scale: 1.02 }}>
                                    <div className="stat-label">Total P&L</div>
                                    <div className={`stat-value ${(portfolio?.total_pnl || 0) >= 0 ? 'text-green' : 'text-red'}`}>
                                        ₹{(portfolio?.total_pnl || 0).toFixed(2)}
                                    </div>
                                    <div className="stat-sub">{portfolio?.total_trades || 0} total trades</div>
                                </motion.div>
                                <motion.div className="glass stat-card" whileHover={{ scale: 1.02 }}>
                                    <div className="stat-label">Win Rate</div>
                                    <div className="stat-value text-blue">{(portfolio?.win_rate || 0).toFixed(1)}%</div>
                                    <div className="stat-sub">Success ratio</div>
                                </motion.div>
                                <motion.div className="glass stat-card" whileHover={{ scale: 1.02 }}>
                                    <div className="stat-label">Bot Status</div>
                                    <div className="stat-value" style={{ fontSize: '20px' }}>
                                        <span className={`status-dot ${isBotRunning ? 'live' : 'offline'}`} style={{ marginRight: '8px' }} />
                                        {botStatus.toUpperCase()}
                                    </div>
                                    <div style={{ marginTop: '8px' }}>
                                        <button className={`btn btn-sm ${isBotRunning ? 'btn-danger' : 'btn-primary'}`}
                                            onClick={() => handleBotControl(isBotRunning ? 'stop' : 'start')} disabled={botLoading}>
                                            {botLoading ? <span className="spinner" /> : (isBotRunning ? '⏹ Stop Bot' : '▶ Start Bot')}
                                        </button>
                                    </div>
                                </motion.div>
                            </div>

                            <div className="glass" style={{ padding: '24px', marginBottom: '24px' }}>
                                <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    {pos ? <>📍 Open Position <span className="status-dot live" /></> : '📍 No Open Position'}
                                </h3>
                                {pos ? (
                                    <div className="grid-3" style={{ gap: '32px' }}>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Symbol</div>
                                            <div className="mono" style={{ fontSize: '16px', fontWeight: 700 }}>
                                                <span style={{
                                                    display: 'inline-block', padding: '2px 8px', borderRadius: '4px',
                                                    background: pos.option_type === 'CE' ? 'var(--accent-green-dim)' : 'var(--accent-red-dim)',
                                                    color: pos.option_type === 'CE' ? 'var(--accent-green)' : 'var(--accent-red)',
                                                    marginRight: '8px', fontSize: '12px',
                                                }}>{pos.option_type}</span>
                                                {pos.trading_symbol || pos.token}
                                            </div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Entry / Current</div>
                                            <div className="mono" style={{ fontSize: '16px' }}>₹{pos.entry_price?.toFixed(2)} → ₹{pos.current_price?.toFixed(2)}</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Live P&L</div>
                                            <motion.div className={`mono ${pos.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`} style={{ fontSize: '24px' }}
                                                key={pos.pnl} initial={{ scale: 1.2 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}>
                                                {pos.pnl >= 0 ? '+' : ''}₹{pos.pnl?.toFixed(2)}
                                            </motion.div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Stop Loss</div>
                                            <div className="mono text-red" style={{ fontSize: '16px' }}>₹{pos.stop_loss?.toFixed(2)}</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Quantity</div>
                                            <div className="mono" style={{ fontSize: '16px' }}>{pos.quantity}</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Entry Time</div>
                                            <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{pos.entry_time}</div>
                                        </div>
                                    </div>
                                ) : (
                                    <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                                        {isBotRunning ? 'Waiting for next signal...' : 'Start the bot to begin trading.'}
                                    </p>
                                )}
                            </div>

                            <div className="glass" style={{ padding: '24px' }}>
                                <h3 style={{ marginBottom: '16px' }}>📜 Recent Trades</h3>
                                {trades.length === 0 ? (
                                    <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>No trades yet</p>
                                ) : (
                                    <div className="table-container">
                                        <table>
                                            <thead><tr><th>Type</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>Reason</th><th>Time</th></tr></thead>
                                            <tbody>
                                                {trades.slice(0, 10).map(t => (
                                                    <tr key={t.id}>
                                                        <td><span style={{ padding: '2px 8px', borderRadius: '4px', background: t.option_type === 'CE' ? 'var(--accent-green-dim)' : 'var(--accent-red-dim)', color: t.option_type === 'CE' ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: '12px', fontWeight: 600 }}>{t.option_type}</span></td>
                                                        <td>{t.trading_symbol || t.token}</td>
                                                        <td>₹{t.entry_price.toFixed(2)}</td>
                                                        <td>{t.exit_price ? `₹${t.exit_price.toFixed(2)}` : '—'}</td>
                                                        <td>{t.quantity}</td>
                                                        <td className={t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>{t.pnl >= 0 ? '+' : ''}₹{t.pnl.toFixed(2)}</td>
                                                        <td style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>{t.close_reason || '—'}</td>
                                                        <td style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{t.exit_time || t.entry_time}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {tab === 'trades' && (
                        <motion.div key="trades" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                            <div className="glass" style={{ padding: '24px' }}>
                                <h3 style={{ marginBottom: '16px' }}>📜 All Trades ({trades.length})</h3>
                                {trades.length === 0 ? (
                                    <p style={{ color: 'var(--text-muted)' }}>No trades recorded yet.</p>
                                ) : (
                                    <div className="table-container">
                                        <table>
                                            <thead><tr><th>#</th><th>Date</th><th>Type</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Qty</th><th>P&L</th><th>Reason</th></tr></thead>
                                            <tbody>
                                                {trades.map((t, i) => (
                                                    <tr key={t.id}>
                                                        <td style={{ color: 'var(--text-muted)' }}>{trades.length - i}</td>
                                                        <td>{t.trade_date || '—'}</td>
                                                        <td><span style={{ padding: '2px 8px', borderRadius: '4px', background: t.option_type === 'CE' ? 'var(--accent-green-dim)' : 'var(--accent-red-dim)', color: t.option_type === 'CE' ? 'var(--accent-green)' : 'var(--accent-red)', fontSize: '12px', fontWeight: 600 }}>{t.option_type} {t.position_type}</span></td>
                                                        <td>{t.trading_symbol || t.token}</td>
                                                        <td>₹{t.entry_price.toFixed(2)}</td>
                                                        <td>{t.exit_price ? `₹${t.exit_price.toFixed(2)}` : '—'}</td>
                                                        <td>{t.quantity}</td>
                                                        <td className={t.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}>{t.pnl >= 0 ? '+' : ''}₹{t.pnl.toFixed(2)}</td>
                                                        <td style={{ color: 'var(--text-secondary)', fontSize: '12px' }}>{t.close_reason || '—'}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {tab === 'settings' && (
                        <motion.div key="settings" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                            <div className="grid-2">
                                <div className="glass" style={{ padding: '32px' }}>
                                    <h3 style={{ marginBottom: '24px' }}>⚙️ Trading Settings</h3>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                        <div className="input-group">
                                            <label>Default Quantity (lots)</label>
                                            <input className="input mono" type="number" min="1" max="100" value={qty} onChange={e => setQty(parseInt(e.target.value) || 1)} />
                                        </div>
                                        <div className="input-group">
                                            <label>Option Price Min (₹)</label>
                                            <input className="input mono" type="number" value={priceMin} onChange={e => setPriceMin(parseFloat(e.target.value) || 0)} />
                                        </div>
                                        <div className="input-group">
                                            <label>Option Price Max (₹)</label>
                                            <input className="input mono" type="number" value={priceMax} onChange={e => setPriceMax(parseFloat(e.target.value) || 0)} />
                                        </div>
                                        <button className="btn btn-primary" onClick={handleSaveSettings} disabled={settingsSaving}>
                                            {settingsSaving ? <span className="spinner" /> : '💾 Save Settings'}
                                        </button>
                                    </div>
                                </div>
                                <div className="glass" style={{ padding: '32px' }}>
                                    <h3 style={{ marginBottom: '24px' }}>👤 Account Info</h3>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Name</div>
                                            <div style={{ fontSize: '16px', fontWeight: 600 }}>{user?.full_name}</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Email</div>
                                            <div className="mono" style={{ fontSize: '14px' }}>{user?.email}</div>
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>Aadhaar</div>
                                            {user?.is_verified ? <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>✅ Verified</span> : <span style={{ color: 'var(--accent-red)' }}>❌ Not verified</span>}
                                        </div>
                                        <div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '4px' }}>AngelOne</div>
                                            {user?.has_angelone ? <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>🔗 Connected</span> : <span style={{ color: 'var(--accent-red)' }}>❌ Not configured</span>}
                                        </div>
                                        <div style={{ borderTop: '1px solid var(--border-glass)', paddingTop: '16px', marginTop: '8px' }}>
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '8px' }}>Trading Window</div>
                                            <div className="mono" style={{ fontSize: '14px' }}>{settings?.trading_start_time} – {settings?.trading_end_time}</div>
                                            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>Auto square-off: {settings?.square_off_time}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
