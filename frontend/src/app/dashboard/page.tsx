'use client';

import { useEffect, useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
    api, clearTokens,
    type PortfolioData, type TradeRecord, type UserProfile,
    type UserSettings, type MarketData, type CandleData,
} from '@/lib/api';
// lightweight-charts is imported dynamically in useEffect (SSR-safe)

type Tab = 'overview' | 'charts' | 'trades' | 'settings';
type ChartTarget = 'NIFTY' | 'CE' | 'PE';

/* ─── Indicator math ─── */
function calcEMA(data: number[], period: number): (number | null)[] {
    const k = 2 / (period + 1);
    const ema: (number | null)[] = [];
    let prev: number | null = null;
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) { ema.push(null); continue; }
        if (prev === null) {
            prev = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
        } else {
            prev = data[i] * k + prev * (1 - k);
        }
        ema.push(prev);
    }
    return ema;
}

function calcSMA(data: number[], period: number): (number | null)[] {
    const sma: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) { sma.push(null); continue; }
        const slice = data.slice(i - period + 1, i + 1);
        sma.push(slice.reduce((a, b) => a + b, 0) / period);
    }
    return sma;
}

function calcRSI(data: number[], period = 14): (number | null)[] {
    const rsi: (number | null)[] = [];
    const gains: number[] = [];
    const losses: number[] = [];
    for (let i = 0; i < data.length; i++) {
        if (i === 0) { rsi.push(null); continue; }
        const diff = data[i] - data[i - 1];
        gains.push(diff > 0 ? diff : 0);
        losses.push(diff < 0 ? -diff : 0);
        if (i < period) { rsi.push(null); continue; }
        if (i === period) {
            const avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
            const avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
            rsi.push(avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
        } else {
            const prevRsi = rsi[rsi.length - 1];
            if (prevRsi === null) { rsi.push(null); continue; }
            const avgGain = (gains[gains.length - 2] * (period - 1) + gains[gains.length - 1]) / period;
            const avgLoss = (losses[losses.length - 2] * (period - 1) + losses[losses.length - 1]) / period;
            rsi.push(avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
        }
    }
    return rsi;
}

function calcBollinger(data: number[], period = 20, mult = 2) {
    const upper: (number | null)[] = [];
    const lower: (number | null)[] = [];
    const middle: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) { upper.push(null); lower.push(null); middle.push(null); continue; }
        const slice = data.slice(i - period + 1, i + 1);
        const mean = slice.reduce((a, b) => a + b, 0) / period;
        const std = Math.sqrt(slice.reduce((a, b) => a + (b - mean) ** 2, 0) / period);
        middle.push(mean);
        upper.push(mean + mult * std);
        lower.push(mean - mult * std);
    }
    return { upper, middle, lower };
}

/* ─── Timestamp parser ─── */
function parseTime(ts: number | string): number {
    if (!ts && ts !== 0) return 0;
    if (typeof ts === 'number') return ts; // Already Unix timestamp
    const d = new Date(ts);
    return isNaN(d.getTime()) ? 0 : Math.floor(d.getTime() / 1000);
}

export default function DashboardPage() {
    const router = useRouter();
    const [user, setUser] = useState<UserProfile | null>(null);
    const [portfolio, setPortfolio] = useState<PortfolioData | null>(null);
    const [trades, setTrades] = useState<TradeRecord[]>([]);
    const [settings, setSettings] = useState<UserSettings | null>(null);
    const [market, setMarket] = useState<MarketData | null>(null);
    const [tab, setTab] = useState<Tab>('overview');
    const [loading, setLoading] = useState(true);
    const [botLoading, setBotLoading] = useState(false);
    const [settingsSaving, setSettingsSaving] = useState(false);
    const [message, setMessage] = useState('');

    // Chart state
    const [chartTarget, setChartTarget] = useState<ChartTarget>('NIFTY');
    const [candles, setCandles] = useState<CandleData[]>([]);
    const [indicators, setIndicators] = useState<{ [key: string]: boolean }>({
        ema9: true, ema21: true, sma50: false, rsi: false, bollinger: false,
    });
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<any>(null);
    const candleSeriesRef = useRef<any>(null);
    const indicatorSeriesRef = useRef<any[]>([]);
    const rsiChartRef = useRef<HTMLDivElement>(null);
    const rsiChartApiRef = useRef<any>(null);

    // Settings state
    const [qty, setQty] = useState(1);
    const [priceMin, setPriceMin] = useState(110);
    const [priceMax, setPriceMax] = useState(150);

    /* ─── Load initial data ─── */
    const loadData = useCallback(async () => {
        try {
            const u = await api.getMe();
            if (!u.is_verified || !u.has_angelone) { router.push('/setup'); return; }
            const [p, t, s] = await Promise.all([api.getPortfolio(), api.getTrades(), api.getSettings()]);
            setUser(u); setPortfolio(p); setTrades(t); setSettings(s);
            setQty(s.default_quantity); setPriceMin(s.price_min); setPriceMax(s.price_max);
        } catch { clearTokens(); router.push('/auth/login'); }
        setLoading(false);
    }, [router]);

    useEffect(() => { loadData(); }, [loadData]);

    /* ─── Poll market data every 1s ─── */
    useEffect(() => {
        let active = true;
        const poll = async () => {
            while (active) {
                try { const m = await api.getMarketData(); if (active) setMarket(m); }
                catch { /* ignore */ }
                await new Promise(r => setTimeout(r, 1000));
            }
        };
        poll();
        return () => { active = false; };
    }, []);

    /* ─── Refresh portfolio every 5s ─── */
    useEffect(() => {
        const iv = setInterval(() => {
            api.getPortfolio().then(setPortfolio).catch(() => { });
        }, 5000);
        return () => clearInterval(iv);
    }, []);

    /* ─── Load candles when chart tab or target changes ─── */
    useEffect(() => {
        if (tab !== 'charts') return;
        let symbolKey = 'NIFTY';
        if (chartTarget === 'CE' && market?.symbols?.CE) symbolKey = market.symbols.CE.symbol;
        else if (chartTarget === 'PE' && market?.symbols?.PE) symbolKey = market.symbols.PE.symbol;

        api.getCandles(symbolKey).then(res => setCandles(res.candles)).catch(() => setCandles([]));

        const iv = setInterval(() => {
            api.getCandles(symbolKey).then(res => setCandles(res.candles)).catch(() => { });
        }, 5000);
        return () => clearInterval(iv);
    }, [tab, chartTarget, market?.symbols?.CE?.symbol, market?.symbols?.PE?.symbol]);

    /* ─── Render chart (dynamic import to avoid SSR crash) ─── */
    useEffect(() => {
        if (tab !== 'charts' || !chartContainerRef.current || candles.length === 0) return;
        let cancelled = false;

        // Clean up old chart
        try {
            if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; }
        } catch { chartRef.current = null; }
        try {
            if (rsiChartApiRef.current) { rsiChartApiRef.current.remove(); rsiChartApiRef.current = null; }
        } catch { rsiChartApiRef.current = null; }

        import('lightweight-charts').then(({ createChart, ColorType, CrosshairMode }) => {
            if (cancelled || !chartContainerRef.current) return;

            const container = chartContainerRef.current;
            const chartHeight = indicators.rsi ? 320 : 420;

            const chart = createChart(container, {
                width: container.clientWidth,
                height: chartHeight,
                layout: { background: { type: ColorType.Solid, color: '#0d1117' }, textColor: '#8b949e' },
                grid: { vertLines: { color: '#161b22' }, horzLines: { color: '#161b22' } },
                crosshair: { mode: CrosshairMode.Normal },
                rightPriceScale: { borderColor: '#21262d' },
                timeScale: { borderColor: '#21262d', timeVisible: true, secondsVisible: false },
            });
            chartRef.current = chart;

            const candleSeries = chart.addCandlestickSeries({
                upColor: '#00d26a', downColor: '#f23645',
                borderUpColor: '#00d26a', borderDownColor: '#f23645',
                wickUpColor: '#00d26a', wickDownColor: '#f23645',
            });

            const chartData = candles.map(c => ({
                time: parseTime(c.time) as any,
                open: c.open, high: c.high, low: c.low, close: c.close,
            })).filter(c => c.time > 0).sort((a, b) => a.time - b.time);

            // Deduplicate
            const seen = new Set<number>();
            const unique = chartData.filter(c => { if (seen.has(c.time)) return false; seen.add(c.time); return true; });

            candleSeries.setData(unique);
            candleSeriesRef.current = candleSeries;
            indicatorSeriesRef.current = [];

            const closes = candles.map(c => c.close);
            const times = unique.map(c => c.time);

            // EMA 9
            if (indicators.ema9) {
                const ema9 = calcEMA(closes, 9);
                const series = chart.addLineSeries({ color: '#f7931a', lineWidth: 1, title: 'EMA 9' });
                const lineData = ema9.map((v, i) => v !== null && times[i] ? { time: times[i], value: v } : null).filter(Boolean) as any[];
                series.setData(lineData);
                indicatorSeriesRef.current.push(series);
            }

            // EMA 21
            if (indicators.ema21) {
                const ema21 = calcEMA(closes, 21);
                const series = chart.addLineSeries({ color: '#2962ff', lineWidth: 1, title: 'EMA 21' });
                const lineData = ema21.map((v, i) => v !== null && times[i] ? { time: times[i], value: v } : null).filter(Boolean) as any[];
                series.setData(lineData);
                indicatorSeriesRef.current.push(series);
            }

            // SMA 50
            if (indicators.sma50) {
                const sma50 = calcSMA(closes, 50);
                const series = chart.addLineSeries({ color: '#e040fb', lineWidth: 1, title: 'SMA 50' });
                const lineData = sma50.map((v, i) => v !== null && times[i] ? { time: times[i], value: v } : null).filter(Boolean) as any[];
                series.setData(lineData);
                indicatorSeriesRef.current.push(series);
            }

            // Bollinger Bands
            if (indicators.bollinger) {
                const bb = calcBollinger(closes, 20, 2);
                const upper = chart.addLineSeries({ color: 'rgba(76, 175, 80, 0.4)', lineWidth: 1, title: 'BB Upper' });
                const lower = chart.addLineSeries({ color: 'rgba(244, 67, 54, 0.4)', lineWidth: 1, title: 'BB Lower' });
                const mid = chart.addLineSeries({ color: 'rgba(255, 255, 255, 0.2)', lineWidth: 1, title: 'BB Mid' });
                upper.setData(bb.upper.map((v, i) => v !== null && times[i] ? { time: times[i], value: v } : null).filter(Boolean) as any[]);
                lower.setData(bb.lower.map((v, i) => v !== null && times[i] ? { time: times[i], value: v } : null).filter(Boolean) as any[]);
                mid.setData(bb.middle.map((v, i) => v !== null && times[i] ? { time: times[i], value: v } : null).filter(Boolean) as any[]);
                indicatorSeriesRef.current.push(upper, lower, mid);
            }

            chart.timeScale().fitContent();

            // RSI sub-chart
            if (indicators.rsi && rsiChartRef.current) {
                const rsiChart = createChart(rsiChartRef.current, {
                    width: container.clientWidth,
                    height: 100,
                    layout: { background: { type: ColorType.Solid, color: '#0d1117' }, textColor: '#8b949e' },
                    grid: { vertLines: { color: '#161b22' }, horzLines: { color: '#161b22' } },
                    rightPriceScale: { borderColor: '#21262d' },
                    timeScale: { borderColor: '#21262d', timeVisible: true, visible: false },
                });
                rsiChartApiRef.current = rsiChart;
                const rsiData = calcRSI(closes, 14);
                const rsiSeries = rsiChart.addLineSeries({ color: '#e040fb', lineWidth: 1, title: 'RSI 14' });
                rsiSeries.setData(rsiData.map((v, i) => v !== null && times[i] ? { time: times[i], value: v } : null).filter(Boolean) as any[]);

                // Overbought/oversold lines
                const ob = rsiChart.addLineSeries({ color: 'rgba(244,67,54,0.3)', lineWidth: 1 });
                ob.setData(times.map(t => ({ time: t, value: 70 })));
                const os = rsiChart.addLineSeries({ color: 'rgba(76,175,80,0.3)', lineWidth: 1 });
                os.setData(times.map(t => ({ time: t, value: 30 })));

                rsiChart.timeScale().fitContent();

                // Sync time scales
                chart.timeScale().subscribeVisibleLogicalRangeChange((range: any) => {
                    if (range) rsiChart.timeScale().setVisibleLogicalRange(range);
                });
            }

            // Resize handler
            const onResize = () => {
                chart.applyOptions({ width: container.clientWidth });
                if (rsiChartApiRef.current) rsiChartApiRef.current.applyOptions({ width: container.clientWidth });
            };
            window.addEventListener('resize', onResize);
        });

        return () => {
            cancelled = true;
            try { if (chartRef.current) { chartRef.current.remove(); chartRef.current = null; } } catch { /* */ }
            try { if (rsiChartApiRef.current) { rsiChartApiRef.current.remove(); rsiChartApiRef.current = null; } } catch { /* */ }
        };
    }, [tab, candles, indicators]);

    /* ─── Handlers ─── */
    const handleBotControl = async (action: string) => {
        setBotLoading(true); setMessage('');
        try {
            const res = await api.controlBot(action);
            setMessage(`Bot ${res.status}`);
            setTimeout(loadData, 1500);
        } catch (e: any) { setMessage(e.message); }
        setBotLoading(false);
    };

    const handleSaveSettings = async () => {
        setSettingsSaving(true);
        try {
            await api.updateSettings({ default_quantity: qty, price_min: priceMin, price_max: priceMax });
            setMessage('Settings saved!'); setTimeout(() => setMessage(''), 3000);
        } catch (e: any) { setMessage(e.message); }
        setSettingsSaving(false);
    };

    const toggleIndicator = (key: string) => {
        setIndicators(prev => ({ ...prev, [key]: !prev[key] }));
    };

    const logout = () => { clearTokens(); router.push('/auth/login'); };

    if (loading) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', gap: '12px' }}>
                <span className="spinner" /> <span style={{ color: 'var(--text-secondary)' }}>Loading dashboard...</span>
            </div>
        );
    }

    const pos = market?.position || portfolio?.open_position;
    const botStatus = portfolio?.bot_status || 'stopped';
    const isBotRunning = botStatus === 'running';
    const niftyPrice = market?.nifty_price || 0;
    const atr = market?.atr || 0;
    const atrOk = atr >= 6.9;

    return (
        <div className="page-container">
            {/* ─── Navbar ─── */}
            <nav className="navbar">
                <div className="navbar-inner">
                    <Link href="/dashboard" className="navbar-brand">
                        <span className="logo-icon">📈</span>
                        <span>Trade<span style={{ color: 'var(--accent-green)' }}>Pulse</span></span>
                    </Link>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        {niftyPrice > 0 && (
                            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                                <div className="mono" style={{ fontSize: '14px', color: 'var(--accent-blue)' }}>
                                    NIFTY <span style={{ fontWeight: 700, fontSize: '16px' }}>{niftyPrice.toFixed(2)}</span>
                                </div>
                                <div className="mono" style={{ fontSize: '13px', color: atrOk ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                    ATR {atr.toFixed(2)} {atrOk ? '✅' : '⚠️'}
                                </div>
                            </div>
                        )}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className={`status-dot ${isBotRunning ? 'live' : 'offline'}`} />
                            <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{isBotRunning ? 'Live' : 'Offline'}</span>
                        </div>
                        <span style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{user?.full_name}</span>
                        <button className="btn btn-outline btn-sm" onClick={logout}>Logout</button>
                    </div>
                </div>
            </nav>

            <div className="container" style={{ marginTop: '24px' }}>
                {/* ─── Market Info Bar ─── */}
                <div className="market-bar glass" style={{ display: 'flex', gap: '16px', padding: '16px 24px', marginBottom: '20px', alignItems: 'center', flexWrap: 'wrap' }}>
                    <div style={{ flex: 1, minWidth: '140px' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>NIFTY 50</div>
                        <div className="mono" style={{ fontSize: '22px', fontWeight: 700, color: 'var(--accent-blue)' }}>
                            {niftyPrice > 0 ? niftyPrice.toFixed(2) : '—'}
                        </div>
                    </div>
                    <div style={{ flex: 1, minWidth: '120px' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>ATR (RMA 100)</div>
                        <div className="mono" style={{ fontSize: '20px', fontWeight: 700, color: atrOk ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                            {atr > 0 ? atr.toFixed(2) : '—'}
                            <span style={{ fontSize: '12px', marginLeft: '6px' }}>{atr > 0 ? (atrOk ? 'TRADABLE' : 'LOW') : ''}</span>
                        </div>
                    </div>
                    {market?.symbols?.CE && (
                        <div style={{ flex: 1, minWidth: '180px', cursor: 'pointer' }} onClick={() => { setChartTarget('CE'); setTab('charts'); }}>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                                CE <span style={{ color: 'var(--accent-green)' }}>▸</span>
                            </div>
                            <div className="mono" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--accent-green)' }}>
                                {market.symbols.CE.symbol}
                            </div>
                            <div className="mono" style={{ fontSize: '16px', fontWeight: 700 }}>
                                ₹{market.symbols.CE.price > 0 ? market.symbols.CE.price.toFixed(2) : '—'}
                            </div>
                        </div>
                    )}
                    {market?.symbols?.PE && (
                        <div style={{ flex: 1, minWidth: '180px', cursor: 'pointer' }} onClick={() => { setChartTarget('PE'); setTab('charts'); }}>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>
                                PE <span style={{ color: 'var(--accent-red)' }}>▸</span>
                            </div>
                            <div className="mono" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--accent-red)' }}>
                                {market.symbols.PE.symbol}
                            </div>
                            <div className="mono" style={{ fontSize: '16px', fontWeight: 700 }}>
                                ₹{market.symbols.PE.price > 0 ? market.symbols.PE.price.toFixed(2) : '—'}
                            </div>
                        </div>
                    )}
                    {market?.signals && (
                        <div style={{ flex: 0, minWidth: '80px', textAlign: 'center' }}>
                            <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px' }}>Signal</div>
                            <div style={{ fontSize: '20px', marginTop: '4px' }}>
                                {market.signals.buy ? '🟢 BUY' : market.signals.sell ? '🔴 SELL' : '⚪ —'}
                            </div>
                        </div>
                    )}
                </div>

                {/* ─── Tabs ─── */}
                <div style={{ display: 'flex', gap: '4px', marginBottom: '20px' }}>
                    {(['overview', 'charts', 'trades', 'settings'] as Tab[]).map(t => (
                        <button key={t} className={`btn btn-sm ${tab === t ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setTab(t)} style={{ textTransform: 'capitalize' }}>
                            {t === 'overview' ? '📊 Overview' : t === 'charts' ? '📈 Charts' : t === 'trades' ? '📜 Trades' : '⚙️ Settings'}
                        </button>
                    ))}
                </div>

                {message && (
                    <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                        className={`alert ${message.toLowerCase().includes('error') || message.toLowerCase().includes('fail') ? 'alert-error' : 'alert-success'}`}
                        style={{ marginBottom: '16px' }}>
                        {message}
                    </motion.div>
                )}

                <AnimatePresence mode="wait">
                    {/* ════════════ OVERVIEW ════════════ */}
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
                                    <div className="stat-value" style={{ fontSize: '18px' }}>
                                        <span className={`status-dot ${isBotRunning ? 'live' : 'offline'}`} style={{ marginRight: '8px' }} />
                                        {botStatus.toUpperCase()}
                                    </div>
                                    <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
                                        <button className={`btn btn-sm ${isBotRunning ? 'btn-danger' : 'btn-primary'}`}
                                            onClick={() => handleBotControl(isBotRunning ? 'stop' : 'start')} disabled={botLoading}>
                                            {botLoading ? <span className="spinner" /> : (isBotRunning ? '⏹ Stop' : '▶ Start')}
                                        </button>
                                    </div>
                                </motion.div>
                            </div>

                            {/* Open Position */}
                            <div className="glass" style={{ padding: '24px', marginBottom: '24px' }}>
                                <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    {pos ? <>📍 Open Position <span className="status-dot live" /></> : '📍 No Open Position'}
                                </h3>
                                {pos ? (
                                    <div className="grid-3" style={{ gap: '28px' }}>
                                        <div>
                                            <div className="field-label">Symbol</div>
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
                                            <div className="field-label">Entry / Current</div>
                                            <div className="mono" style={{ fontSize: '16px' }}>₹{pos.entry_price?.toFixed(2)} → ₹{pos.current_price?.toFixed(2)}</div>
                                        </div>
                                        <div>
                                            <div className="field-label">Live P&L</div>
                                            <motion.div className={`mono ${pos.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`} style={{ fontSize: '24px' }}
                                                key={pos.pnl} initial={{ scale: 1.15 }} animate={{ scale: 1 }} transition={{ type: 'spring' }}>
                                                {pos.pnl >= 0 ? '+' : ''}₹{pos.pnl?.toFixed(2)}
                                            </motion.div>
                                        </div>
                                        <div>
                                            <div className="field-label">Stop Loss</div>
                                            <div className="mono text-red" style={{ fontSize: '16px' }}>₹{pos.stop_loss?.toFixed(2)}</div>
                                        </div>
                                        <div>
                                            <div className="field-label">Quantity</div>
                                            <div className="mono" style={{ fontSize: '16px' }}>{pos.quantity}</div>
                                        </div>
                                        <div>
                                            <div className="field-label">Entry Time</div>
                                            <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>{pos.entry_time}</div>
                                        </div>
                                    </div>
                                ) : (
                                    <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
                                        {isBotRunning ? 'Waiting for signal... (ATR must be ≥ 6.9)' : 'Start the bot to begin trading.'}
                                    </p>
                                )}
                            </div>

                            {/* Recent Trades */}
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

                    {/* ════════════ CHARTS ════════════ */}
                    {tab === 'charts' && (
                        <motion.div key="charts" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                            <div className="glass" style={{ padding: '20px' }}>
                                {/* Chart symbol switcher */}
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '12px' }}>
                                    <div style={{ display: 'flex', gap: '6px' }}>
                                        <button className={`btn btn-sm ${chartTarget === 'NIFTY' ? 'btn-primary' : 'btn-outline'}`}
                                            onClick={() => setChartTarget('NIFTY')}>
                                            📊 NIFTY
                                        </button>
                                        {market?.symbols?.CE && (
                                            <button className={`btn btn-sm ${chartTarget === 'CE' ? 'btn-primary' : 'btn-outline'}`}
                                                onClick={() => setChartTarget('CE')} style={{ borderColor: 'var(--accent-green)' }}>
                                                🟢 {market.symbols.CE.symbol.slice(-10)}
                                            </button>
                                        )}
                                        {market?.symbols?.PE && (
                                            <button className={`btn btn-sm ${chartTarget === 'PE' ? 'btn-primary' : 'btn-outline'}`}
                                                onClick={() => setChartTarget('PE')} style={{ borderColor: 'var(--accent-red)' }}>
                                                🔴 {market.symbols.PE.symbol.slice(-10)}
                                            </button>
                                        )}
                                    </div>
                                    <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                                        {[
                                            { key: 'ema9', label: 'EMA 9', color: '#f7931a' },
                                            { key: 'ema21', label: 'EMA 21', color: '#2962ff' },
                                            { key: 'sma50', label: 'SMA 50', color: '#e040fb' },
                                            { key: 'bollinger', label: 'BB', color: '#4caf50' },
                                            { key: 'rsi', label: 'RSI', color: '#e040fb' },
                                        ].map(ind => (
                                            <button key={ind.key}
                                                className={`btn btn-sm ${indicators[ind.key] ? '' : 'btn-outline'}`}
                                                onClick={() => toggleIndicator(ind.key)}
                                                style={{
                                                    fontSize: '11px', padding: '4px 10px',
                                                    background: indicators[ind.key] ? ind.color + '22' : undefined,
                                                    borderColor: ind.color,
                                                    color: indicators[ind.key] ? ind.color : 'var(--text-secondary)',
                                                }}>
                                                {ind.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>

                                {/* Chart title */}
                                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>
                                    {chartTarget === 'NIFTY' ? 'NIFTY 50 Index' :
                                        chartTarget === 'CE' ? (market?.symbols?.CE?.symbol || 'CE Option') :
                                            (market?.symbols?.PE?.symbol || 'PE Option')} — 1 Min Candles
                                </div>

                                {/* Main chart container */}
                                <div ref={chartContainerRef} className="chart-container" />

                                {/* RSI sub-chart */}
                                {indicators.rsi && (
                                    <div style={{ marginTop: '4px' }}>
                                        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>RSI (14)</div>
                                        <div ref={rsiChartRef} className="chart-container" />
                                    </div>
                                )}

                                {candles.length === 0 && (
                                    <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                                        <div style={{ fontSize: '40px', marginBottom: '12px' }}>📊</div>
                                        <div>No candle data available. Start the bot to begin receiving market data.</div>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    )}

                    {/* ════════════ TRADES ════════════ */}
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

                    {/* ════════════ SETTINGS ════════════ */}
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
                                            <div className="field-label">Name</div>
                                            <div style={{ fontSize: '16px', fontWeight: 600 }}>{user?.full_name}</div>
                                        </div>
                                        <div>
                                            <div className="field-label">Email</div>
                                            <div className="mono" style={{ fontSize: '14px' }}>{user?.email}</div>
                                        </div>
                                        <div>
                                            <div className="field-label">Verification</div>
                                            {user?.is_verified ? <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>✅ Verified</span> : <span style={{ color: 'var(--accent-red)' }}>❌ Not verified</span>}
                                        </div>
                                        <div>
                                            <div className="field-label">AngelOne</div>
                                            {user?.has_angelone ? <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>🔗 Connected</span> : <span style={{ color: 'var(--accent-red)' }}>❌ Not configured</span>}
                                        </div>
                                        <div style={{ borderTop: '1px solid var(--border-glass)', paddingTop: '16px', marginTop: '8px' }}>
                                            <div className="field-label">Trading Window</div>
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
