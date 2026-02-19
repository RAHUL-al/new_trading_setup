/* API client and auth helpers for the frontend. */

const API_BASE = '/api';

export interface UserProfile {
    id: number;
    email: string;
    full_name: string;
    phone: string | null;
    is_verified: boolean;
    is_active: boolean;
    has_angelone: boolean;
    created_at: string;
}

export interface TokenResponse {
    access_token: string;
    refresh_token: string;
    token_type: string;
    user: UserProfile;
}

export interface PortfolioData {
    total_trades: number;
    today_trades: number;
    today_pnl: number;
    total_pnl: number;
    win_rate: number;
    open_position: any | null;
    bot_status: string;
}

export interface TradeRecord {
    id: number;
    token: string;
    trading_symbol: string;
    option_type: string;
    position_type: string;
    entry_price: number;
    exit_price: number | null;
    quantity: number;
    entry_time: string;
    exit_time: string | null;
    stop_loss: number | null;
    pnl: number;
    close_reason: string | null;
    trade_date: string | null;
}

export interface UserSettings {
    default_quantity: number;
    price_min: number;
    price_max: number;
    trading_start_time: string;
    trading_end_time: string;
    square_off_time: string;
}

export interface SymbolInfo {
    symbol: string;
    token: string;
    price: number;
}

export interface MarketData {
    nifty_price: number;
    atr: number;
    symbols: { CE?: SymbolInfo; PE?: SymbolInfo };
    signals: { buy: boolean; sell: boolean };
    last_candle: { timestamp: string; open: number; high: number; low: number; close: number } | null;
    position: any | null;
    error?: string;
}

export interface CandleData {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export interface CandleResponse {
    symbol: string;
    date: string;
    candles: CandleData[];
}

function getToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('access_token');
}

export function setTokens(access: string, refresh: string) {
    localStorage.setItem('access_token', access);
    localStorage.setItem('refresh_token', refresh);
}

export function clearTokens() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
}

async function fetchAPI(url: string, options: RequestInit = {}) {
    const token = getToken();
    const headers: any = { 'Content-Type': 'application/json', ...options.headers };
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(`${API_BASE}${url}`, { ...options, headers });

    if (res.status === 401) {
        clearTokens();
        if (typeof window !== 'undefined') window.location.href = '/auth/login';
        throw new Error('Unauthorized');
    }

    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(err.detail || 'API Error');
    }

    return res.json();
}

// Auth
export const api = {
    signup: (data: { email: string; full_name: string; phone?: string; password: string }) =>
        fetchAPI('/auth/signup', { method: 'POST', body: JSON.stringify(data) }),

    login: (data: { email: string; password: string }) =>
        fetchAPI('/auth/login', { method: 'POST', body: JSON.stringify(data) }),

    getMe: () => fetchAPI('/auth/me'),

    sendEmailOTP: () =>
        fetchAPI('/auth/email/send-otp', { method: 'POST' }),

    verifyEmailOTP: (otp: string) =>
        fetchAPI('/auth/email/verify-otp', { method: 'POST', body: JSON.stringify({ otp }) }),

    getEmailStatus: () => fetchAPI('/auth/email/status'),

    saveAngelOneCreds: (data: { api_key: string; client_id: string; password: string; totp_secret: string }) =>
        fetchAPI('/auth/angelone/credentials', { method: 'POST', body: JSON.stringify(data) }),

    getAngelOneStatus: () => fetchAPI('/auth/angelone/status'),

    // Trading
    getPortfolio: (): Promise<PortfolioData> => fetchAPI('/trading/portfolio'),
    getTrades: (limit = 50, offset = 0): Promise<TradeRecord[]> => fetchAPI(`/trading/trades?limit=${limit}&offset=${offset}`),
    getBotStatus: () => fetchAPI('/trading/bot/status'),
    controlBot: (action: string) => fetchAPI('/trading/bot/control', { method: 'POST', body: JSON.stringify({ action }) }),
    getSettings: (): Promise<UserSettings> => fetchAPI('/trading/settings'),
    updateSettings: (data: Partial<UserSettings>) => fetchAPI('/trading/settings', { method: 'PUT', body: JSON.stringify(data) }),

    // Market Data (live from Redis)
    getMarketData: (): Promise<MarketData> => fetchAPI('/trading/market-data'),
    getCandles: (symbolKey: string): Promise<CandleResponse> => fetchAPI(`/trading/candles/${symbolKey}`),
};

// WebSocket
export function createWSConnection(onMessage: (data: any) => void): WebSocket | null {
    const token = getToken();
    if (!token) return null;

    const wsUrl = `ws://localhost:8000/ws?token=${token}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            onMessage(msg);
        } catch (e) { /* ignore parse errors */ }
    };

    ws.onerror = () => { /* reconnect handled by component */ };

    return ws;
}
