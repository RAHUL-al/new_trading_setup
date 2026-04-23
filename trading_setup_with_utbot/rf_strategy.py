"""
rf_strategy.py — Random Forest + UT Bot + StochRSI (NIFTY 2-MIN)

IMPROVEMENTS:
  1. Chandelier Exit — trails from highest high (longs) / lowest low (shorts)
  2. Entry cooldown — 5 candles after SL hit, prevents whipsaw re-entries
  3. Wider initial SL — max(ATR × 2.0, 18 pts), gives trades room to breathe
  4. Trend alignment — EMA20 > EMA50 required for LONG (vice-versa for SHORT)
  5. Morning bias feature — direction of first 90 min as model input
  6. RF confidence — only enters when model ≥55% probability

ZERO LOOK-AHEAD:
  Labels use forward EMA slope [training only].
  Backtest iterates candle-by-candle using only past/current data.
"""

import pandas as pd
import numpy as np
import argparse, pickle, json, os, calendar
from datetime import datetime, time as dt_time
import warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ─── Config ───
ATR_PERIOD        = 14
ATR_KEY_VALUE     = 1.0
ATR_SL_MULT       = 2.0        # initial SL = max(ATR * 2.0, MIN_SL)
MIN_SL_POINTS     = 18.0
MIN_ATR           = 12.0
COOLDOWN_CANDLES  = 5           # bars to wait after SL hit

OBSERVATION_END   = dt_time(13, 0)
ENTRY_START       = dt_time(13, 0)
ENTRY_END         = dt_time(15, 12)
SQUARE_OFF        = dt_time(15, 20)

STOCH_BUY  = 10;  STOCH_SELL = 90
STOCH_RSI_PERIOD = 14;  STOCH_K_SMOOTH = 3;  STOCH_D_SMOOTH = 3
RF_CONFIDENCE    = 0.55
RF_TREES         = 500
REGIME_FWD       = 30;  REGIME_THRESH = 0.02

LOT_SIZE = 65;  BASE_LOTS = 2
HIGH_PTS = 30;  MID_PTS = 10
ATR_LEVELS = [6, 8, 10, 12, 14, 16, 18, 20]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════ INDICATORS ═══════════════
def rma(s, p): return s.ewm(alpha=1/p, adjust=False).mean()

def atr(df, p=14):
    h,l,c = df['High'].astype(float), df['Low'].astype(float), df['Close'].astype(float)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    tr.iloc[0] = (h-l).iloc[0]; return rma(tr, p)

def rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0.0); l = (-d).where(d<0,0.0)
    return 100 - 100/(1 + rma(g,p)/rma(l,p).replace(0,1e-10))

def stoch_rsi(close, rp=14, sp=14, ks=3, ds=3):
    r = rsi(close, rp)
    rl = r.rolling(sp).min(); rh = r.rolling(sp).max()
    sr = (r - rl) / (rh - rl).replace(0,1e-10) * 100
    k = sr.rolling(ks).mean(); d = k.rolling(ds).mean()
    return k, d

def ut_bot(close, atr_v, kv=1.0):
    c = close.values if hasattr(close,'values') else np.array(close)
    a = atr_v.values if hasattr(atr_v,'values') else np.array(atr_v)
    n = len(c); ts = np.zeros(n); dr = np.zeros(n)
    ts[0] = c[0]; dr[0] = 1
    for i in range(1, n):
        nl = a[i]*kv
        if dr[i-1]==1:
            ts[i] = max(c[i]-nl, ts[i-1])
            if c[i] < ts[i]: dr[i]=-1; ts[i]=c[i]+nl
            else: dr[i]=1
        else:
            ts[i] = min(c[i]+nl, ts[i-1])
            if c[i] > ts[i]: dr[i]=1; ts[i]=c[i]-nl
            else: dr[i]=-1
    return ts, dr

def ema(s, p): return s.ewm(span=p, adjust=False).mean()

# ═══════════════ FEATURES ═══════════════
def build_features(df):
    c = df['Close'].astype(float); h = df['High'].astype(float)
    l = df['Low'].astype(float); o = df['Open'].astype(float)
    _atr = atr(df, ATR_PERIOD); _rsi = rsi(c, 14)
    sk, sd = stoch_rsi(c, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD, STOCH_K_SMOOTH, STOCH_D_SMOOTH)
    tr, dr = ut_bot(c, _atr, ATR_KEY_VALUE)
    e5=ema(c,5); e10=ema(c,10); e20=ema(c,20); e50=ema(c,50)

    f = pd.DataFrame(index=df.index)
    f['atr']=_atr; f['rsi']=_rsi; f['stoch_k']=sk; f['stoch_d']=sd
    f['stoch_kd_diff']=sk-sd; f['ut_dir']=dr; f['close_vs_trail']=c.values-tr

    uc = np.diff(dr, prepend=dr[0])
    f['ut_change']=uc; f['ut_buy']=(uc>0).astype(int); f['ut_sell']=(uc<0).astype(int)
    f['stoch_os']=(sk<STOCH_BUY).astype(int); f['stoch_ob']=(sk>STOCH_SELL).astype(int)
    f['stoch_kxu']=((sk>sd)&(sk.shift(1)<=sd.shift(1))).astype(int)
    f['stoch_kxd']=((sk<sd)&(sk.shift(1)>=sd.shift(1))).astype(int)

    f['mom3']=c.pct_change(3)*100; f['mom5']=c.pct_change(5)*100
    f['mom10']=c.pct_change(10)*100; f['mom20']=c.pct_change(20)*100

    f['body']=c-o; f['body_pct']=(c-o)/o*100
    f['uwick']=h-c.where(c>o,o); f['lwick']=c.where(c<o,o)-l
    f['rng']=h-l; f['body_rng']=(c-o).abs()/(h-l).replace(0,1e-10)

    f['std5']=c.rolling(5).std(); f['std10']=c.rolling(10).std(); f['std20']=c.rolling(20).std()
    f['atr_pctl']=_atr.rolling(50).apply(lambda x:(x.iloc[-1]>x).mean()*100, raw=False)

    f['c_e5']=c-e5; f['c_e10']=c-e10; f['c_e20']=c-e20; f['c_e50']=c-e50
    f['e5_e10']=e5-e10; f['e10_e20']=e10-e20; f['e20_e50']=e20-e50
    f['e20_slope']=e20.diff(3)/3; f['e50_slope']=e50.diff(5)/5
    f['ema_aligned_bull']=((e5>e10)&(e10>e20)&(e20>e50)).astype(int)
    f['ema_aligned_bear']=((e5<e10)&(e10<e20)&(e20<e50)).astype(int)

    f['c_h5']=c-h.rolling(5).max(); f['c_l5']=c-l.rolling(5).min()
    f['c_h10']=c-h.rolling(10).max(); f['c_l10']=c-l.rolling(10).min()

    bull=(c>o).astype(int)
    f['bull_r10']=bull.rolling(10).sum()/10; f['bull_r20']=bull.rolling(20).sum()/20
    f['rsi_slope5']=_rsi.diff(5); f['price_slope5']=c.diff(5)
    f['day_return']=0.0; f['morning_bias']=0.0

    return f

# ═══════════════ LABELS ═══════════════
def regime_labels(df, fwd=30, thresh=0.02):
    c = df['Close'].astype(float); e = ema(c,20); n = len(df)
    lab = np.zeros(n, dtype=int)
    for i in range(n-fwd):
        if e.iloc[i]==0: continue
        s = (e.iloc[i+fwd]-e.iloc[i])/e.iloc[i]*100
        if s>thresh: lab[i]=1
        elif s<-thresh: lab[i]=-1
    return lab

# ═══════════════ TRAINING ═══════════════
def train_rf(X_tr, y_tr, X_te, y_te, n=500):
    m = RandomForestClassifier(n_estimators=n, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=0)
    print(f"\n  Training RF ({n} trees)...")
    m.fit(X_tr, y_tr+1)
    tr_p = m.predict(X_tr)-1; te_p = m.predict(X_te)-1
    print(f"  Train: {accuracy_score(y_tr,tr_p)*100:.1f}% | Test: {accuracy_score(y_te,te_p)*100:.1f}%")
    print(classification_report(y_te+1, te_p+1, target_names=['BEAR','CHOP','BULL'], zero_division=0))
    return m

# ═══════════════ BACKTEST (CHANDELIER + COOLDOWN) ═══════════════
def backtest(df, regime, proba, sk_v, ut_v, atr_v,
             sb=10, ss=90, min_atr=12.0, rf_conf=0.55, cooldown=5):
    close=df['Close'].astype(float); hi=df['High'].astype(float); lo=df['Low'].astype(float)
    # EMA for trend alignment
    e20 = ema(close, 20); e50 = ema(close, 50)

    pos=None; trades=[]; daily={}; skipped=[]
    prev_date=None; lots=BASE_LOTS; acc_loss=0.0; recovering=False
    last_sl_idx = -999; last_sl_dir = None

    for i in range(len(df)):
        t = df.iloc[i]['Time'].time(); cd = df.iloc[i]['Time'].date()
        c=float(close.iloc[i]); h=float(hi.iloc[i]); l=float(lo.iloc[i])
        ca=float(atr_v[i]); cv20=float(e20.iloc[i]); cv50=float(e50.iloc[i])

        # Day boundary
        if prev_date and cd!=prev_date:
            if pos:
                tr = _mk(pos, float(close.iloc[i-1]), df.iloc[i-1]['Time'], "DAY_END", lots)
                trades.append(tr); _ad(daily, prev_date, tr); pos=None
            if cd not in daily: daily[cd]={'trades':[],'pnl':0,'lots':lots}
            if prev_date in daily:
                pp = daily[prev_date]['pnl']
                if pp<0: acc_loss+=pp; recovering=True; lots+=2
                elif pp>0 and recovering:
                    acc_loss+=pp
                    if acc_loss>=0: lots=BASE_LOTS; acc_loss=0; recovering=False
            if cd in daily: daily[cd]['lots']=lots
        prev_date = cd
        if cd not in daily: daily[cd]={'trades':[],'pnl':0,'lots':lots}

        # Square off
        if pos and t>=SQUARE_OFF:
            tr=_mk(pos,c,df.iloc[i]['Time'],"SQUARE_OFF",lots)
            trades.append(tr); _ad(daily,cd,tr); pos=None; continue

        inw = ENTRY_START<=t<=ENTRY_END

        # SL check
        if pos:
            if pos['dir']=="LONG" and l<=pos['sl']:
                tr=_mk(pos,pos['sl'],df.iloc[i]['Time'],"TRAIL_SL",lots)
                trades.append(tr); _ad(daily,cd,tr)
                last_sl_idx=i; last_sl_dir=pos['dir']; pos=None
            elif pos['dir']=="SHORT" and h>=pos['sl']:
                tr=_mk(pos,pos['sl'],df.iloc[i]['Time'],"TRAIL_SL",lots)
                trades.append(tr); _ad(daily,cd,tr)
                last_sl_idx=i; last_sl_dir=pos['dir']; pos=None

        # Chandelier trailing stop (from highest high / lowest low)
        if pos:
            if pos['dir']=="LONG":
                pos['best'] = max(pos.get('best',h), h)
                new_sl = pos['best'] - ca * ATR_KEY_VALUE
                if new_sl > pos['sl']: pos['sl'] = new_sl
            else:
                pos['best'] = min(pos.get('best',l), l)
                new_sl = pos['best'] + ca * ATR_KEY_VALUE
                if new_sl < pos['sl']: pos['sl'] = new_sl

        reg = regime[i]; sk = sk_v[i] if not np.isnan(sk_v[i]) else 50
        ud = ut_v[i]; conf = proba[i] if i<len(proba) else 0.5

        # Regime flip
        if reg==1 and pos and pos['dir']=="SHORT":
            tr=_mk(pos,c,df.iloc[i]['Time'],"REGIME_FLIP",lots); trades.append(tr); _ad(daily,cd,tr); pos=None
        elif reg==-1 and pos and pos['dir']=="LONG":
            tr=_mk(pos,c,df.iloc[i]['Time'],"REGIME_FLIP",lots); trades.append(tr); _ad(daily,cd,tr); pos=None

        # Entry
        if not pos and inw:
            # Cooldown check
            in_cooldown = (i - last_sl_idx) < cooldown

            b_reg=(reg==1); b_sk=(sk<sb); b_ut=(ud==1); b_atr=(ca>=min_atr)
            b_conf=(conf>=rf_conf); b_ema=(cv20>cv50)  # trend aligned
            s_reg=(reg==-1); s_sk=(sk>ss); s_ut=(ud==-1); s_ema=(cv20<cv50)

            entered=False
            if b_reg and b_sk and b_ut and b_atr and b_conf and b_ema and not in_cooldown:
                isl = max(ca*ATR_SL_MULT, MIN_SL_POINTS)
                pos={'dir':'LONG','entry':c,'sl':c-isl,'initial_sl':c-isl,
                     'entry_time':df.iloc[i]['Time'],'entry_idx':i,'best':h}
                entered=True
            elif s_reg and s_sk and s_ut and b_atr and b_conf and s_ema and not in_cooldown:
                isl = max(ca*ATR_SL_MULT, MIN_SL_POINTS)
                pos={'dir':'SHORT','entry':c,'sl':c+isl,'initial_sl':c+isl,
                     'entry_time':df.iloc[i]['Time'],'entry_idx':i,'best':l}
                entered=True

            if not entered:
                sr = []
                if b_reg:
                    if not b_sk: sr.append(f"StochK={sk:.0f} (need<{sb})")
                    if not b_ut: sr.append("UT_Bot bearish")
                    if not b_atr: sr.append(f"ATR={ca:.1f} (<{min_atr})")
                    if not b_conf: sr.append(f"Conf={conf:.0%} (<{rf_conf:.0%})")
                    if not b_ema: sr.append("EMA20<EMA50")
                    if in_cooldown: sr.append(f"Cooldown({cooldown-(i-last_sl_idx)})")
                elif s_reg:
                    if not s_sk: sr.append(f"StochK={sk:.0f} (need>{ss})")
                    if not s_ut: sr.append("UT_Bot bullish")
                    if not b_atr: sr.append(f"ATR={ca:.1f} (<{min_atr})")
                    if not b_conf: sr.append(f"Conf={conf:.0%} (<{rf_conf:.0%})")
                    if not s_ema: sr.append("EMA20>EMA50")
                    if in_cooldown: sr.append(f"Cooldown({cooldown-(i-last_sl_idx)})")

                if sr and len(sr)<=3 and (b_reg or s_reg):
                    ts_v = int(calendar.timegm(df.iloc[i]['Time'].timetuple()))
                    skipped.append({'time':ts_v,'price':round(c,2),
                        'potential_dir':'LONG' if b_reg else 'SHORT',
                        'reasons':sr,'atr':round(ca,1),'stoch_k':round(sk,1),
                        'ut_dir':int(ud),'rf_conf':round(conf,3)})

    return trades, daily, skipped

def _pnl(p,x): return (x-p['entry']) if p['dir']=="LONG" else (p['entry']-x)
def _mk(p,x,t,r,lots=2):
    rp=_pnl(p,x); m=lots//BASE_LOTS
    tr={'dir':p['dir'],'entry':p['entry'],'exit':round(x,2),
        'entry_time':p['entry_time'],'exit_time':t,
        'pnl':round(rp*m,2),'raw_pnl':round(rp,2),
        'lots':lots,'qty':lots*LOT_SIZE,'reason':r}
    if 'initial_sl' in p: tr['sl']=round(p['initial_sl'],2)
    return tr
def _ad(d,dt,tr):
    if dt not in d: d[dt]={'trades':[],'pnl':0,'lots':tr.get('lots',BASE_LOTS)}
    d[dt]['trades'].append(tr); d[dt]['pnl']+=tr['pnl']

# ═══════════════ MULTI-ATR ═══════════════
def multi_atr(df_t, reg, prb, sk, ut, at, levels, sb, ss, rc):
    res={}
    for lv in levels:
        tr,dy,sk_s = backtest(df_t,reg,prb,sk,ut,at,sb,ss,float(lv),rc)
        w=sum(1 for t in tr if t['pnl']>0); tp=sum(t['pnl'] for t in tr)
        print(f"    ATR≥{lv:>2}: {len(tr):>3}T W:{w} P&L:{tp:+.2f}")
        res[str(lv)]={'trades':tr,'daily':dy,'skipped':sk_s}
    return res

# ═══════════════ EXPORT ═══════════════
def export_json(df_t, results, sk_v, sd_v, at_v, ut_v, reg, prb):
    candles=[]
    for _,r in df_t.iterrows():
        ts=int(calendar.timegm(r['Time'].timetuple()))
        candles.append({'time':ts,'open':round(float(r['Open']),2),
            'high':round(float(r['High']),2),'low':round(float(r['Low']),2),
            'close':round(float(r['Close']),2)})

    stoch=[]; atr_d=[]
    for i in range(len(df_t)):
        ts=int(calendar.timegm(df_t.iloc[i]['Time'].timetuple()))
        _k=float(sk_v[i]) if not np.isnan(sk_v[i]) else None
        _d=float(sd_v[i]) if not np.isnan(sd_v[i]) else None
        stoch.append({'time':ts,'k':round(_k,2) if _k else None,'d':round(_d,2) if _d else None})
        atr_d.append({'time':ts,'value':round(float(at_v[i]),2)})

    atr_res={}
    for lv,r in results.items():
        tr=r['trades']; sk_s=r['skipped']
        marks=[]
        for t in tr:
            et=int(calendar.timegm(t['entry_time'].timetuple())) if hasattr(t['entry_time'],'timetuple') else 0
            xt=int(calendar.timegm(t['exit_time'].timetuple())) if hasattr(t['exit_time'],'timetuple') else 0
            marks.append({'type':'entry','time':et,'price':t['entry'],'dir':t['dir'],'pnl':t['pnl'],'reason':t['reason']})
            marks.append({'type':'exit','time':xt,'price':t['exit'],'dir':t['dir'],'pnl':t['pnl'],'reason':t['reason']})

        w=sum(1 for t in tr if t['pnl']>0); ls=sum(1 for t in tr if t['pnl']<=0)
        tp=sum(t['pnl'] for t in tr)
        gp=sum(t['pnl'] for t in tr if t['pnl']>0) or 0
        gl=abs(sum(t['pnl'] for t in tr if t['pnl']<=0)) or 1
        cum=0;pk=0;dd=0
        for t in tr: cum+=t['pnl']; pk=max(pk,cum); dd=max(dd,pk-cum)

        hi=[t for t in tr if abs(t['raw_pnl'])>=HIGH_PTS]
        mi=[t for t in tr if MID_PTS<=abs(t['raw_pnl'])<HIGH_PTS]
        lo=[t for t in tr if abs(t['raw_pnl'])<MID_PTS]

        tl=[{'dir':t['dir'],'entry':t['entry'],'exit':t['exit'],
             'entry_time':t['entry_time'].isoformat() if hasattr(t['entry_time'],'isoformat') else str(t['entry_time']),
             'exit_time':t['exit_time'].isoformat() if hasattr(t['exit_time'],'isoformat') else str(t['exit_time']),
             'pnl':t['pnl'],'raw_pnl':t['raw_pnl'],'reason':t['reason']} for t in tr]

        atr_res[lv]={'markers':marks,'trades':tl,'skipped':sk_s[:200],
            'summary':{'total_trades':len(tr),'wins':w,'losses':ls,
                'total_pnl':round(tp,2),'win_rate':round(w/max(len(tr),1)*100,1),
                'profit_factor':round(gp/gl,2),'max_drawdown':round(dd,2),
                'avg_win':round(np.mean([t['pnl'] for t in tr if t['pnl']>0]),2) if w else 0,
                'avg_loss':round(np.mean([t['pnl'] for t in tr if t['pnl']<=0]),2) if ls else 0,
                'high_pts':len(hi),'mid_pts':len(mi),'low_pts':len(lo)}}

    data={'candles':candles,'stochRSI':stoch,'atr':atr_d,'atr_results':atr_res,
          'atr_levels':[str(l) for l in ATR_LEVELS],
          'config':{'stochBuy':STOCH_BUY,'stochSell':STOCH_SELL,'defaultATR':MIN_ATR,
                    'entryStart':str(ENTRY_START),'entryEnd':str(ENTRY_END),
                    'squareOff':str(SQUARE_OFF),'rfConfidence':RF_CONFIDENCE}}

    op=os.path.join(SCRIPT_DIR,"frontend_data.json")
    with open(op,'w') as f: json.dump(data,f)
    print(f"💾 {op} ({os.path.getsize(op)/1024/1024:.1f}MB)")

# ═══════════════ PRINT ═══════════════
def show(tr, dy, label=""):
    if not tr: print("❌ No trades"); return
    n=len(tr); w=[t for t in tr if t['pnl']>0]; ls=[t for t in tr if t['pnl']<=0]
    tp=sum(t['pnl'] for t in tr)
    gp=sum(t['pnl'] for t in w) or 0; gl=abs(sum(t['pnl'] for t in ls)) or 1
    pl=[t['pnl'] for t in tr]; cm=np.cumsum(pl); pk=np.maximum.accumulate(cm); dd=(pk-cm).max()
    td=sum(1 for d in dy.values() if d['trades']); wd=sum(1 for d in dy.values() if d['pnl']>0)

    print(f"\n{'='*60}")
    print(f"  🌲 RF [2-MIN] — {label}")
    print(f"{'='*60}")
    print(f"  {n} trades | WR: {len(w)/n*100:.1f}% | P&L: {tp:+.2f}")
    print(f"  PF: {gp/gl:.2f} | DD: {dd:.2f} | Days: {td} (Win: {wd})")

    for day in sorted(dy.keys()):
        t=dy[day]['trades']
        if not t: continue
        dp=dy[day]['pnl']; ic="✅" if dp>0 else "❌"
        print(f"  {day} L:{dy[day].get('lots',2)} T:{len(t)} P&L:{dp:+.2f} {ic}")
    print(f"  TOTAL: {tp:+.2f}")
    print(f"{'='*60}")

# ═══════════════ MAIN ═══════════════
def main():
    global ATR_KEY_VALUE,MIN_ATR,ENTRY_START,ENTRY_END,SQUARE_OFF,STOCH_BUY,STOCH_SELL,RF_CONFIDENCE

    pa = argparse.ArgumentParser()
    pa.add_argument("--file", default=os.path.join(SCRIPT_DIR,"nifty_2min_data.csv"))
    pa.add_argument("--min-atr", type=float, default=MIN_ATR)
    pa.add_argument("--stoch-buy", type=float, default=STOCH_BUY)
    pa.add_argument("--stoch-sell", type=float, default=STOCH_SELL)
    pa.add_argument("--rf-trees", type=int, default=RF_TREES)
    pa.add_argument("--rf-confidence", type=float, default=RF_CONFIDENCE)
    pa.add_argument("--obs-end", default="13:00")
    pa.add_argument("--window-end", default="15:12")
    pa.add_argument("--square-off", default="15:20")
    pa.add_argument("--test-from", default="2026-03-01")
    pa.add_argument("--regime-fwd", type=int, default=REGIME_FWD)
    pa.add_argument("--regime-thresh", type=float, default=REGIME_THRESH)
    pa.add_argument("--cooldown", type=int, default=COOLDOWN_CANDLES)
    a = pa.parse_args()

    MIN_ATR=a.min_atr; STOCH_BUY=a.stoch_buy; STOCH_SELL=a.stoch_sell; RF_CONFIDENCE=a.rf_confidence
    def pt(s): h,m=map(int,s.split(':')); return dt_time(h,m)
    ENTRY_START=pt(a.obs_end); ENTRY_END=pt(a.window_end); SQUARE_OFF=pt(a.square_off)
    test_from=datetime.strptime(a.test_from,"%Y-%m-%d").date()

    print(f"{'='*60}")
    print(f"  🌲 RF + StochRSI + UT Bot + Chandelier [2-MIN]")
    print(f"{'='*60}")
    print(f"  Window: {a.obs_end}→{a.window_end} | ATR≥{MIN_ATR} | Cooldown: {a.cooldown}")
    print(f"  StochRSI: <{STOCH_BUY}/>{ STOCH_SELL} | RF≥{RF_CONFIDENCE:.0%}")

    df=pd.read_csv(a.file); df['Time']=pd.to_datetime(df['Time'])
    df=df.sort_values('Time').reset_index(drop=True); c=df['Close'].astype(float)
    print(f"\n  {len(df):,} candles | {df['Time'].dt.date.nunique()} days")

    feat=build_features(df)
    # Fill day_return + morning_bias
    for day,grp in df.groupby(df['Time'].dt.date):
        do=float(grp.iloc[0]['Open'])
        if do>0: feat.loc[grp.index,'day_return']=(c.loc[grp.index]-do)/do*100
        # Morning bias: avg return from 9:15 to 13:00
        morning = grp[grp['Time'].dt.time < dt_time(13,0)]
        if len(morning)>5:
            mb = (float(morning.iloc[-1]['Close'])-do)/do*100
            feat.loc[grp.index,'morning_bias']=mb

    lab=regime_labels(df, a.regime_fwd, a.regime_thresh)
    print(f"  Labels: B={sum(lab==1)} S={sum(lab==-1)} C={sum(lab==0)}")

    times=df['Time'].dt.time; dates=df['Time'].dt.date
    wm=(times>=dt_time(9,20))&(times<=ENTRY_END)
    tr_m=(dates<test_from)&wm; te_m=(dates>=test_from)&wm
    feat=feat.fillna(0).replace([np.inf,-np.inf],0)
    X_tr,y_tr=feat[tr_m].values,lab[tr_m]; X_te,y_te=feat[te_m].values,lab[te_m]
    print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

    if len(X_tr)==0 or len(X_te)==0: print("❌ No data!"); return

    model=train_rf(X_tr,y_tr,X_te,y_te,a.rf_trees)
    top=sorted(zip(feat.columns,model.feature_importances_),key=lambda x:-x[1])[:10]
    print(f"\n  TOP FEATURES:"); [print(f"    {n:>20}: {v:.4f}") for n,v in top]

    pred=np.zeros(len(df),dtype=int); prob=np.full(len(df),0.33)
    pred[te_m]=model.predict(X_te)-1; prob[te_m]=model.predict_proba(X_te).max(axis=1)

    at_v=atr(df,ATR_PERIOD).values
    sk_v,sd_v=stoch_rsi(c,STOCH_RSI_PERIOD,STOCH_RSI_PERIOD,STOCH_K_SMOOTH,STOCH_D_SMOOTH)
    sk_v,sd_v=sk_v.values,sd_v.values
    _,ut_v=ut_bot(c,atr(df,ATR_PERIOD),ATR_KEY_VALUE)

    tm=dates>=test_from; df_t=df[tm].reset_index(drop=True)
    p_t=pred[tm]; pb_t=prob[tm]; a_t=at_v[tm]; sk_t=sk_v[tm]; sd_t=sd_v[tm]; ut_t=ut_v[tm]

    print(f"\n  🔄 Multi-ATR:")
    res=multi_atr(df_t,p_t,pb_t,sk_t,ut_t,a_t,ATR_LEVELS,STOCH_BUY,STOCH_SELL,RF_CONFIDENCE)

    mk=str(int(MIN_ATR))
    if mk in res: show(res[mk]['trades'],res[mk]['daily'],f"ATR≥{mk}")

    export_json(df_t,res,sk_t,sd_t,a_t,ut_t,p_t,pb_t)

    with open(os.path.join(SCRIPT_DIR,"rf_model.pkl"),'wb') as f: pickle.dump(model,f)
    meta={"strategy":"RF+Chandelier+StochRSI+UTBot (2MIN)","min_atr":MIN_ATR,
          "rf_confidence":RF_CONFIDENCE,"test_from":a.test_from,
          "last_trained":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(SCRIPT_DIR,"model_metadata.json"),'w') as f: json.dump(meta,f,indent=2)
    print("✅ Done")

if __name__=="__main__": main()
