#!/usr/bin/env python3
"""
chart_trades.py — Visual chart of S&R zone scalp trades + ORB

Generates 4 charts:
  1. RESPECT trade example   — zone holds, scalp wins
  2. BREAK/STOP trade        — zone breaks, scalp stopped
  3. ORB breakout trade      — ORB fires after zone rejected earlier
  4. Full day combo          — ORB + scalp running simultaneously

Uses real ES 1-min price data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from datetime import time, date
import warnings
warnings.filterwarnings('ignore')

DATA  = "/Users/ishanbhardwaj/cme_competition_system/output/mtf/ES_NQ_1min_aligned.parquet"
OFILE = "/Users/ishanbhardwaj/cme_competition_system/output/trade_charts.png"


# ─── Load data ────────────────────────────────────────────────────────────────

def load_day(target_date):
    df = pd.read_parquet(DATA)
    df["ts_et"]   = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date_et"] = df["ts_et"].dt.date
    df["time_et"] = df["ts_et"].dt.time
    df = df.sort_values("ts_et").reset_index(drop=True)
    day = df[df["date_et"]==pd.Timestamp(target_date).date()].copy()
    day = day[(day["time_et"]>=time(9,30))&(day["time_et"]<=time(16,0))].reset_index(drop=True)
    day["bar_num"] = range(len(day))
    return day


# ─── Chart helper ─────────────────────────────────────────────────────────────

def draw_candles(ax, day, t_start, t_end, col_o, col_h, col_l, col_c, color_up='#26a69a', color_dn='#ef5350'):
    sub = day[(day["time_et"]>=t_start)&(day["time_et"]<=t_end)].reset_index(drop=True)
    for i, row in sub.iterrows():
        o, h, l, c = row[col_o], row[col_h], row[col_l], row[col_c]
        color = color_up if c >= o else color_dn
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
        body_lo = min(o, c); body_hi = max(o, c)
        rect = plt.Rectangle((i-0.3, body_lo), 0.6, body_hi-body_lo,
                               color=color, zorder=3)
        ax.add_patch(rect)
    return sub


def zone_rect(ax, lo, hi, x_start, x_end, color, alpha=0.15, label=None):
    ax.axhspan(lo, hi, xmin=x_start/(x_end+0.01), xmax=1.0,
               color=color, alpha=alpha, zorder=1)
    ax.axhline(y=lo, color=color, linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)
    ax.axhline(y=hi, color=color, linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('S&R Zone Scalp + ORB System  —  ES Futures (1-min bars)',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    ax_style = dict(facecolor='#16213e', grid=True)
    for ax in axes.flatten():
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        ax.spines['bottom'].set_color('#333355')
        ax.spines['top'].set_color('#333355')
        ax.spines['left'].set_color('#333355')
        ax.spines['right'].set_color('#333355')
        ax.grid(color='#333355', linewidth=0.4, alpha=0.7)

    COLS = dict(o="ES_open", h="ES_high", l="ES_low", c="ES_close")

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 1: S&R RESPECT trade — PDH acts as resistance, scalp wins
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0,0]
    ax.set_title('Chart 1: S&R RESPECT — PDH Zone Holds (Scalp Wins)',
                 color='white', fontsize=10, fontweight='bold', pad=8)

    day = load_day("2024-09-20")
    prev_day = load_day("2024-09-19")
    prev_rth = prev_day[(prev_day["time_et"]>=time(9,30))&(prev_day["time_et"]<=time(16,0))]
    PDH = prev_rth["ES_high"].max()

    # Show 10:30 to 14:00 for clarity
    sub = day[(day["time_et"]>=time(10,30))&(day["time_et"]<=time(14,30))].reset_index(drop=True)
    for i, row in sub.iterrows():
        o,h,l,c = row[COLS['o']],row[COLS['h']],row[COLS['l']],row[COLS['c']]
        col = '#26a69a' if c>=o else '#ef5350'
        ax.plot([i,i],[l,h],color=col,linewidth=0.8,zorder=2)
        rect=plt.Rectangle((i-0.3,min(o,c)),0.6,abs(c-o),color=col,zorder=3)
        ax.add_patch(rect)

    # PDH zone ±5pts
    ZW = 5
    ax.axhspan(PDH-ZW, PDH+ZW, color='#ff6b6b', alpha=0.15, zorder=1, label=f'PDH Zone (±{ZW}pt)')
    ax.axhline(PDH, color='#ff6b6b', linewidth=1.5, linestyle='-', alpha=0.9, zorder=4, label=f'PDH = {PDH:.2f}')
    ax.axhline(PDH+ZW, color='#ff6b6b', linewidth=0.8, linestyle='--', alpha=0.6, zorder=4)
    ax.axhline(PDH-ZW, color='#ff6b6b', linewidth=0.8, linestyle='--', alpha=0.6, zorder=4)

    # Find approach bar
    approach_idx = None
    for i in range(1, len(sub)-12):
        if (sub.iloc[i-1]['ES_close'] < PDH-ZW and
            sub.iloc[i]['ES_close'] >= PDH-ZW and
            sub.iloc[i]['ES_close'] <= PDH+ZW):
            approach_idx = i
            break

    if approach_idx is not None:
        ep = sub.iloc[approach_idx]['ES_close']
        # Entry marker
        ax.scatter(approach_idx, ep, marker='v', color='#ff6b6b', s=120, zorder=6,
                   label=f'SHORT Entry @ {ep:.2f}')
        # Stop line
        ax.axhline(PDH+ZW+0.25, color='#ff4444', linewidth=1.2, linestyle='-.', alpha=0.9, zorder=5)
        ax.text(len(sub)*0.98, PDH+ZW+0.5, f'STOP @ {PDH+ZW+0.25:.2f}',
                color='#ff4444', fontsize=7, ha='right', va='bottom')
        # Target line
        ax.axhline(PDH-ZW-0.25, color='#00e676', linewidth=1.2, linestyle='-.', alpha=0.9, zorder=5)
        ax.text(len(sub)*0.98, PDH-ZW-0.8, f'TARGET @ {PDH-ZW-0.25:.2f}',
                color='#00e676', fontsize=7, ha='right', va='top')
        # Find exit bar
        for k in range(approach_idx+1, min(approach_idx+11, len(sub))):
            if sub.iloc[k]['ES_close'] < PDH-ZW:
                exit_idx = k
                ax.scatter(exit_idx, PDH-ZW-0.25, marker='o', color='#00e676', s=120, zorder=6,
                           label=f'✅ EXIT TARGET +{ep-(PDH-ZW-0.25):.1f}pts')
                ax.annotate('', xy=(exit_idx, PDH-ZW-0.3), xytext=(approach_idx, ep+0.1),
                           arrowprops=dict(arrowstyle='->', color='#00e676', lw=1.5))
                break
        # Shade trade period
        exit_idx_final = min(approach_idx+10, len(sub)-1)
        ax.axvspan(approach_idx, exit_idx_final, color='#00e676', alpha=0.05)

    ax.text(0.02, 0.97, f'Result: RESPECT ✅\n+{ZW*0.9:.1f}pts gain\nWR = 74.7% of all touches',
            transform=ax.transAxes, color='#00e676', fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))

    ax.legend(loc='upper left', fontsize=7, facecolor='#0d1117', edgecolor='#333355',
              labelcolor='white', framealpha=0.9)
    ax.set_ylabel('ES Price', color='#aaaaaa', fontsize=8)
    ax.set_xlabel('Bar number (1 bar = 1 min)', color='#aaaaaa', fontsize=8)

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 2: S&R BREAK trade — zone breaks, scalp stopped out
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0,1]
    ax.set_title('Chart 2: S&R BREAK — Zone Breaks, Scalp Stopped (−7.4pts)',
                 color='white', fontsize=10, fontweight='bold', pad=8)

    day2 = load_day("2024-11-06")   # post-election rally — strong breakout day
    prev2 = load_day("2024-11-05")
    prev_rth2 = prev2[(prev2["time_et"]>=time(9,30))&(prev2["time_et"]<=time(16,0))]
    PDH2 = prev_rth2["ES_high"].max()

    sub2 = day2[(day2["time_et"]>=time(10,30))&(day2["time_et"]<=time(14,0))].reset_index(drop=True)
    for i, row in sub2.iterrows():
        o,h,l,c = row[COLS['o']],row[COLS['h']],row[COLS['l']],row[COLS['c']]
        col = '#26a69a' if c>=o else '#ef5350'
        ax.plot([i,i],[l,h],color=col,linewidth=0.8,zorder=2)
        rect=plt.Rectangle((i-0.3,min(o,c)),0.6,abs(c-o),color=col,zorder=3)
        ax.add_patch(rect)

    ax.axhspan(PDH2-ZW, PDH2+ZW, color='#ff6b6b', alpha=0.15, zorder=1)
    ax.axhline(PDH2, color='#ff6b6b', linewidth=1.5, alpha=0.9, zorder=4, label=f'PDH = {PDH2:.2f}')
    ax.axhline(PDH2+ZW, color='#ff6b6b', linewidth=0.8, linestyle='--', alpha=0.6, zorder=4)
    ax.axhline(PDH2-ZW, color='#ff6b6b', linewidth=0.8, linestyle='--', alpha=0.6, zorder=4)

    # Find first approach
    approach_idx2 = None
    for i in range(1, len(sub2)-5):
        if (sub2.iloc[i-1]['ES_close'] < PDH2-ZW and
            PDH2-ZW <= sub2.iloc[i]['ES_close'] <= PDH2+ZW):
            approach_idx2 = i
            break

    if approach_idx2 is not None:
        ep2 = sub2.iloc[approach_idx2]['ES_close']
        ax.scatter(approach_idx2, ep2, marker='v', color='#ff6b6b', s=120, zorder=6,
                   label=f'SHORT Entry @ {ep2:.2f}')
        stop_px = PDH2+ZW+0.25
        ax.axhline(stop_px, color='#ff4444', linewidth=1.2, linestyle='-.', alpha=0.9, zorder=5)
        ax.text(len(sub2)*0.98, stop_px+0.3, f'STOP @ {stop_px:.2f}',
                color='#ff4444', fontsize=7, ha='right', va='bottom')
        ax.axhline(PDH2-ZW-0.25, color='#00e676', linewidth=0.8, linestyle='-.', alpha=0.5)
        ax.text(len(sub2)*0.98, PDH2-ZW-0.8, f'TARGET (never reached)',
                color='#00e676', fontsize=7, ha='right', va='top', alpha=0.6)

        # Find stop bar
        for k in range(approach_idx2+1, min(approach_idx2+11, len(sub2))):
            if sub2.iloc[k]['ES_close'] > PDH2+ZW:
                stop_idx = k
                loss = stop_px - ep2
                ax.scatter(stop_idx, stop_px, marker='x', color='#ff4444', s=150, zorder=6,
                           linewidth=2.5, label=f'❌ STOP −{loss:.1f}pts')
                ax.annotate('', xy=(stop_idx, stop_px+0.5), xytext=(approach_idx2, ep2),
                           arrowprops=dict(arrowstyle='->', color='#ff4444', lw=1.5))
                break

        # Show price continuing up after stop (the "run")
        ax.axvspan(approach_idx2, min(approach_idx2+10, len(sub2)-1),
                  color='#ff4444', alpha=0.05)

    ax.text(0.02, 0.97, f'Result: BREAK ❌\n−7.4pts loss avg\nOnly 8.3% of touches\nbut wipes 5-6 wins',
            transform=ax.transAxes, color='#ff4444', fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))

    ax.legend(loc='upper left', fontsize=7, facecolor='#0d1117', edgecolor='#333355',
              labelcolor='white', framealpha=0.9)
    ax.set_ylabel('ES Price', color='#aaaaaa', fontsize=8)
    ax.set_xlabel('Bar number (1 bar = 1 min)', color='#aaaaaa', fontsize=8)

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 3: ORB BREAKOUT — the momentum trade that scalp fades
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1,0]
    ax.set_title('Chart 3: ORB Trade — Breakout With PDH/L Confirmation',
                 color='white', fontsize=10, fontweight='bold', pad=8)

    day3 = load_day("2024-03-14")
    prev3 = load_day("2024-03-13")
    prev_rth3 = prev3[(prev3["time_et"]>=time(9,30))&(prev3["time_et"]<=time(16,0))]
    PDH3 = prev_rth3["ES_high"].max(); PDL3 = prev_rth3["ES_low"].min()

    or_b3 = day3[(day3["time_et"]>=time(9,30))&(day3["time_et"]<time(10,0))]
    if len(or_b3):
        OR_H3 = or_b3["ES_high"].max(); OR_L3 = or_b3["ES_low"].min(); OR_R3=OR_H3-OR_L3
    else:
        OR_H3=PDH3+10; OR_L3=PDH3-30; OR_R3=40

    sub3 = day3[(day3["time_et"]>=time(9,30))&(day3["time_et"]<=time(15,30))].reset_index(drop=True)
    for i, row in sub3.iterrows():
        o,h,l,c = row[COLS['o']],row[COLS['h']],row[COLS['l']],row[COLS['c']]
        col = '#26a69a' if c>=o else '#ef5350'
        ax.plot([i,i],[l,h],color=col,linewidth=0.6,zorder=2)
        rect=plt.Rectangle((i-0.3,min(o,c)),0.6,abs(c-o),color=col,zorder=3)
        ax.add_patch(rect)

    # OR zone shading
    or_end_idx = sub3[sub3["time_et"]==time(10,0)].index[0] if len(sub3[sub3["time_et"]==time(10,0)]) else 30
    ax.axvspan(0, or_end_idx, color='#4444ff', alpha=0.1, label='Opening Range (9:30-10:00)')
    ax.axhspan(OR_L3, OR_H3, color='#4444ff', alpha=0.12, zorder=1, label='OR Zone')
    ax.axhline(OR_H3, color='#6699ff', linewidth=1.5, zorder=4, label=f'OR High = {OR_H3:.2f}')
    ax.axhline(OR_L3, color='#6699ff', linewidth=1.0, linestyle='--', zorder=4, alpha=0.7)

    # PDH level
    ax.axhline(PDH3, color='#ff6b6b', linewidth=1.0, linestyle=':', alpha=0.7,
               label=f'PDH = {PDH3:.2f}')

    # Find ORB breakout bar (after 10:00)
    orb_idx3 = None
    for i in range(or_end_idx, min(or_end_idx+30, len(sub3)-10)):
        if sub3.iloc[i]['ES_high'] > OR_H3:
            orb_idx3 = i
            break

    if orb_idx3 is not None:
        ep3 = sub3.iloc[orb_idx3]['ES_close']
        sp3 = ep3 - OR_R3
        tp3 = ep3 + 3*OR_R3

        ax.scatter(orb_idx3, ep3, marker='^', color='#00e676', s=150, zorder=7,
                   label=f'LONG Entry @ {ep3:.2f}')
        ax.axhline(sp3, color='#ff4444', linewidth=1.2, linestyle='-.', zorder=5)
        ax.axhline(tp3, color='#00e676', linewidth=1.2, linestyle='-.', zorder=5)
        ax.text(len(sub3)*0.02, sp3-0.3, f'Stop  −1R @ {sp3:.0f}',
                color='#ff4444', fontsize=7, va='top')
        ax.text(len(sub3)*0.02, tp3+0.3, f'Target +3R @ {tp3:.0f}',
                color='#00e676', fontsize=7, va='bottom')

        # Check if target reached
        for k in range(orb_idx3+1, len(sub3)):
            if sub3.iloc[k]['ES_high'] >= tp3:
                ax.scatter(k, tp3, marker='*', color='#ffd700', s=200, zorder=8,
                           label=f'✅ TARGET HIT +3R')
                ax.axvspan(orb_idx3, k, color='#00e676', alpha=0.06)
                break
            elif sub3.iloc[k]['ES_low'] <= sp3:
                ax.scatter(k, sp3, marker='x', color='#ff4444', s=150, zorder=8, linewidth=2.5,
                           label=f'❌ STOPPED')
                break

        # Entry annotations
        ax.annotate(f'OR Range = {OR_R3:.1f}pts\nRisk = 1R = {OR_R3:.1f}pts\nTarget = 3R = {3*OR_R3:.1f}pts',
                   xy=(orb_idx3, ep3), xytext=(orb_idx3+15, ep3-OR_R3*1.5),
                   color='white', fontsize=7,
                   arrowprops=dict(arrowstyle='->', color='white', lw=1),
                   bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))

    ax.legend(loc='upper left', fontsize=7, facecolor='#0d1117', edgecolor='#333355',
              labelcolor='white', framealpha=0.9)
    ax.set_ylabel('ES Price', color='#aaaaaa', fontsize=8)
    ax.set_xlabel('Bar number (1 bar = 1 min)', color='#aaaaaa', fontsize=8)

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 4: THE PAYOFF PROBLEM — wins vs losses distribution
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1,1]
    ax.set_title('Chart 4: Why High Win Rate ≠ Profit — Payoff Asymmetry',
                 color='white', fontsize=10, fontweight='bold', pad=8)

    # Simulate 100 trades with the known stats
    np.random.seed(42)
    n_target = 73; n_stop = 8; n_time = 19
    wins   = [np.random.normal(1.26, 0.4) for _ in range(n_target)]
    stops  = [np.random.normal(-7.42, 1.5) for _ in range(n_stop)]
    times  = [np.random.normal(-2.69, 1.2) for _ in range(n_time)]

    all_pnl = wins + stops + times
    colors_pnl = (['#00e676']*n_target + ['#ff4444']*n_stop + ['#ff9900']*n_time)

    x_pos = range(len(all_pnl))
    bars = ax.bar(x_pos, all_pnl, color=colors_pnl, width=0.8, alpha=0.85, zorder=3)

    ax.axhline(0, color='white', linewidth=0.8, zorder=4)

    # Running cumulative
    cum = np.cumsum(all_pnl)
    ax2 = ax.twinx()
    ax2.set_facecolor('#16213e')
    ax2.plot(x_pos, cum, color='#ffffff', linewidth=1.5, label='Cumulative P&L', zorder=5, alpha=0.7)
    ax2.axhline(0, color='white', linewidth=0.5, linestyle=':')
    ax2.set_ylabel('Cumulative P&L (pts)', color='#aaaaaa', fontsize=8)
    ax2.tick_params(colors='#aaaaaa', labelsize=8)
    ax2.spines['bottom'].set_color('#333355')
    ax2.spines['top'].set_color('#333355')
    ax2.spines['left'].set_color('#333355')
    ax2.spines['right'].set_color('#333355')

    # Annotations
    ax.text(n_target//2, 2.2,
            f'73 TARGET wins\n@ +1.26pts avg\n= +{sum(wins):.0f}pts total',
            ha='center', color='#00e676', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))
    ax.text(n_target+3, -9,
            f'8 STOPS\n@ −7.42pts avg\n= −{abs(sum(stops)):.0f}pts total',
            ha='left', color='#ff4444', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))
    ax.text(n_target+n_stop+2, -5,
            f'19 TIME STOPS\n@ −2.69pts avg\n= −{abs(sum(times)):.0f}pts total',
            ha='left', color='#ff9900', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.8))

    total = sum(all_pnl)
    ax.text(0.98, 0.05,
            f'NET P&L: {total:+.0f}pts\n(−0.21 per trade × 100)\n\nWin Rate: {n_target}%\nBut NEGATIVE EV',
            transform=ax.transAxes, color='#ff4444' if total < 0 else '#00e676',
            fontsize=9, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.9))

    # Legend patches
    patches = [
        mpatches.Patch(color='#00e676', label=f'TARGET wins ({n_target}%)  +{np.mean(wins):.2f}pts avg'),
        mpatches.Patch(color='#ff4444', label=f'STOP losses ({n_stop}%)    {np.mean(stops):.2f}pts avg'),
        mpatches.Patch(color='#ff9900', label=f'TIME stops ({n_time}%)   {np.mean(times):.2f}pts avg'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=7,
              facecolor='#0d1117', edgecolor='#333355', labelcolor='white', framealpha=0.9)
    ax.set_ylabel('P&L per trade (ES points)', color='#aaaaaa', fontsize=8)
    ax.set_xlabel('Trade number (100 simulated trades)', color='#aaaaaa', fontsize=8)

    # ─── Final layout ──────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OFILE, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print(f"Chart saved to: {OFILE}")
    plt.close()


if __name__ == "__main__":
    main()
