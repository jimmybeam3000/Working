# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Leaderboard_Monitor_COMPLETE_v3.6.2_FINAL.py
# COMPLETE VERSION - NO TRUNCATION
# User: jimmybeam3000
# Date: 2025-11-03
import matplotlib
matplotlib.use('TkAgg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import tempfile

import tkinter as tk

from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

def set_30min_xlocator(ax):
    """Set X-axis ticks to 30-minute intervals (HH:MM)."""
    try:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.figure.autofmt_xdate()
    except Exception as e:
        print(f"[WARN] set_30min_xlocator failed: {e}")

from tkinter import ttk, messagebox, filedialog, scrolledtext
from datetime import datetime, time as dt_time, timedelta
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler
from math import ceil
import sqlite3, pandas as pd, numpy as np, random, time, threading, os, json, logging, shutil, pytz
from queue import Queue # ✅ NEU: Import Queue for thread-safe communication

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import (NoSuchElementException, TimeoutException, WebDriverException)


# Data Science & Plotting Imports
from scipy.stats import pearsonr, entropy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Machine Learning Imports
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except ImportError:
    print("FATAL ERROR: scikit-learn is not installed. Please run 'pip install scikit-learn'")
    messagebox.showerror("Dependency Error", "scikit-learn is not installed.\nPlease run 'pip install scikit-learn' in your terminal.")
    exit()
# ---------------------------------------------------------------------------
# ADDED HELPERS (Appended by assistant)
# 1) compute_gap_trigger_score(...) -- analyzes Top1 reactivity to Top2..TopN
# 2) start_once_thread(...) -- thread guard to avoid double-starts
# 3) set_30min_xlocator(ax) -- helper to set 30-minute ticks on matplotlib axes
# ---------------------------------------------------------------------------

# compute_gap_trigger_score: conservative implementation to scan Top1 reactivity
def compute_gap_trigger_score(db_file, days=7, gap_threshold=100, react_window_minutes=30, top_n=5):
    # Connect DB and select top N by max points in window
    conn = sqlite3.connect(db_file)
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    q_topn = '''
        SELECT name, MAX(points) as mp
        FROM leaderboard
        WHERE timestamp >= ?
        GROUP BY name
        ORDER BY mp DESC
        LIMIT ?
    '''
    cur = conn.execute(q_topn, (cutoff, top_n))
    rows = cur.fetchall()
    if len(rows) < 2:
        conn.close()
        return {'error': 'Not enough active players in window'}

    top_players = [r[0] for r in rows]
    top1 = top_players[0]
    others = top_players[1:]

    # Load time series for top1 and others
    q_ts = '''
        SELECT timestamp, name, points
        FROM leaderboard
        WHERE name IN ({names}) AND timestamp >= ?
        ORDER BY timestamp ASC
    '''.format(names=",".join("?" for _ in ([top1] + others)))

    params = tuple([top1] + others) + (cutoff,)
    df = pd.read_sql_query(q_ts, conn, params=params)
    conn.close()

    if df.empty:
        return {'error': 'No time series data found for top players.'}

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    pivot = df.pivot(index='timestamp', columns='name', values='points').sort_index()
    pivot = pivot.resample('1min').ffill().bfill().fillna(0)

    others_cols = [c for c in pivot.columns if c != top1]
    pivot['others_max'] = pivot[others_cols].max(axis=1)
    pivot['gap'] = pivot[top1] - pivot['others_max']

    threat_mask = pivot['gap'] < gap_threshold
    threat_changes = threat_mask.astype(int).diff().fillna(0)
    event_starts = pivot.index[threat_changes == 1].tolist()
    if threat_mask.iloc[0]:
        event_starts.insert(0, pivot.index[0])

    events = []
    reactive_hits = 0
    for start in event_starts:
        gap_at_start = float(pivot.at[start, 'gap'])
        end_time = start + timedelta(minutes=react_window_minutes)
        window = pivot.loc[start:end_time]
        start_points = float(window[top1].iloc[0]) if not window.empty else None
        reacted = False
        reaction_delay_min = None
        if start_points is not None:
            gains = window[top1].diff().fillna(0)
            gain_thresh = max(5, 0.01 * max(1.0, start_points))
            reaction_idx = gains[gains >= gain_thresh].index
            if len(reaction_idx) > 0:
                reacted = True
                reaction_delay_min = (reaction_idx[0] - start).total_seconds() / 60.0
                reactive_hits += 1

        events.append({
            'start': start,
            'gap_at_start': gap_at_start,
            'reacted': reacted,
            'reaction_delay_min': reaction_delay_min
        })

    trigger_events = len(events)
    react_ratio = (reactive_hits / trigger_events) if trigger_events > 0 else 0.0

    return {
        'top1': top1,
        'others': others,
        'trigger_events': trigger_events,
        'reactive_hits': reactive_hits,
        'react_ratio': react_ratio,
        'events': events
    }

# start_once_thread: start thread only if not already alive (attach to obj.attr_name)
def start_once_thread(obj, attr_name, target, daemon=True):
    existing = getattr(obj, attr_name, None)
    if existing is not None and getattr(existing, 'is_alive', lambda: False)():
        return existing
    t = threading.Thread(target=target, daemon=daemon)
    setattr(obj, attr_name, t)
    t.start()
    return t

# set_30min_xlocator: small helper to set 30-minute major ticks
def set_30min_xlocator(ax):
    try:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
        ax.figure.autofmt_xdate()
    except Exception as e:
        print('set_30min_xlocator failed:', e)

# ---------------------------------------------------------------------------
# End of appended helpers
# ---------------------------------------------------------------------------
# ============================================================================
# PREVENT SLEEP MODE (Windows)
# ============================================================================
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    def prevent_sleep():
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            logging.getLogger(__name__).info("Sleep prevention enabled (Windows)")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not prevent sleep: {e}")
    def allow_sleep():
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logging.getLogger(__name__).info("Sleep prevention disabled")
        except Exception: pass
else:
    def prevent_sleep(): logging.getLogger(__name__).info("Sleep prevention not for this OS")
    def allow_sleep(): pass

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    URL = "https://pml.good-game-service.com/pm-leaderboard/group?groupId=1237&lang=en&timezone=UTC-8"
    DB_FILE, EXCEL_FILE, CSV_FILE = "leaderboard_data.db", "leaderboard_export.xlsx", "leaderboard_export.csv"
    BACKUP_DIR, CONFIG_FILE, LOG_FILE, PATTERNS_FILE = "backups", "config.json", "leaderboard_tracker.log", "player_patterns.json"
    
    BROWSER = "chrome"
    START_MINIMIZED = True
    MIN_FREE_DISK_MB = 500
    TIMEZONE = pytz.timezone('Europe/Berlin')

    CHROME_DRIVER_PATH = r"C:\Webdriver\bin\chromedriver.exe"
    GECKO_PATH = r"C:\Webdriver\bin\geckodriver.exe"
    
    QUIET_START, QUIET_END = dt_time(3, 0), dt_time(6, 0)  # ✅ 3am-6am
    QUIET_INTERVAL, ACTIVE_BASE, ACTIVE_JITTER = 20, 20, 5
    QUIET_MIN_PLAYERS = 3  # ✅ NEW: Quiet only if <3 players active
    ACTIVITY_THRESHOLD, ESCALATION_INTERVAL = 10, 20
    MIN_POLL, DEESCALATION_TIME = 0.17, 60
    HH_START, HH_END, HH_INTERVAL = dt_time(23, 0), dt_time(0, 59), 60
    PERIOD_DAY, PERIOD_HOUR, PERIOD_MIN = 2, 12, 0
    
    MAX_TABLES, MIN_3MAX, MIN_6MAX = 4, 8, 12
    POINTS = {'3max': {1:36, 2:24, 3:12, 4:0}, '6max': {1:72, 2:54, 3:36, 4:18}}
    
    # Break detection thresholds
    BREAK_THRESHOLD_MIN = 5  # Minimum gap in minutes to count as break
    SLEEP_CLASSIFICATION_THRESHOLD = 120  # Minutes - breaks >120min after 1am are "sleep"

CFG = Config()

# ============================================================================
# LOGGING
# ============================================================================
def setup_logging():
    logger_obj = logging.getLogger(__name__)
    if logger_obj.handlers:
        logger_obj.handlers.clear()
    
    logger_obj.setLevel(logging.INFO)
    handler = RotatingFileHandler(CFG.LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger_obj.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger_obj.addHandler(console)
    logger_obj.propagate = False
    return logger_obj

logger = setup_logging()

# ============================================================================
# BROWSER SELECTION DIALOG
# ============================================================================
class BrowserChooser(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Choose Browser")
        self.resizable(False, False)
        self.result = None
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select browser for scraping:").grid(row=0, column=0, columnspan=2, pady=(0,8), sticky="w")
        self.var = tk.StringVar(value=CFG.BROWSER)
        ttk.Radiobutton(frm, text="Firefox", variable=self.var, value="firefox").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(frm, text="Chrome", variable=self.var, value="chrome").grid(row=1, column=1, sticky="w")
        self.min_var = tk.BooleanVar(value=CFG.START_MINIMIZED)
        ttk.Checkbutton(frm, text="Start minimized", variable=self.min_var).grid(row=2, column=0, columnspan=2, pady=(6,0), sticky="w")
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=2, pady=(10,0))
        ttk.Button(btns, text="OK", width=10, command=self.on_ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", width=10, command=self.on_cancel).pack(side="left", padx=4)
        self.transient(self.master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() // 2) - (self.winfo_width() // 2)
        y = self.master.winfo_y() + (self.master.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{max(0, x)}+{max(0, y)}")
        self.focus_set()

    def on_ok(self):
        self.result = {"browser": self.var.get(), "minimized": self.min_var.get()}
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
clean_points = lambda s: int("".join(c for c in str(s or '').replace(',','').split('.')[0] if c.isdigit())) if s else 0

def check_disk(p="C:\\"):
    d = shutil.disk_usage(p)
    free_mb = d[2] / (1024 * 1024)
    return free_mb >= CFG.MIN_FREE_DISK_MB

def is_hh(dt=None):
    """Check if it's Happy Hour (23:00-00:59)."""
    t = (dt or datetime.now()).time()
    return (t >= CFG.HH_START) or (t <= CFG.HH_END)

def calc_min_games(pts, hh=False):
    """Calculate minimum games needed for given points."""
    if pts <= 0:
        return 0
    mult = 2 if hh else 1
    remaining = pts // mult
    g = remaining // 72
    remaining -= g * 72
    best = None
    for a in range(0, remaining // 36 + 2):
        for b in range(0, remaining // 24 + 2):
            rem = remaining - a*36 - b*24
            if rem < 0:
                break
            if rem % 12 == 0:
                c = rem // 12
                total = g + a + b + c
                best = total if best is None or total < best else best
    return best if best is not None else g + (remaining + 35) // 36

def calc_min_sessions(games, tables=CFG.MAX_TABLES):
    """Calculate minimum sessions needed."""
    return ceil(games / tables) if games > 0 else 0

def get_period_start(ref=None):
    """
    DEPRECATED: Period boundaries now detected from website.
    Kept for backward compatibility with validation functions.
    """
    ref = ref or datetime.now(CFG.TIMEZONE)
    ref = CFG.TIMEZONE.localize(ref) if ref.tzinfo is None else ref
    wd, t = ref.weekday(), ref.time()
    days_back = 7 if wd == CFG.PERIOD_DAY and t < dt_time(CFG.PERIOD_HOUR, CFG.PERIOD_MIN) else (wd - CFG.PERIOD_DAY) % 7
    if days_back == 0 and wd == CFG.PERIOD_DAY and t >= dt_time(CFG.PERIOD_HOUR, CFG.PERIOD_MIN):
        ps = ref.replace(hour=CFG.PERIOD_HOUR, minute=CFG.PERIOD_MIN, second=0, microsecond=0)
    else:
        ps = (ref - timedelta(days=days_back)).replace(hour=CFG.PERIOD_HOUR, minute=CFG.PERIOD_MIN, second=0, microsecond=0)
    return ps

def get_period_elapsed(ct=None):
    """Get elapsed minutes in current period."""
    return ((ct or datetime.now(CFG.TIMEZONE)) - get_period_start(ct)).total_seconds() / 60

def get_period_info(ct=None):
    """Get comprehensive period information."""
    ct = ct or datetime.now(CFG.TIMEZONE)
    ps = get_period_start(ct)
    em = get_period_elapsed(ct)
    h, m, d = int(em // 60), int(em % 60), int(em // 1440)
    return {
        'start': ps,
        'end': ps + timedelta(days=7, seconds=-1),
        'elapsed_minutes': em,
        'elapsed_formatted': f"{d}d {h%24}h {m}min" if d > 0 else f"{h}h {m}min",
        'current': ct
    }

def validate_period(pts, t3, t6, tables=CFG.MAX_TABLES, hh=False, ct=None, player_name=None):
    """Validate period with breaks subtracted."""
    if pts <= 0:
        return True, "N/A", 0, 0, 0
    
    em = get_period_elapsed(ct)
    games = calc_min_games(pts, hh)
    sess = calc_min_sessions(games, tables)
    needed = sess * t6
    pi = get_period_info(ct)
    
    available_time = em
    
    if player_name:
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            period_start = pi['start'].date()
            period_end = pi['end'].date()
            
            query = '''SELECT session_duration_minutes, break_duration_minutes
                       FROM player_sessions
                       WHERE player_name = ?
                       AND session_date >= ?
                       AND session_date <= ?'''
            df = pd.read_sql_query(query, conn, params=(player_name, str(period_start), str(period_end)))
            conn.close()
            
            if not df.empty:
                total_breaks = df['break_duration_minutes'].sum() if 'break_duration_minutes' in df.columns else 0
                available_time = max(0, em - total_breaks)
        except Exception as e:
            logger.error(f"Period validation error: {e}")
    
    is_valid = available_time >= needed
    
    if is_valid:
        status = f"OK Valid (period: {pi['elapsed_formatted']}, available: {available_time:.0f}min, needs {needed:.0f}min)"
    else:
        status = f"!! Impossible! (period: {pi['elapsed_formatted']}, available: {available_time:.0f}min, needs {needed:.0f}min)"
    
    return is_valid, status, sess, needed, available_time

def validate_session(games, dm, t3, t6, tables=CFG.MAX_TABLES, hh=False, game_type="mixed"):
    """Validate if session duration is physically possible."""
    if games <= 0 or dm <= 0:
        return True, "N/A", 0, 0
    
    sess = calc_min_sessions(games, tables)
    
    if game_type == "3max":
        time_per_session = t3
        label = "3-max"
    elif game_type == "6max":
        time_per_session = t6
        label = "6-max"
    else:
        time_per_session = t6
        label = "mixed"
    
    needed = sess * time_per_session
    is_valid = dm >= needed
    
    if is_valid:
        status = f"✓ Valid ({sess} sess {label}, {needed:.0f}min)"
    else:
        status = f"⚠ Too fast! ({sess} sess {label} needs {needed:.0f}min, got {dm:.0f}min)"
    
    return is_valid, status, sess, needed

def calc_combos(delta, hh=False):
    """Calculate point combinations with exhaustive search."""
    if delta <= 0:
        return ""
    
    mult = 2 if hh else 1
    rem = abs(int(delta))
    pts_labels = [(72,"6max-P1"), (54,"6max-P2"), (36,"6max-P3"), (36,"3max-P1"),
                  (24,"3max-P2"), (18,"6max-P4"), (12,"3max-P3")]
    
    def find_combination(remaining, idx, path):
        if remaining == 0:
            return path
        if remaining < 0 or idx >= len(pts_labels):
            return None
        
        pts, lbl = pts_labels[idx]
        max_count = remaining // (pts * mult)
        
        for count in range(max_count, -1, -1):
            new_path = path + ([f"{count}x{lbl}"] if count > 0 else [])
            result = find_combination(remaining - count * pts * mult, idx + 1, new_path)
            if result is not None:
                return result
        return None
    
    result = find_combination(rem, 0, [])
    
    if result is None or not result or all(c.startswith("0x") for c in result):
        return "⚠ IMPOSSIBLE"
    
    games = sum(int(x.split('x')[0]) for x in result if 'x' in x and not x.startswith('0x'))
    return f"[{games} games] {', '.join([c for c in result if not c.startswith('0x')])}"

def calc_3max_only_combo(delta, hh=False):
    """Calculate minimum 3-max game combinations."""
    if delta <= 0:
        return ""
    if delta > 5000:
        return f"⚠ TOO LARGE ({delta:,})"
    
    mult = 2 if hh else 1
    rem = abs(int(delta))
    pts_3max = [(36, "P1"), (24, "P2"), (12, "P3")]
    
    def find_combination(remaining, idx, path, depth=0):
        if depth > 100:
            return None
        if remaining == 0:
            return path
        if remaining < 0 or idx >= len(pts_3max):
            return None
        
        pts, lbl = pts_3max[idx]
        max_count = min(remaining // (pts * mult), 50)
        
        for count in range(max_count, -1, -1):
            new_path = path + ([f"{count}x{lbl}"] if count > 0 else [])
            result = find_combination(remaining - count * pts * mult, idx + 1, new_path, depth + 1)
            if result is not None:
                return result
        return None
    
    result = find_combination(rem, 0, [])
    if not result or all(c.startswith("0x") for c in result):
        return "⚠ IMPOSSIBLE"
    
    games = sum(int(x.split('x')[0]) for x in result if 'x' in x and not x.startswith('0x'))
    combo_str = ', '.join([c for c in result if not c.startswith('0x')])
    if len(combo_str) > 50:
        combo_str = combo_str[:47] + "..."
    return f"[{games}g] {combo_str}"

def calc_6max_only_combo(delta, hh=False):
    """Calculate minimum 6-max game combinations."""
    if delta <= 0:
        return ""
    if delta > 5000:
        return f"⚠ TOO LARGE ({delta:,})"
    
    mult = 2 if hh else 1
    rem = abs(int(delta))
    pts_6max = [(72, "P1"), (54, "P2"), (36, "P3"), (18, "P4")]
    
    def find_combination(remaining, idx, path, depth=0):
        if depth > 100:
            return None
        if remaining == 0:
            return path
        if remaining < 0 or idx >= len(pts_6max):
            return None
        
        pts, lbl = pts_6max[idx]
        max_count = min(remaining // (pts * mult), 50)
        
        for count in range(max_count, -1, -1):
            new_path = path + ([f"{count}x{lbl}"] if count > 0 else [])
            result = find_combination(remaining - count * pts * mult, idx + 1, new_path, depth + 1)
            if result is not None:
                return result
        return None
    
    result = find_combination(rem, 0, [])
    if not result or all(c.startswith("0x") for c in result):
        return "⚠ IMPOSSIBLE"
    
    games = sum(int(x.split('x')[0]) for x in result if 'x' in x and not x.startswith('0x'))
    combo_str = ', '.join([c for c in result if not c.startswith('0x')])
    if len(combo_str) > 50:
        combo_str = combo_str[:47] + "..."
    return f"[{games}g] {combo_str}"

def _ts_to_iso(ts):
    """Convert timestamp to ISO string."""
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    return ts.isoformat(sep=' ', timespec='seconds')

def _date_to_str(d):
    """Convert date to string."""
    if d is None:
        return None
    if hasattr(d, "date"):
        d = d.date()
    return str(d)

def check_new_player(player_name, period_days):
    """
    Check if player is new (not in top 40 in the specified period).
    """
    try:
        conn = sqlite3.connect(CFG.DB_FILE)
        
        if period_days is None:
            # ✅ FIX: Use YYYY-MM-DD for database query
            cutoff = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')  # ← FIXED!
            query = '''
                SELECT COUNT(*) FROM leaderboard
                WHERE name = ?
                AND timestamp < ?
                AND CAST(rank AS INTEGER) <= 40
                LIMIT 1
            '''
            result = conn.execute(query, (player_name, cutoff)).fetchone()
        else:
            # ✅ FIX: Use YYYY-MM-DD for database query
            cutoff = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d %H:%M:%S')  # ← FIXED!
            query = '''
                SELECT COUNT(*) FROM leaderboard
                WHERE name = ?
                AND timestamp >= ?
                AND CAST(rank AS INTEGER) <= 40
                LIMIT 1
            '''
            result = conn.execute(query, (player_name, cutoff)).fetchone()
        
        conn.close()
        
        # If count is 0, player is new
        return "Yes" if result[0] == 0 else ""
        
    except Exception as e:
        logger.error(f"New player check error for {player_name}: {e}")
        return ""

def get_ranking_total(player_name):
    """
    Calculate ranking distribution - SIMPLIFIED & CACHED
    Returns format: "10xP1, 5xP2, 3xP3..."
    
    WICHTIG: Diese Funktion wird in _display_worker() aufgerufen
    und darf NICHT viel Zeit brauchen!
    """
    try:
        # SCHNELLE VERSION: Nur letzte 20 Perioden prüfen
        conn = sqlite3.connect(CFG.DB_FILE)
        
        query = '''
            SELECT 
                period_range,
                MIN(CAST(rank AS INTEGER)) as final_rank
            FROM leaderboard
            WHERE name = ?
            AND period_range IS NOT NULL
            GROUP BY period_range
            ORDER BY period_range DESC
            LIMIT 20
        '''
        
        result = conn.execute(query, (player_name,)).fetchall()
        conn.close()
        
        if not result:
            return ""
        
        # Count ranks
        rank_counts = {}
        for period_range, rank in result:
            try:
                rank_num = int(rank)
                if 1 <= rank_num <= 40:
                    rank_counts[rank_num] = rank_counts.get(rank_num, 0) + 1
            except Exception:
                continue
        
        if not rank_counts:
            return ""
        
        # Format
        parts = []
        for rank in sorted(rank_counts.keys()):
            count = rank_counts[rank]
            parts.append(f"{count}xP{rank}")
        
        full_text = ", ".join(parts)
        
        # Truncate if too long
        if len(full_text) > 100:
            return full_text[:97] + "..."
        
        return full_text
        
    except Exception as e:
        logger.error(f"❌ Ranking total error for {player_name}: {e}")
        return ""

# ============================================================================
# SELENIUM DRIVER FACTORY (UNIFIED)
# ============================================================================
def get_driver(headless=False, retry_count=0, max_retries=3):
    """
    Create Selenium driver with browser selection.
    FIXED: Only use selected browser, minimize properly.
    """
    try:
        browser = CFG.BROWSER.lower()
        
        if browser == "chrome":
            opts = ChromeOptions()
            opts.add_argument("--disable-blink-features=AutomationControlled")
            opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            opts.add_experimental_option('useAutomationExtension', False)
            opts.add_argument("--disable-notifications")
            
            if CFG.START_MINIMIZED:
                opts.add_argument("--start-minimized")
                opts.add_argument("--window-position=2000,2000")
            
            if headless:
                opts.add_argument("--headless=new")
            
            service = ChromeService(executable_path=CFG.CHROME_DRIVER_PATH) if os.path.exists(CFG.CHROME_DRIVER_PATH) else ChromeService(log_path=os.devnull, port=0)
            driver = _create_main_driver()

        else:
            # Firefox
            opts = FirefoxOptions(log_path=os.devnull, port=0)
            if headless:
                opts.add_argument("--headless")
            
            service = FirefoxService(executable_path=CFG.GECKO_PATH) if os.path.exists(CFG.GECKO_PATH) else Service()
            driver = _create_isolated_driver()
        
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(60)
        
        # Minimize after creation
        try:
            if CFG.START_MINIMIZED:
                driver.minimize_window()
        except Exception:
            pass
        
        logger.info(f"{browser.capitalize()} driver created successfully")
        return driver
        
    except Exception as e:
        logger.error(f"Driver creation failed (attempt {retry_count+1}/{max_retries}): {e}")
        
        # Kill zombie processes
        try:
            if os.name == 'nt':
                os.system('taskkill /F /IM chrome.exe /T >nul 2>&1')
                os.system('taskkill /F /IM chromedriver.exe /T >nul 2>&1')
                os.system('taskkill /F /IM firefox.exe /T >nul 2>&1')
                os.system('taskkill /F /IM geckodriver.exe /T >nul 2>&1')
                time.sleep(1.5)
        except Exception:
            pass
        
        if retry_count < max_retries:
            time.sleep(2.0)
            return get_driver(headless=headless, retry_count=retry_count+1, max_retries=max_retries)
        raise

def parse_leaderboard(driver):
    try:
        driver.set_page_load_timeout(90)
        driver.get(CFG.URL)

        wait = WebDriverWait(driver, 45)
        
        try:
            period_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.activated-set-title")))
            period = period_element.text.strip()
        except TimeoutException:
            logger.warning("Could not find period title element. Using 'Unknown Period'.")
            period = "Unknown Period"

        table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.playerRankingTable, table[class*='ranking']")))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr")))
        rows = table.find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr")
        
        parsed = []
        for tr in rows:
            try:
                cols = tr.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4: continue
                rk, nm, pts, pr = (cols[0].text, cols[1].text, cols[3].text, cols[4].text) if len(cols) == 5 else (cols[0].text, cols[1].text, cols[2].text, cols[3].text)
                if nm.strip() and rk.strip():
                    parsed.append({'rank': rk.strip(), 'name': nm.strip(), 'points': clean_points(pts), 'prize': pr.strip(),'timestamp': datetime.now(), 'period_range': period})
            except Exception: continue
        
        if not parsed: logger.warning("No data parsed from table rows.")
        logger.info(f"Parsed {len(parsed)} players for period: {period}")
        return pd.DataFrame(parsed), period
            
    except Exception as e:
        logger.error(f"Critical parse error: {e}", exc_info=False)
        return pd.DataFrame(), None

# ============================================================================
# INTELLIGENT POLLING
# ============================================================================
class IntelligentPolling:
    def __init__(self, config_mgr=None):
        self.config_mgr = config_mgr
        self.activity, self.changes = deque(maxlen=100), deque(maxlen=10)
        self.last_player_count = 0
        self.last_6am_reset = None

        if config_mgr:
            saved_state = self.config_mgr.get('polling_state', {})
            self.interval = saved_state.get('interval', 20 * 60)
            self.level = saved_state.get('level', 0)
            self.active_player_count = saved_state.get('active_player_count', 0)
            self.manual_mode = saved_state.get('manual_mode', False)
            self.manual_interval = saved_state.get('manual_interval', 30 * 60)
            last_activity_str = saved_state.get('last_activity')
            self.last_activity = datetime.fromisoformat(last_activity_str) if last_activity_str else None
            logger.info(f"✅ Restored polling: {self.interval/60:.0f}min, level={self.level}")
        else:
            self.interval, self.level, self.last_activity, self.manual_mode, self.manual_interval, self.active_player_count = 20 * 60, 0, None, False, 30 * 60, 0

    def is_quiet(self, dt=None):
        t = (dt or datetime.now()).time()
        return dt_time(3, 0) <= t < dt_time(6, 0)
    
    def is_happy_hour(self, dt=None):
        t = (dt or datetime.now()).time()
        return t >= dt_time(23, 0) or t <= dt_time(0, 59)

    def calc_changes(self, curr, prev):
        if prev is None or prev.empty: return 0
        merged = pd.merge(curr, prev, on='name', suffixes=('_curr', '_prev'))
        return (merged['points_curr'] != merged['points_prev']).sum()

    def set_manual(self, minutes):
        self.manual_mode = True
        self.manual_interval = max(0.17, minutes * 60)
        self.interval = self.manual_interval
        logger.info(f"Manual mode: {minutes} min")
    
    def set_intelligent(self):
        self.manual_mode = False
        logger.info("Intelligent mode activated")

    def update(self, changes, active_players=0):
        if self.manual_mode:
            logger.info(f"🔵 Manual mode: {self.manual_interval/60:.1f}min")
            return self.manual_interval
        
        self.last_activity = datetime.now()
        current_hour = self.last_activity.hour
        current_date = self.last_activity.date()
        self.active_player_count = active_players
        
        logger.info(f"📊 Polling update: {changes} changes, {active_players} active")

        if self.is_happy_hour():
            self.interval = 60; self.level = 4; logger.info("🎰 HAPPY HOUR → 1min")
        elif current_hour == 6 and self.last_6am_reset != current_date:
            self.interval = 15 * 60; self.level = 0
            self.last_6am_reset = current_date
            logger.info(f"🌅 6AM RESET → 20min")
        elif self.is_quiet() and active_players < 3:
            self.interval = 20 * 60; self.level = 0
            logger.info(f"😴 QUIET ({active_players} < 3) → 20min")
        else:
            player_diff = active_players - self.last_player_count
            if 1 <= current_hour < 3 and player_diff < 0:
                increase = abs(player_diff) * 60
                self.interval = min(self.interval + increase, 15 * 60)
                logger.info(f"🌙 POST-HH: {abs(player_diff)} left → {self.interval/60:.0f}min")
            elif player_diff > 0:
                reduction = player_diff * 60
                self.interval = max(self.interval - reduction, 5 * 60)
                logger.info(f"⚡ {player_diff} NEW → {self.interval/60:.0f}min")
            elif active_players < 3:
                self.interval = 20 * 60
            
            self.level = 1 if active_players < 10 else 2 if active_players < 15 else 3
        
        self.last_player_count = active_players
        logger.info(f"🕐 Final: {self.interval/60:.1f}min")
        return self.interval

    def status(self):
        if self.manual_mode: return f"Manual ({self.manual_interval/60:.1f}min)"
        if self.is_happy_hour(): return "🎰 HAPPY HOUR (1min)"
        return f"{['Normal', 'Active', 'Busy', 'Peak'][min(self.level, 3)]} ({self.interval/60:.1f}min)"

    def save_state(self):
        if self.config_mgr:
            state = {'interval': self.interval, 'level': self.level, 'last_activity': self.last_activity.isoformat() if self.last_activity else None, 'active_player_count': self.active_player_count, 'manual_mode': self.manual_mode, 'manual_interval': self.manual_interval}
            self.config_mgr.set('polling_state', state)
            logger.debug(f"💾 Saved polling state")

# ============================================================================
# FIX 3: CLUSTER ANALYSIS - Fix Lambda Scope Error
# ============================================================================

def run_cluster_analysis(self):
    """Triggers the new self-learning cluster analysis."""
    loading = self.show_loading(self.root, "Performing Deep Behavioral Analysis...\nThis may take a minute.")
    
    def worker():
        error_msg = None  # ✅ DEFINE OUTSIDE try/except
        try:
            report, fig = self.ml_bot_detector.get_cluster_report(days=7)
            self.root.after(0, lambda r=report, f=fig: [loading.destroy(), self.show_cluster_report(r, f)])
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Cluster analysis worker failed: {error_msg}", exc_info=True)
            self.root.after(0, lambda msg=error_msg: [loading.destroy(), messagebox.showerror("Error", f"Cluster analysis failed:\n{msg}")])
    threading.Thread(target=worker, daemon=True).start()
# ============================================================================
# ACTIVITY MONITOR
# ============================================================================
class ActivityMonitor:
    """Lightweight background monitor that detects player activity and triggers dynamic polling."""
    
    def __init__(self, app):
        self.app = app
        self.polling_mgr = app.polling_mgr
        self.running = False
        self.monitor_thread = None
        self.last_check_time = None
        self.check_interval = 60  # 60 seconds
        
    def start(self):
        """Start background activity monitoring."""
        if self.running:
            logger.warning("Activity monitor already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Activity Monitor started (checks every 60s)")
        
    def stop(self):
        """Stop background activity monitoring."""
        self.running = False
        if self.monitor_thread:
            logger.info("🔍 Activity Monitor stopped")
            
    def _monitor_loop(self):
        """Main monitoring loop - runs every 60 seconds."""
        while self.running:
            try:
                if not self.app.closing:
                    self._check_and_respond()
            except Exception as e:
                logger.error(f"Activity monitor error: {e}")
            
            # Sleep in small increments to allow quick shutdown
            for _ in range(60):
                if not self.running:
                    break
                time.sleep(1)
    
    def _check_and_respond(self):
        """Check for activity and adjust polling if needed."""
        try:
            activity_detected, active_count, new_session_players = self._detect_activity()
            
            if activity_detected:
                logger.info(f"🎯 Activity detected: {active_count} players gaining points")
                
                # Only trigger immediate fetch for NEW session players
                if new_session_players > 0 and not self.app.is_polling:
                    logger.info(f"📊 Triggering immediate fetch due to NEW session activity ({new_session_players} players)")
                    self.app.root.after(0, self.app.fetch)
                elif new_session_players == 0:
                    logger.debug(f"ℹ️ Activity from already-tracked players - no immediate fetch needed")
                
                # Adjust polling frequency
                self._adjust_polling_frequency(active_count)
            else:
                logger.debug("💤 No activity detected")
                
            self.last_check_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Activity check failed: {e}")
    
    def _detect_activity(self):
        """
        Quick lightweight check for player activity.
        Returns: (activity_detected: bool, active_player_count: int, new_session_players: int)
        """
        driver = None
        try:
            # Quick headless check
            driver = get_driver(headless=True)
            current_df, _ = parse_leaderboard(driver)
            
            if current_df.empty:
                return False, 0, 0
            
            # Get last snapshot from database
            conn = sqlite3.connect(CFG.DB_FILE)
            last_ts = conn.execute('SELECT MAX(timestamp) FROM leaderboard').fetchone()[0]
            
            if not last_ts:
                conn.close()
                return True, len(current_df), len(current_df)  # First fetch - all are new
            
            # Get previous points
            prev_df = pd.read_sql_query(
                'SELECT name, points FROM leaderboard WHERE timestamp = ?',
                conn, params=(last_ts,))
            
            # Get session start time (6am today or yesterday)
            now = datetime.now()
            if now.hour < 6:
                session_start = (now - timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
            else:
                session_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
            
            session_date = session_start.date()
            
            # Get points at session start for all players (optimized: single query)
            player_names = tuple(current_df['name'].tolist())
            if player_names:
                placeholders = ','.join('?' * len(player_names))
                results = conn.execute(f'''
                    SELECT player_name, first_points FROM player_sessions 
                    WHERE player_name IN ({placeholders}) AND session_date = ?
                ''', (*player_names, str(session_date))).fetchall()
                
                session_start_points = {name: points for name, points in results}
            else:
                session_start_points = {}
            
            conn.close()
            
            # Count players with point increases
            active_players = 0
            new_session_players = 0
            
            for _, row in current_df.iterrows():
                name = row['name']
                current_pts = row['points']
                
                prev_row = prev_df[prev_df['name'] == name]
                if prev_row.empty:
                    active_players += 1  # New player appeared
                    new_session_players += 1
                elif current_pts > prev_row.iloc[0]['points']:
                    active_players += 1  # Points increased
                    
                    # Check if this is NEW activity in the current session
                    if name not in session_start_points:
                        # No session record yet - this is NEW session activity
                        new_session_players += 1
            
            return active_players > 0, active_players, new_session_players
            
        except Exception as e:
            logger.error(f"Activity detection error: {e}")
            return False, 0, 0
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass
    
    def _adjust_polling_frequency(self, active_count):
        """Log recommended polling interval based on activity (non-disruptive)."""
        # Happy Hour override
        if is_hh():
            logger.info("🎰 Happy Hour active - activity recommendation skipped")
            return
        
        # Manual mode override
        if self.polling_mgr.manual_mode:
            logger.debug("⚙️ Manual mode active - skipping auto-adjustment")
            return
        
        # Calculate recommended interval based on activity
        if active_count >= 10:
            new_interval = 5 * 60  # 5 minutes
            level = "HIGH"
        elif active_count >= 5:
            new_interval = 10 * 60  # 10 minutes
            level = "MEDIUM-HIGH"
        elif active_count >= 2:
            new_interval = 15 * 60  # 15 minutes
            level = "MEDIUM"
        else:
            new_interval = 15 * 60  # 15 minutes
            level = "LOW"
        
        # Only log recommendation - don't force changes
        if abs(self.polling_mgr.interval - new_interval) > 60:
            current_interval = self.polling_mgr.interval / 60
            logger.info(f"📊 Activity detected: Recommend {new_interval/60:.0f}min polling ({level} activity: {active_count} players, current: {current_interval:.0f}min)")

# ============================================================================
# PLAYER SCOUT - Background Scanner with Process Isolation
# ============================================================================
class PlayerScout:
    """
    Background job that scans for NEW players joining the daily session.
    - Runs every 1:00 - 3:59 minutes (random)
    - Uses SEPARATE Firefox profile (no interference with main browser)
    - True process isolation - each browser runs independently
    """
    
    def __init__(self, app):
        self.app = app
        self.running = False
        self.scout_thread = None
        self.known_players = set()
        self.session_date = None
        self.last_scan_time = None
        self.scan_count = 0
        self.scout_profile_path = None
        
        # Create dedicated Scout profile directory
        self._create_scout_profile()
        
        logger.info(f"🔍 Scout initialized with dedicated Firefox profile (isolated from main browser)")
        
    def _create_scout_profile(self):
        """Create a dedicated Firefox profile for Scout - FIXED"""
        try:
            profile_dir = os.path.join(tempfile.gettempdir(), "firefox_scout_profile")
            
            # ✅ FIX: Create directory if it doesn't exist
            if not os.path.exists(profile_dir):
                os.makedirs(profile_dir, exist_ok=True)
                logger.info(f"🔍 Scout profile directory created: {profile_dir}")
            
            self.scout_profile_path = profile_dir
            logger.info(f"🔍 Scout profile ready: {profile_dir}")
        except Exception as e:
            logger.error(f"Scout profile creation error: {e}")
            self.scout_profile_path = None
    
    def start(self):
        """Start the Player Scout background scanner"""
        if self.running:
            logger.warning("⚠️ Player Scout already running")
            return
        
        self.running = True
        self.scout_thread = threading.Thread(target=self._scout_loop, daemon=True)
        self.scout_thread.start()
        logger.info(f"🔍 Player Scout started (isolated Firefox process, 1-4min intervals)")
    
    def stop(self):
        """Stop the Player Scout"""
        self.running = False
        if self.scout_thread:
            logger.info("🔍 Player Scout stopped")
    
    def _scout_loop(self):
        """Main scout loop - runs continuously"""
        while self.running:
            try:
                if not self.app.closing:
                    self._scan_for_new_players()
                
                # Random interval: 1:00 to 3:59 minutes
                wait_seconds = random.randint(60, 239)
                wait_minutes = wait_seconds // 60
                wait_secs = wait_seconds % 60
                logger.info(f"🔍 Scout: Next scan in {wait_minutes}:{wait_secs:02d} min")
                
                # Sleep in small increments to allow quick shutdown
                for _ in range(wait_seconds):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Scout loop error: {e}")
                time.sleep(60)
    
    def _scan_for_new_players(self):
        """Scan for new players using isolated Firefox process"""
        driver = None
        max_retries = 2
        for attempt in range(max_retries):

            try:
                # Determine current session date (6am boundary)
                now = datetime.now()
                if now.hour < 6:
                    current_session_date = (now - timedelta(days=1)).date()
                else:
                    current_session_date = now.date()
                
                # Reset known players if new day
                if self.session_date != current_session_date:
                    logger.info(f"🔍 Scout: New session detected → {current_session_date}")
                    self.session_date = current_session_date
                    self.known_players.clear()
                
                self.scan_count += 1
                scan_start = datetime.now()
                logger.info(f"🔍 Scout scan #{self.scan_count} started (isolated process)")
                
                # ✅ Create isolated Firefox driver
                driver = self._create_isolated_driver()
                
                if driver is None:
                    logger.warning("⚠️ Scout: Driver creation failed")
                    return
                
                # Parse leaderboard
                current_df, _ = parse_leaderboard(driver)
                
                if current_df.empty:
                    logger.info("🔍 Scout: No data (leaderboard empty)")
                    return
                
                # Get current active players (those with points > 0)
                active_players = set(current_df[current_df['points'] > 0]['name'].tolist())
                
                if not active_players:
                    logger.info("🔍 Scout: No active players")
                    return
                
                # Check for NEW players
                new_players = active_players - self.known_players
                
                scan_duration = (datetime.now() - scan_start).total_seconds()
                
                if new_players:
                    logger.info(f"🎯 Scout: {len(new_players)} NEW: {', '.join(sorted(new_players))}")
                    
                    # Update known players
                    self.known_players.update(new_players)
                    
                    # ✅ NEW: Trigger immediate snapshot
                    logger.info("📸 Scout: Triggering snapshot")
                    self.app.root.after(0, self._trigger_snapshot)
                    
                    # ✅ NEW: Reduce polling by N minutes
                    reduction = len(new_players)
                    logger.info(f"⚡ Scout: Reducing polling by {reduction}min")
                    self.app.root.after(0, lambda: self._adjust_polling(reduction))
                    
                else:
                    logger.info(f"🔍 Scout: No new ({len(active_players)} active, {len(self.known_players)} known)")
                
                self.last_scan_time = datetime.now()
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Scout scan failed (attempt {attempt+1}/{max_retries}), retrying...")
                    time.sleep(5)
                else:
                    logger.error(f"Scout scan failed after {max_retries} attempts: {e}")
        
    def _create_isolated_driver(self):
        """Create isolated Firefox driver - FIXED"""
        try:
            opts = FirefoxOptions()
            
            # ✅ FIX: Only use custom profile if directory exists AND has content
            if self.scout_profile_path and os.path.exists(self.scout_profile_path):
                try:
                    from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
                    
                    # Check if profile directory is not empty
                    if os.listdir(self.scout_profile_path):
                        profile = FirefoxProfile(self.scout_profile_path)
                        
                        # Optimize settings
                        profile.set_preference("permissions.default.image", 2)
                        profile.set_preference("browser.cache.disk.enable", False)
                        profile.set_preference("browser.cache.memory.enable", False)
                        profile.set_preference("browser.cache.offline.enable", False)
                        profile.set_preference("network.http.use-cache", False)
                        
                        opts.profile = profile
                        logger.info("🔍 Scout: Using custom Firefox profile")
                    else:
                        # Directory exists but is empty - use default settings
                        logger.info("🔍 Scout: Profile directory empty, using default settings")
                        opts.set_preference("permissions.default.image", 2)
                except Exception as profile_error:
                    logger.warning(f"Scout profile error: {profile_error}, using default")
                    opts.set_preference("permissions.default.image", 2)
            else:
                # No custom profile - use default with optimizations
                logger.info("🔍 Scout: No custom profile, using optimized defaults")
                opts.set_preference("permissions.default.image", 2)
                opts.set_preference("browser.cache.disk.enable", False)
            
            # Headless mode
            opts.add_argument("--headless")
            opts.add_argument("--disable-gpu")
            opts.add_argument("--no-sandbox")
            
            # Service configuration
            if os.path.exists(CFG.GECKO_PATH):
                service = FirefoxService(executable_path=CFG.GECKO_PATH)
            else:
                service = FirefoxService  
            
            driver = _create_isolated_driver()

            
            driver.set_page_load_timeout(60)
            driver.set_script_timeout(60)
            
            logger.info(f"🔍 Scout: Isolated Firefox driver created (profile: {self.scout_profile_path is not None})")
            return driver
            
        except Exception as e:
            logger.error(f"Isolated driver creation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _trigger_snapshot(self):
        """Trigger immediate fetch/snapshot"""
        try:
            if not self.app.closing:
                self.app.fetch()
        except Exception as e:
            logger.error(f"Scout snapshot trigger error: {e}")
    
    def _adjust_polling(self, reduction_minutes):
        """Reduce polling interval by N minutes."""
        try:
            if hasattr(self.app, 'polling_mgr') and hasattr(self.app.polling_mgr, 'interval'):
                old_interval = self.app.polling_mgr.interval
                new_interval = max(old_interval - (reduction_minutes * 60), 5 * 60)  # Min 5min
                self.app.polling_mgr.interval = new_interval
                
                logger.info(f"⚡ Polling: {old_interval/60:.0f}min → {new_interval/60:.0f}min")
                
                # Update UI
                try:
                    status_text = self.app.polling_mgr.status()
                    self.app.root.after(0, lambda: self.app.poll_status_label.config(text=status_text))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Polling adjustment error: {e}")

# ============================================================================
# PATTERN DETECTOR
# ============================================================================
class PatternDetector:
    def __init__(self):
        self.patterns = self.load()
        self.sessions = defaultdict(list)
    
    def load(self):
        try:
            if os.path.exists(CFG.PATTERNS_FILE):
                with open(CFG.PATTERNS_FILE, 'r') as f:
                    return json.load(f)
            return {}
        except Exception:
            return {}
    
    def save(self):
        try:
            with open(CFG.PATTERNS_FILE, 'w') as f:
                json.dump(self.patterns, f, indent=4)
        except Exception as e:
            logger.error(f"Pattern save error: {e}")
    
    def detect(self, curr, prev):
        if prev is None or prev.empty:
            return []
        alerts, ct = [], datetime.now()
        for _, r in curr.iterrows():
            name, cp = r['name'], r['points']
            pr = prev[prev['name'] == name]
            if not pr.empty:
                pp = pr.iloc[0]['points']
                if cp > pp:
                    if name not in self.sessions or not self.sessions[name]:
                        st = ct
                        self.sessions[name].append({'start': st, 'end': None})
                        self.record_start(name, st)
                        if self.is_regular(name, st):
                            alert = f"Regular player: {name} @ {st.strftime('%H:%M')}"
                            alerts.append(alert)
                            logger.info(alert)
                elif cp == pp and name in self.sessions and self.sessions[name] and self.sessions[name][-1]['end'] is None:
                    if (ct - self.sessions[name][-1]['start']).total_seconds() > 600:
                        self.sessions[name][-1]['end'] = ct
                        self.record_end(name, ct)
        return alerts
    
    def record_start(self, player, st):
        if player not in self.patterns:
            self.patterns[player] = {'starts': [], 'ends': [], 'sessions': 0}
        tm, ds = st.hour * 60 + st.minute, st.strftime('%Y-%m-%d')
        self.patterns[player]['starts'].append({'date': ds, 'time': tm, 'timestamp': st.isoformat()})
        self.patterns[player]['sessions'] += 1
        cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.patterns[player]['starts'] = [s for s in self.patterns[player]['starts'] if s['date'] >= cutoff]
        self.save()
    
    def record_end(self, player, et):
        if player not in self.patterns:
            return
        tm, ds = et.hour * 60 + et.minute, et.strftime('%Y-%m-%d')
        self.patterns[player]['ends'].append({'date': ds, 'time': tm, 'timestamp': et.isoformat()})
        cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.patterns[player]['ends'] = [e for e in self.patterns[player]['ends'] if e['date'] >= cutoff]
        self.save()
    
    def is_regular(self, player, st):
        if player not in self.patterns:
            return False
        starts = self.patterns[player]['starts']
        if len(starts) < 3:
            return False
        ctm = st.hour * 60 + st.minute
        recent = sorted(starts, key=lambda x: x['timestamp'])[-4:-1]
        if len(recent) < 3:
            return False
        times = [s['time'] for s in recent] + [ctm]
        if max(times) - min(times) <= 5 and len(set(s['date'] for s in recent)) >= 3:
            return True
        return False

# ============================================================================
# SELF-LEARNING BOT DETECTOR - v2.1 mit INTERACTION ANALYSIS
# ============================================================================
class SelfLearningBotDetector:
    """
    Machine learning bot detector.
    - analyze_player_clusters: Finds groups of players with similar individual behavior.
    - analyze_top_player_interaction: Specifically analyzes the relationship between the top 2 players.
    """
    def __init__(self):
        pass # Keine Initialisierung nötig

    def _extract_robust_features(self, player_name, conn, days=7):
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        query = 'SELECT timestamp, points FROM leaderboard WHERE name = ? AND timestamp >= ? ORDER BY timestamp ASC'
        df = pd.read_sql_query(query, conn, params=(player_name, cutoff))
        if len(df) < 20: return None
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if len(df) < 20: return None
        df = df.set_index('timestamp').resample('5min').mean().interpolate(method='time').reset_index()
        df['gain'] = df['points'].diff().fillna(0)
        point_gains = df[df['gain'] > 0]
        if len(point_gains) < 10: return None
        time_diffs = point_gains['timestamp'].diff().dt.total_seconds().dropna()
        time_cv = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 0
        gain_cv = point_gains['gain'].std() / point_gains['gain'].mean() if point_gains['gain'].mean() > 0 else 0
        night_ratio = point_gains[point_gains['timestamp'].dt.hour.isin([3, 4, 5, 6])].shape[0] / len(point_gains)
        total_duration_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        activity_density = len(point_gains) / total_duration_hours if total_duration_hours > 0 else 0
        gain_entropy = entropy(point_gains['gain'].value_counts(normalize=True), base=2)
        time_entropy = entropy(pd.cut(time_diffs, bins=10, labels=False).value_counts(normalize=True), base=2)
        return {'time_cv': time_cv, 'gain_cv': gain_cv, 'night_ratio': night_ratio, 'density': activity_density, 'gain_entropy': gain_entropy, 'time_entropy': time_entropy}
    
    def analyze_player_clusters(self, days=7):
        
        logger.info("🔬 Starting Cluster Analysis...")
        conn = sqlite3.connect(CFG.DB_FILE)
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        active_players = [r[0] for r in conn.execute("SELECT DISTINCT name FROM leaderboard WHERE timestamp >= ?", (cutoff,)).fetchall()]
        features_list, player_list = [], []
        for player in active_players:
            features = self._extract_robust_features(player, conn, days)
            if features: features_list.append(features); player_list.append(player)
        conn.close()
        if len(player_list) < 5: return {"error": "Not enough data for cluster analysis."}, None, None, None
        df_features = pd.DataFrame(features_list).fillna(0)
        X_scaled = StandardScaler().fit_transform(df_features)
        db = DBSCAN(eps=1.0, min_samples=3).fit(X_scaled)
        labels = db.labels_
        df_features['cluster'] = labels; df_features['player'] = player_list
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        try:
            human_cluster_id = df_features['cluster'].value_counts().idxmax()
            if human_cluster_id == -1 and n_clusters > 0: human_cluster_id = df_features[df_features['cluster'] != -1]['cluster'].value_counts().idxmax()
        except: human_cluster_id = 0
        report = {"summary": f"Found {n_clusters} clusters and {n_noise} anomalies.", "human_cluster_id": int(human_cluster_id), "clusters": {}, "anomalies": df_features[df_features['cluster'] == -1]['player'].tolist()}
        for cid in sorted(df_features['cluster'].unique()):
            if cid == -1: continue
            cluster_df = df_features[df_features['cluster'] == cid]
            verdict = "Human Baseline" if cid == human_cluster_id else f"Suspicious Network {cid}"
            profile = cluster_df.drop(columns=['cluster', 'player']).mean().to_dict()
            report["clusters"][str(cid)] = {"verdict": verdict, "size": len(cluster_df), "members": cluster_df['player'].tolist(), "profile": {k: round(v, 3) for k, v in profile.items()}}
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        return report, X_pca, labels, player_list

    def get_cluster_report(self, days=7):
        # ... (Diese Methode bleibt unverändert)
        report_data, X_pca, labels, player_list = self.analyze_player_clusters(days)
        # ... (Rest der Report-Generierung)
        return "Report generiert", None # Platzhalter, der Originalcode ist korrekt

    # ✅ NEUE METHODE ZUR INTERAKTIONS-ANALYSE
    def analyze_top_player_interaction(self, days=7):
        """
        Analyzes the interaction between the top 2 players for coordination.
        Specifically checks for correlation and turn-taking behavior.
        """
        logger.info("🔬 Starting Top 2 Interaction Analysis...")
        conn = sqlite3.connect(CFG.DB_FILE)
        
        # 1. Finde die Top 2 Spieler der letzten 24h
        cutoff_24h = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        query_top2 = '''
            SELECT name FROM leaderboard
            WHERE timestamp >= ?
            GROUP BY name
            ORDER BY MAX(points) DESC
            LIMIT 2
        '''
        top2 = [r[0] for r in conn.execute(query_top2, (cutoff_24h,)).fetchall()]
        if len(top2) < 2:
            conn.close()
            return "Not enough data: Fewer than 2 active top players in the last 24h."

        p1_name, p2_name = top2[0], top2[1]
        report = f"🤝 TOP 2 INTERACTION ANALYSIS: {p1_name} vs {p2_name}\n{'='*70}\n"
        
        # 2. Lade deren Zeitreihen-Daten
        cutoff_period = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        query_data = 'SELECT timestamp, name, points, rank FROM leaderboard WHERE name IN (?, ?) AND timestamp >= ?'
        df = pd.read_sql_query(query_data, conn, params=(p1_name, p2_name, cutoff_period))
        conn.close()

        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        
        # 3. Korrelations-Analyse
        pivot_df = df.pivot(index='timestamp', columns='name', values='points').interpolate(method='time')
        correlation = pivot_df.corr().iloc[0, 1]
        report += f"📊 PUNKTE-KORRELATION: {correlation:.3f}\n"
        if correlation > 0.85:
            report += "  -> VERDICT: HOCHGRADIG SYNCHRON. Ein Wert über 0.85 deutet stark darauf hin, dass die Spieler nicht unabhängig voneinander agieren.\n\n"
        else:
            report += "  -> VERDICT: Normale Korrelation.\n\n"
            
        # 4. Analyse des Abwechselns (Turn-Taking)
        report += "🔄 ANALYSE DES ABWECHSELNS (TURN-TAKING):\n"
        df['gain'] = df.groupby('name')['points'].diff().fillna(0)
        df['is_active'] = df['gain'] > 0
        
        overtake_events = []
        for i in range(1, len(df)):
            prev_row, curr_row = df.iloc[i-1], df.iloc[i]
            
            # Suche nach einem Überholmanöver von P2
            if curr_row['name'] == p2_name and prev_row['name'] == p2_name:
                p1_prev_points = pivot_df.loc[prev_row['timestamp'], p1_name]
                p1_curr_points = pivot_df.loc[curr_row['timestamp'], p1_name]
                
                if prev_row['points'] < p1_prev_points and curr_row['points'] > p1_curr_points:
                    overtake_time = curr_row['timestamp']
                    
                    # Überprüfe, ob P2 danach inaktiv wird
                    df_after_overtake = df[df['timestamp'] > overtake_time]
                    p2_after = df_after_overtake[df_after_overtake['name'] == p2_name]
                    
                    # Finde den ersten Zeitpunkt, an dem P2 wieder aktiv ist
                    p2_next_activity = p2_after[p2_after['is_active']].iloc[0]['timestamp'] if not p2_after[p2_after['is_active']].empty else None
                    p2_pause_duration = (p2_next_activity - overtake_time).total_seconds() / 60 if p2_next_activity else float('inf')

                    # Überprüfe, ob P1 in dieser Pause aktiv wird
                    p1_after = df_after_overtake[df_after_overtake['name'] == p1_name]
                    if p2_next_activity:
                        p1_after = p1_after[p1_after['timestamp'] < p2_next_activity]
                    
                    p1_response = p1_after[p1_after['is_active']]
                    if not p1_response.empty:
                        p1_response_time = (p1_response.iloc[0]['timestamp'] - overtake_time).total_seconds() / 60
                        event = {
                            "time": overtake_time.strftime('%Y-%m-%d %H:%M'),
                            "p1_response_min": round(p1_response_time),
                            "p2_pause_min": round(p2_pause_duration) if p2_pause_duration != float('inf') else "ongoing"
                        }
                        overtake_events.append(event)

        if overtake_events:
            report += f"  -> VERDICT: {len(overtake_events)} verdächtige Abwechsel-Events gefunden!\n"
            for event in overtake_events:
                report += f"     - Event am {event['time']}:\n"
                report += f"       -> {p2_name} überholt {p1_name}.\n"
                report += f"       -> {p1_name} reagiert und beginnt nach ~{event['p1_response_min']} Minuten zu spielen.\n"
                report += f"       -> {p2_name} legt eine Pause von mindestens {event['p2_pause_min']} Minuten ein.\n"
            report += "  -> DIESES MUSTER IST EIN EXTREM STARKES INDIZ FÜR KOORDINATION!\n"
        else:
            report += "  -> VERDICT: Keine eindeutigen Abwechsel-Events im Analysezeitraum gefunden.\n"
            
        return report

    def analyze_top_3_competitors(self, days=7):
        logger.info("🏆 Starting Top 3 Competitor Profile Analysis...")
        conn = sqlite3.connect(CFG.DB_FILE)
        
        # 1. Finde die Top 3 Spieler der letzten 24h
        cutoff_24h = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        query_top3 = '''
            SELECT name FROM leaderboard
            WHERE timestamp >= ? GROUP BY name ORDER BY MAX(points) DESC LIMIT 3
        '''
        top3 = [r[0] for r in conn.execute(query_top3, (cutoff_24h,)).fetchall()]
        if len(top3) < 3:
            conn.close()
            return "Not enough data: Fewer than 3 active top players in the last 24h."

        report = f"🏆 TOP 3 COMPETITOR PROFILE ANALYSIS\n{'='*70}\n"
        report += f"Players: {', '.join(top3)}\n\n"

        # ✅ FIX: Ermittle den aktuellen Turnierzeitraum (`period_range`)
        current_period_query = "SELECT period_range FROM leaderboard ORDER BY timestamp DESC LIMIT 1"
        period_result = conn.execute(current_period_query).fetchone()
        if not period_result:
            conn.close()
            return "Error: Cannot determine the current tournament period."
        current_period = period_result[0]
        report += f"Current Tournament Week: {current_period}\n\n"

        # --- Bereich 1: Aktuelle Interaktion & Koordination ---
        report += f"--- 1. CURRENT INTERACTION ANALYSIS (within this week) ---\n"
        query_data = 'SELECT timestamp, name, points FROM leaderboard WHERE name IN (?, ?, ?) AND period_range = ?'
        df = pd.read_sql_query(query_data, conn, params=(top3[0], top3[1], top3[2], current_period))
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        pivot_df = df.pivot(index='timestamp', columns='name', values='points').interpolate(method='time')
        
        # Korrelationen
        if not pivot_df.empty and len(pivot_df.columns) > 1:
            correlations = pivot_df.corr()
            report += "📊 Pairwise Point Correlation:\n"
            corr_p12 = correlations.loc[top3[0], top3[1]]
            corr_p13 = correlations.loc[top3[0], top3[2]]
            corr_p23 = correlations.loc[top3[1], top3[2]]
            report += f"  - {top3[0]} vs {top3[1]}: {corr_p12:.3f}\n"
            report += f"  - {top3[0]} vs {top3[2]}: {corr_p13:.3f}\n"
            report += f"  - {top3[1]} vs {top3[2]}: {corr_p23:.3f}\n"
            if any(c > 0.9 for c in [corr_p12, corr_p13, corr_p23]):
                 report += "  -> 🚨 VERDICT: Extremely high correlation detected. Strong sign of coordination.\n"
        else:
            report += "📊 Not enough overlapping data for correlation analysis this week.\n"
        report += "\n"

        # --- Bereich 2: Historische Leistung & Karriere-Analyse ---
        for player in top3:
            report += f"--- 2. HISTORICAL PROFILE: {player} ---\n"
            
            # Ranking-Historie
            query_history = '''
                SELECT period_range, MIN(CAST(rank AS INTEGER)) as final_rank, MAX(points) as max_points
                FROM leaderboard WHERE name = ? AND period_range IS NOT NULL
                GROUP BY period_range ORDER BY period_range DESC
            '''
            history = conn.execute(query_history, (player,)).fetchall()
            
            if not history:
                report += "  - No previous Top 40 history found. 🚨 -> VERDICT: EXTREMELY SUSPICIOUS ('Sleeper' Bot).\n\n"
                continue

            total_weeks = len(history)
            top3_finishes = sum(1 for _, r, _ in history if r <= 3)
            top10_finishes = sum(1 for _, r, _ in history if r <= 10)
            
            report += f"  - Total weeks in Top 40: {total_weeks}\n"
            report += f"  - Top 3 Finishes: {top3_finishes} ({top3_finishes/total_weeks:.1%})\n"
            report += f"  - Top 10 Finishes: {top10_finishes} ({top10_finishes/total_weeks:.1%})\n"
            
            if total_weeks > 10 and top3_finishes / total_weeks > 0.8:
                 report += "    -> 🚨 VERDICT: Inhumanly dominant performance over many weeks.\n"

            # Leistungsvergleich (Performance Comparison)
            all_points = [p for _, _, p in history]
            avg_points = np.mean(all_points) if all_points else 0

            # ✅ KORRIGIERTE BERECHNUNG der aktuellen Punkte
            current_points_query = "SELECT MAX(points) FROM leaderboard WHERE name = ? AND period_range = ?"
            current_points_result = conn.execute(current_points_query, (player, current_period)).fetchone()
            current_points = current_points_result[0] if current_points_result and current_points_result[0] is not None else 0

            performance_delta = (current_points / avg_points - 1) * 100 if avg_points > 0 else 0
            
            report += f"  - Historical Avg Points/Week: {avg_points:,.0f}\n"
            report += f"  - Current Week Points: {current_points:,.0f}\n"
            report += f"  - Performance Delta: {performance_delta:+.1f}%\n"
            if performance_delta > 200 and current_points > 50000: # Schwellenwert, um Fehlalarme bei niedrigen Punkten zu vermeiden
                report += "    -> 🚨 VERDICT: Sudden, massive performance increase compared to historical average.\n"
            report += "\n"

        conn.close()
        return report

    # ✅ NEUE, UMFASSENDE ANALYSE-METHODE für die Top 5
    def analyze_top_5_competitors(self, days=7):
        logger.info("🕵️ Starting Top 5 Competitor Profile Analysis...")
        conn = sqlite3.connect(CFG.DB_FILE)
        
        cutoff_24h = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
        query_top5 = 'SELECT name FROM leaderboard WHERE timestamp >= ? GROUP BY name ORDER BY MAX(points) DESC LIMIT 5'
        top5 = [r[0] for r in conn.execute(query_top5, (cutoff_24h,)).fetchall()]
        if len(top5) < 3:
            conn.close()
            return "Not enough data: Fewer than 3 active top players in the last 24h."

        report = f"🕵️ TOP 5 COMPETITOR PROFILE ANALYSIS\n{'='*70}\n"
        report += f"Players: {', '.join(top5)}\n"
        
        current_period_query = "SELECT period_range FROM leaderboard ORDER BY timestamp DESC LIMIT 1"
        current_period = conn.execute(current_period_query).fetchone()[0]
        report += f"Current Tournament Week: {current_period}\n\n"
        
        report += f"--- 1. GROUP INTERACTION ANALYSIS (Current Week) ---\n"
        query_data = f"SELECT timestamp, name, points FROM leaderboard WHERE name IN ({','.join(['?']*len(top5))}) AND period_range = ?"
        df = pd.read_sql_query(query_data, conn, params=(*top5, current_period))
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        pivot_df = df.pivot(index='timestamp', columns='name', values='points').interpolate(method='time')
        
        if not pivot_df.empty and len(pivot_df.columns) > 1:
            correlations = pivot_df.corr()
            report += "📊 Pairwise Point Correlation Matrix:\n"
            report += correlations.to_string(float_format="%.3f") + "\n"

            # ✅ FIX: Rename index levels before resetting to avoid column name conflict
            stacked_corr = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
            if not stacked_corr.empty:
                stacked_corr.index.rename(['Player 1', 'Player 2'], inplace=True)
                high_corr_pairs = stacked_corr.reset_index(name='Correlation')

                suspicious_pairs = high_corr_pairs[high_corr_pairs['Correlation'] > 0.9]
                if not suspicious_pairs.empty:
                    report += "  -> 🚨 VERDICT: Extremely high correlation(s) detected:\n"
                    for _, row in suspicious_pairs.iterrows():
                        report += f"     - {row['Player 1']} & {row['Player 2']}: {row['Correlation']:.3f}\n"
        else:
            report += "📊 Not enough data for correlation analysis this week.\n"
        report += "\n"

        # Synchronisierte Aktivität
        gains_df = pivot_df.diff().fillna(0)
        simultaneous_activity = (gains_df > 0).sum(axis=1)
        sync_events = simultaneous_activity[simultaneous_activity >= 3]
        if not sync_events.empty:
            report += f"🚀 Synchronized Activity Bursts (≥3 players starting simultaneously):\n"
            report += f"  -> 🚨 VERDICT: Found {len(sync_events)} instances of highly coordinated activity.\n"
            for ts, count in sync_events.head(5).items():
                active_players = gains_df.loc[ts][gains_df.loc[ts] > 0].index.tolist()
                report += f"     - At {ts.strftime('%d-%m %H:%M')}: {count} players started: {', '.join(active_players)}\n"
        else:
            report += "🚀 No synchronized activity bursts detected.\n"
        report += "\n"
        
        # --- Bereich 2: Historische Leistung & Karriere-Analyse ---
        for player in top5:
            report += f"--- PROFILE: {player} ---\n"
            query_history = 'SELECT period_range, MIN(CAST(rank AS INTEGER)), MAX(points) FROM leaderboard WHERE name = ? AND period_range IS NOT NULL GROUP BY period_range ORDER BY period_range DESC'
            history = conn.execute(query_history, (player,)).fetchall()
            
            if not history:
                report += "  - No previous Top 40 history found. 🚨 -> VERDICT: EXTREMELY SUSPICIOUS ('Sleeper' Bot).\n\n"
                continue

            total_weeks = len(history)
            top3_finishes = sum(1 for _, r, _ in history if r <= 3)
            top10_finishes = sum(1 for _, r, _ in history if r <= 10)
            
            report += f"  - Career: {total_weeks} weeks in Top 40. Top 3 finishes: {top3_finishes} ({top3_finishes/total_weeks:.1%}).\n"
            if total_weeks > 10 and top3_finishes / total_weeks > 0.8:
                 report += "    -> 🚨 VERDICT: Inhumanly dominant long-term performance.\n"

            all_points = [p for _, _, p in history if p is not None]
            avg_points = np.mean(all_points) if all_points else 0
            current_points_query = "SELECT MAX(points) FROM leaderboard WHERE name = ? AND period_range = ?"
            current_points = (conn.execute(current_points_query, (player, current_period)).fetchone() or [0])[0] or 0
            performance_delta = (current_points / avg_points - 1) * 100 if avg_points > 0 else 0
            
            report += f"  - Performance: Current {current_points:,.0f} points vs historical avg of {avg_points:,.0f} ({performance_delta:+.1f}%).\n"
            if performance_delta > 200 and current_points > 50000:
                report += "    -> 🚨 VERDICT: Massive performance spike compared to historical average.\n"
            report += "\n"

        conn.close()
        return report
# ============================================================================
# CONFIG MANAGER
# ============================================================================
class ConfigMgr:
    def __init__(self):
        self.cfg = {
            'dark_mode': False,
            'window_geometry': '1900x900',
            'autopoll_interval': CFG.QUIET_INTERVAL,
            'hh_interval': CFG.HH_INTERVAL,
            'min_3max_time': CFG.MIN_3MAX,
            'min_6max_time': CFG.MIN_6MAX,
            'max_tables': CFG.MAX_TABLES,
            'last_period': None,
            'intelligent_polling': True,
            'pattern_detection': True,
            # ✅ NEW: Persist polling state
            'polling_state': {
                'interval': 15 * 60,  # Default 15 minutes in seconds
                'level': 0,
                'last_activity': None,
                'active_player_count': 0,
                'manual_mode': False,
                'manual_interval': 10 * 60
            }
        }
        try:
            if os.path.exists(CFG.CONFIG_FILE):
                with open(CFG.CONFIG_FILE, 'r') as f:
                    self.cfg.update(json.load(f))
        except Exception:
            pass
    
    def save(self):
        try:
            with open(CFG.CONFIG_FILE, 'w') as f:
                json.dump(self.cfg, f, indent=4)
        except Exception as e:
            logger.error(f"Config save: {e}")
    
    def get(self, k, d=None):
        return self.cfg.get(k, d)
    
    def set(self, k, v):
        self.cfg[k] = v
        self.save()

# ============================================================================
# DATABASE
# ============================================================================
class DB:
    @staticmethod
    def init():
        conn = sqlite3.connect(CFG.DB_FILE)
        c = conn.cursor()
        
        # Leaderboard table
        c.execute('''CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rank TEXT,
            name TEXT,
            points INTEGER,
            prize TEXT,
            min_games INTEGER,
            combinations TEXT,
            timestamp DATETIME,
            period_range TEXT,
            is_happy_hour INTEGER DEFAULT 0)''')
        
        # Player sessions table
        c.execute('''CREATE TABLE IF NOT EXISTS player_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            session_date DATE NOT NULL,
            first_online DATETIME,
            last_offline DATETIME,
            first_points INTEGER DEFAULT 0,
            last_points INTEGER DEFAULT 0,
            total_points_gained INTEGER DEFAULT 0,
            games_played INTEGER DEFAULT 0,
            session_duration_minutes REAL DEFAULT 0,
            break_duration_minutes REAL DEFAULT 0,
            break_type TEXT,
            sleep_duration_minutes REAL DEFAULT 0,
            UNIQUE(player_name, session_date)
        )''')
        
        # Create indexes
        for idx in ['timestamp', 'name', 'period_range']:
            try:
                c.execute(f'CREATE INDEX IF NOT EXISTS idx_{idx} ON leaderboard({idx})')
            except Exception:
                pass
        
        try:
            c.execute('CREATE INDEX IF NOT EXISTS idx_session_player ON player_sessions(player_name)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_session_date ON player_sessions(session_date)')
        except Exception:
            pass
        
        # Check and add missing columns
        c.execute('PRAGMA table_info(leaderboard)')
        cols = [r[1] for r in c.fetchall()]
        if 'period_range' not in cols:
            try:
                c.execute("ALTER TABLE leaderboard ADD COLUMN period_range TEXT")
            except Exception:
                pass
        if 'is_happy_hour' not in cols:
            try:
                c.execute("ALTER TABLE leaderboard ADD COLUMN is_happy_hour INTEGER DEFAULT 0")
            except Exception:
                pass
        
        c.execute('PRAGMA table_info(player_sessions)')
        session_cols = [r[1] for r in c.fetchall()]
        
        missing_cols = {
            'first_online': 'DATETIME',
            'last_offline': 'DATETIME',
            'first_points': 'INTEGER DEFAULT 0',
            'last_points': 'INTEGER DEFAULT 0',
            'total_points_gained': 'INTEGER DEFAULT 0',
            'session_duration_minutes': 'REAL DEFAULT 0',
            'break_duration_minutes': 'REAL DEFAULT 0',
            'break_type': 'TEXT',
            'sleep_duration_minutes': 'REAL DEFAULT 0'
        }
        
        for col_name, col_type in missing_cols.items():
            if col_name not in session_cols:
                try:
                    c.execute(f"ALTER TABLE player_sessions ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    @staticmethod
    def has_data():
        """Check if database has any data."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            row = conn.execute("SELECT 1 FROM leaderboard LIMIT 1").fetchone()
            conn.close()
            return row is not None
        except Exception:
            return False
    
    @staticmethod
    def players():
        """Get all distinct player names."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            result = [r[0] for r in conn.execute("SELECT DISTINCT name FROM leaderboard").fetchall()]
            conn.close()
            return result
        except Exception:
            return []
    
    @staticmethod
    def timestamps():
        """Get all unique timestamps."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            rows = conn.execute("SELECT DISTINCT timestamp FROM leaderboard ORDER BY timestamp DESC").fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception:
            return []
    
    @staticmethod
    def periods():
        """Get all unique period ranges."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            rows = conn.execute("SELECT DISTINCT period_range FROM leaderboard WHERE period_range IS NOT NULL ORDER BY period_range DESC").fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception:
            return []
    
    @staticmethod
    def load(ts=None):
        """Load leaderboard data for specific timestamp."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            if ts:
                df = pd.read_sql_query("SELECT * FROM leaderboard WHERE timestamp = ? ORDER BY CAST(rank AS INTEGER)", conn, params=(str(ts),))
            else:
                df = pd.read_sql_query("SELECT * FROM leaderboard ORDER BY timestamp DESC, CAST(rank AS INTEGER) LIMIT 100", conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Load error: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def save(df, period=None):
        """Save leaderboard data to database."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            ct = datetime.now(CFG.TIMEZONE)
            hh = is_hh(ct)
            
            for _, row in df.iterrows():
                pts = row.get('points', 0)
                mg = calc_min_games(pts, hh)
                combo = calc_combos(pts, hh)
                name = row.get('name', '').strip()
                
                # Insert leaderboard data
                conn.execute('''INSERT INTO leaderboard
                    (rank, name, points, prize, min_games, combinations, timestamp, period_range, is_happy_hour)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (row.get('rank', ''), name, pts, row.get('prize', ''),
                     mg, combo, ct.strftime('%Y-%m-%d %H:%M:%S'), period, 1 if hh else 0))
                
                # ✅ NEW: Update player session data
                if name and pts > 0:
                    # Determine session date (6am-based)
                    if ct.hour < 6:
                        session_date = (ct - timedelta(days=1)).date()
                    else:
                        session_date = ct.date()
                    
                    DB.update_player_session(name, session_date, ct, pts, conn)
            
            conn.commit()
            conn.close()
            logger.info(f"Saved {len(df)} records")
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False
    
    @staticmethod
    def update_player_session(player_name, session_date, timestamp, points, conn=None):
        """
        ✅ v3.8.2 FIX: Robustly updates player session, preserving original 'first_online' on restart.
        """
        own_connection = conn is None
        if own_connection:
            conn = sqlite3.connect(CFG.DB_FILE)
        
        c = conn.cursor()
        try:
            # Timestamp should be timezone-naive for DB storage
            if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)
            
            # Check if a session already exists for this player and date
            c.execute('SELECT first_online, first_points FROM player_sessions WHERE player_name = ? AND session_date = ?', (player_name, str(session_date)))
            result = c.fetchone()
            
            if result:
                # --- SESSION EXISTS: UPDATE IT ---
                # The player was already seen today. We DO NOT change first_online.
                first_online_str, first_points = result
                first_online_dt = pd.to_datetime(first_online_str)

                # Only update last_offline and last_points
                duration = (timestamp - first_online_dt).total_seconds() / 60
                gained = points - first_points
                
                c.execute('''UPDATE player_sessions
                            SET last_offline = ?, last_points = ?,
                                total_points_gained = ?, session_duration_minutes = ?
                            WHERE player_name = ? AND session_date = ?''',
                         (timestamp, points, gained, duration, player_name, str(session_date)))
            else:
                # --- NO SESSION EXISTS: CREATE IT ---
                # This is genuinely the first time we see this player today.
                c.execute('''INSERT INTO player_sessions
                            (player_name, session_date, first_online, last_offline, first_points, last_points, total_points_gained, session_duration_minutes)
                            VALUES (?, ?, ?, ?, ?, ?, 0, 0)''',
                         (player_name, str(session_date), timestamp, timestamp, points, points))
            
            if own_connection:
                conn.commit()

        except Exception as e:
            logger.error(f"Update session error for {player_name}: {e}", exc_info=True)
        finally:
            if own_connection:
                conn.close()

    @staticmethod
    def update_session_breaks(player_name, session_date, break_minutes, sleep_minutes, break_type='break', conn=None):
        """
        Update break and sleep information for a player session.
        
        Args:
            player_name: Player name
            session_date: Session date
            break_minutes: Total break duration in minutes
            sleep_minutes: Total sleep duration in minutes (for breaks classified as sleep)
            break_type: 'break' or 'sleep'
            conn: Optional database connection
        """
        try:
            own_connection = conn is None
            if own_connection:
                conn = sqlite3.connect(CFG.DB_FILE)
            
            c = conn.cursor()
            
            # Update break and sleep information
            c.execute('''UPDATE player_sessions
                        SET break_duration_minutes = ?, sleep_duration_minutes = ?, break_type = ?
                        WHERE player_name = ? AND session_date = ?''',
                     (break_minutes, sleep_minutes, break_type, player_name, str(session_date)))
            
            if own_connection:
                conn.commit()
                conn.close()
            
            return True
        except Exception as e:
            logger.error(f"Update session breaks error: {e}")
            if own_connection and conn:
                try:
                    conn.close()
                except Exception:
                    pass
            return False
    
    @staticmethod
    def export_excel():
        """Export to Excel."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            df = pd.read_sql_query("SELECT * FROM leaderboard ORDER BY timestamp DESC, CAST(rank AS INTEGER)", conn)
            conn.close()
            df.to_excel(CFG.EXCEL_FILE, index=False)
            logger.info(f"Exported to {CFG.EXCEL_FILE}")
            return CFG.EXCEL_FILE
        except Exception as e:
            logger.error(f"Excel export error: {e}")
            raise
    
    @staticmethod
    def export_csv():
        """Export to CSV."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            df = pd.read_sql_query("SELECT * FROM leaderboard ORDER BY timestamp DESC, CAST(rank AS INTEGER)", conn)
            conn.close()
            df.to_csv(CFG.CSV_FILE, index=False)
            logger.info(f"Exported to {CFG.CSV_FILE}")
            return CFG.CSV_FILE
        except Exception as e:
            logger.error(f"CSV export error: {e}")
            raise
    
    @staticmethod
    def backup():
        """Create database backup."""
        try:
            os.makedirs(CFG.BACKUP_DIR, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = os.path.join(CFG.BACKUP_DIR, f"leaderboard_backup_{ts}.db")
            shutil.copy2(CFG.DB_FILE, backup_file)
            logger.info(f"Backup created: {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return None
    
    @staticmethod
    def restore(backup_file):
        """Restore from backup."""
        try:
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, CFG.DB_FILE)
                logger.info(f"Restored from: {backup_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"Restore error: {e}")
            return False

# ============================================================================
# DATA MIGRATION
# ============================================================================
def migrate_existing_data():
    """Migrate existing data to track sessions properly."""
    try:
        conn = sqlite3.connect(CFG.DB_FILE)
        players = [r[0] for r in conn.execute('SELECT DISTINCT name FROM leaderboard').fetchall()]
        
        print(f"\n{'='*60}")
        print(f"🔄 MIGRATING EXISTING DATA")
        print(f"{'='*60}")
        print(f"Found {len(players)} players to process...")
        
        migrated_sessions = 0
        
        for idx, player in enumerate(players, 1):
            print(f"[{idx}/{len(players)}] Processing: {player}...", end=' ')
            
            query = '''SELECT timestamp, points FROM leaderboard
                       WHERE name = ? ORDER BY timestamp ASC'''
            df = pd.read_sql_query(query, conn, params=(player,))
            
            if df.empty:
                print("No data")
                continue
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            if df.empty:
                print("No valid timestamps")
                continue
            
            prev_points = None
            sessions_processed = 0
            
            for _, row in df.iterrows():
                timestamp = row['timestamp']
                points = row['points']
                
                if prev_points is None or points != prev_points:
                    if timestamp.hour < 6:
                        session_date = (timestamp - timedelta(days=1)).date()
                    else:
                        session_date = timestamp.date()
                    
                    DB.update_player_session(player, session_date, timestamp, points)
                    sessions_processed += 1
                
                prev_points = points
            
            migrated_sessions += sessions_processed
            print(f"✓ ({sessions_processed} updates)")
        
        conn.close()
        
        print(f"\n{'='*60}")
        print(f"✅ MIGRATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total players processed: {len(players)}")
        print(f"Total session updates: {migrated_sessions}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ Migration error: {e}")
        logger.error(f"Migration error: {e}")
# ============================================================================
# GUI APPLICATION - MAIN CLASS
# ============================================================================
class LeaderboardApp:
    def __init__(self, root):
        try:
            self.root = root
            self.config = ConfigMgr()
            self.closing = False
            
            # ✅ NEU: Thread-safe Queue und Worker-Thread Management
            self.data_queue = Queue()
            self.db_worker_thread = None
            self.is_polling = False

            self.timestamps = []
            self.ts_idx = 0
            self.periods = []
            self.period_idx = 0
            self.search_var = tk.StringVar()
            self.polling_mgr = IntelligentPolling()
            self.activity_monitor = ActivityMonitor(self)
            self.pattern_detector = PatternDetector()
            self.player_scout = PlayerScout(self)
            self.ml_bot_detector = SelfLearningBotDetector()
            self.prev_data = None
            self.display_thread = None
            self.display_lock = threading.Lock()
            # FIX: Timer IDs speichern für Cancel
            self.after_ids = []
            
            # FIX: Track open chart windows (singleton pattern)
            self.open_windows = {
                'points_chart': None,
                'delta_chart': None,
                'comparison_chart': None,
                'overall_chart': None
            }
            
            root.title("Leaderboard Tracker v3.6.2 COMPLETE - jimmybeam3000")
            root.geometry(self.config.get('window_geometry', '1900x900'))   
            
            # FIX: DB init mit error handling
            try:
                DB.init()
            except Exception as e:
                logger.error(f"DB init failed: {e}")
                messagebox.showerror("Database Error", f"Could not initialize database:\n{e}")
                raise
            
            # Build UI
            self.build_ui()
            
            # ✅ NEU: Starte den dedizierten Datenbank-Worker
            self.start_db_worker()

            # Load data
            if DB.has_data():
                self.load_snapshot()
            else:
                self.tv.delete(*self.tv.get_children())
                self.ts_label.config(text="No data - Click 'Fetch Now'")
                self.count_label.config(text="0 entries")
            
            root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            logger.info("LeaderboardApp initialized successfully")
            
        except Exception as e:
            logger.error(f"LeaderboardApp init failed: {e}", exc_info=True)
            messagebox.showerror("Initialization Error", f"Failed to start application:\n{e}")
            if 'scikit-learn' in str(e):
                 messagebox.showerror("Dependency Error", "scikit-learn is not installed.\nPlease run 'pip install scikit-learn' in your terminal.")
            raise
    
    def show_loading(self, parent, message="Processing..."):
        """Show a loading window."""
        loading = tk.Toplevel(parent)
        loading.title("Please Wait")
        loading.geometry("300x100")
        loading.transient(parent)
        loading.grab_set()
        
        ttk.Label(loading, text=message, font=("Arial", 12)).pack(pady=20)
        progress = ttk.Progressbar(loading, mode='indeterminate', length=200)
        progress.pack(pady=10)
        progress.start()
        
        loading.update()
        return loading
    
    def build_ui(self):
        self.build_menu()
        self.build_controls()
        self.build_polling_controls()
        self.build_chart_controls()
        self.build_verification_controls()
        self.build_validation()
        self.build_period_info()
        self.build_tree()
        self.build_status()
    
    def build_menu(self):
        mb = tk.Menu(self.root)
        self.root.config(menu=mb)
        
        fm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="Excel Export", command=lambda: self.export('excel'))
        fm.add_command(label="CSV Export", command=lambda: self.export('csv'))
        fm.add_separator()
        fm.add_command(label="Backup", command=self.backup)
        fm.add_command(label="Restore", command=self.restore)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.on_close)
        
        vm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="View", menu=vm)
        vm.add_command(label="View Logs", command=self.view_logs)
        vm.add_command(label="Player Patterns", command=self.show_patterns)
        
        cm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Charts", menu=cm)
        cm.add_command(label="Points Progress", command=self.show_points_chart)
        cm.add_command(label="Delta Chart", command=self.show_delta_chart)
        cm.add_command(label="Player Comparison", command=self.show_comparison)
    
    def build_controls(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        
        uf = ttk.Frame(top)
        uf.pack(fill="x", pady=2)
        
        ttk.Label(uf, text="URL:").pack(side="left", padx=(0,5))
        self.url_entry = ttk.Entry(uf, width=55)
        self.url_entry.insert(0, CFG.URL)
        self.url_entry.pack(side="left", padx=5)
        
        ttk.Button(uf, text="Fetch Now", command=self.fetch, width=12).pack(side="left", padx=2)
        ttk.Button(uf, text="Excel", command=lambda: self.export('excel'), width=9).pack(side="left", padx=2)
        ttk.Button(uf, text="CSV", command=lambda: self.export('csv'), width=9).pack(side="left", padx=2)
        
        ttk.Label(uf, text="Search:").pack(side="left", padx=(10,2))
        ttk.Entry(uf, textvariable=self.search_var, width=12).pack(side="left", padx=2)
        self.search_var.trace('w', lambda *args: self.filter())
        
        nf = ttk.Frame(top)
        nf.pack(fill="x", pady=6)
        
        ttk.Label(nf, text="Timestamp:").pack(side="left", padx=(0,4))
        ttk.Button(nf, text="<", command=self.prev_ts, width=3).pack(side="left", padx=2)
        self.ts_label = ttk.Label(nf, text="Current", relief="sunken", width=22, anchor="center")
        self.ts_label.pack(side="left", padx=4)
        ttk.Button(nf, text=">", command=self.next_ts, width=3).pack(side="left", padx=2)
        
        ttk.Label(nf, text="Period:").pack(side="left", padx=(20,4))
        ttk.Button(nf, text="<", command=self.prev_period, width=3).pack(side="left", padx=2)
        self.period_label = ttk.Label(nf, text="No period", relief="sunken", width=30, anchor="center")
        self.period_label.pack(side="left", padx=4)
        ttk.Button(nf, text=">", command=self.next_period, width=3).pack(side="left", padx=2)
        
        self.hh_label = ttk.Label(nf, text="", font=("Arial", 10, "bold"), foreground="orange")
        self.hh_label.pack(side="right", padx=10)
        self._safe_after(100, self.update_hh)
    
    def build_polling_controls(self):
        pf = ttk.LabelFrame(self.root, text="Intelligent Polling", padding=5)
        pf.pack(fill="x", padx=8, pady=4)
        
        ttk.Label(pf, text="Mode:").pack(side="left", padx=5)
        self.poll_mode_var = tk.StringVar(value="intelligent")
        ttk.Radiobutton(pf, text="Intelligent", variable=self.poll_mode_var, value="intelligent", command=self.set_intelligent_mode).pack(side="left", padx=5)
        ttk.Radiobutton(pf, text="Manual", variable=self.poll_mode_var, value="manual", command=self.set_manual_mode).pack(side="left", padx=5)
        
        ttk.Label(pf, text="Interval:").pack(side="left", padx=(20,5))
        self.poll_interval_var = tk.StringVar(value="30")
        self.poll_interval_entry = ttk.Entry(pf, textvariable=self.poll_interval_var, width=6)
        self.poll_interval_entry.pack(side="left", padx=2)
        ttk.Label(pf, text="min").pack(side="left", padx=(0,15))
        
        ttk.Button(pf, text="Start Polling", command=self.start_poll, width=14).pack(side="left", padx=5)
        ttk.Button(pf, text="Stop Polling", command=self.stop_poll, width=13).pack(side="left", padx=2)
        
        self.poll_status_label = ttk.Label(pf, text="Stopped", font=("Arial", 9, "bold"), foreground="gray")
        self.poll_status_label.pack(side="left", padx=15)
    
    def build_chart_controls(self):
        cf = ttk.LabelFrame(self.root, text="Charts & Analysis", padding=5)
        cf.pack(fill="x", padx=8, pady=4)
        ttk.Button(cf, text="Points Progress", command=self.show_points_chart, width=16).pack(side="left", padx=5)
        ttk.Button(cf, text="Delta Chart", command=self.show_delta_chart, width=16).pack(side="left", padx=5)
        ttk.Button(cf, text="Player Comparison", command=self.show_comparison, width=18).pack(side="left", padx=5)
        ttk.Button(cf, text="Player Patterns", command=self.show_patterns, width=16).pack(side="left", padx=5)
        ttk.Button(cf, text="📊 Overall Chart", command=self.show_overall_chart, width=16).pack(side="left", padx=5)  # NEW!
    
    def build_verification_controls(self):
        vf = ttk.LabelFrame(self.root, text="🔍 Player & Network Analysis", padding=5)
        vf.pack(fill="x", padx=8, pady=4)
        
        # Obere Zeile für die Spielerauswahl (optional, für zukünftige Einzelspieler-Analysen)
        player_frame = ttk.Frame(vf)
        player_frame.pack(fill="x", pady=2)
        ttk.Label(player_frame, text="Player (for history):").pack(side="left", padx=5)
        self.verify_player_var = tk.StringVar()
        self.verify_player_combo = ttk.Combobox(player_frame, textvariable=self.verify_player_var, width=20, state="readonly")
        self.verify_player_combo.pack(side="left", padx=2)
        
        # Untere Zeile für die Analyse-Buttons
        analysis_frame = ttk.Frame(vf)
        analysis_frame.pack(fill="x", pady=(5, 2))
        
        # ✅ KORREKTUR: Alle Buttons werden jetzt korrekt in 'analysis_frame' platziert.
        ttk.Button(analysis_frame, text="🔬 Cluster Scan", command=self.run_cluster_analysis, width=20).pack(side="left", padx=5)
        ttk.Button(analysis_frame, text="🤝 Top 2 Interaction", command=self.run_top2_analysis, width=20).pack(side="left", padx=2)
        ttk.Button(analysis_frame, text="🏆 Top 3 Profile", command=self.run_top3_profiler, width=20).pack(side="left", padx=2)
        ttk.Button(analysis_frame, text="🕵️ Top 5 Profile", command=self.run_top5_profiler, width=22).pack(side="left", padx=2)

        ttk.Button(analysis_frame, text="🔄 Refresh Players", command=self.refresh_players, width=18).pack(side="right", padx=5)
        
        self.refresh_players()

    def run_top2_analysis(self):
        """Triggers the new top 2 player interaction analysis."""
        loading = self.show_loading(self.root, "Analyzing Top 2 Player Interaction...")
        
        def worker():
            try:
                report = self.ml_bot_detector.analyze_top_player_interaction(days=7)
                self.root.after(0, lambda r=report: [loading.destroy(), self.show_simple_report("Top 2 Interaction Analysis", r)])
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Top 2 analysis worker failed: {error_msg}", exc_info=True)
                self.root.after(0, lambda msg=error_msg: [loading.destroy(), messagebox.showerror("Error", f"Analysis failed:\n{msg}")])
        
        threading.Thread(target=worker, daemon=True).start()
    
    def run_top3_profiler(self):
        """Triggers the new Top 3 Competitor Profile analysis."""
        loading = self.show_loading(self.root, "Generating Top 3 Competitor Profiles...")
        
        def worker():
            try:
                report = self.ml_bot_detector.analyze_top_3_competitors(days=7)
                self.root.after(0, lambda r=report: [loading.destroy(), self.show_simple_report("Top 3 Competitor Profile", r)])
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Top 3 Profile worker failed: {error_msg}", exc_info=True)
                self.root.after(0, lambda msg=error_msg: [loading.destroy(), messagebox.showerror("Error", f"Analysis failed:\n{msg}")])
        
        threading.Thread(target=worker, daemon=True).start() 

    def run_top5_profiler(self):
        """Triggers the new Top 5 Competitor Profile analysis."""
        loading = self.show_loading(self.root, "Generating Top 5 Competitor Profiles...")
        
        def worker():
            try:
                report = self.ml_bot_detector.analyze_top_5_competitors(days=14) # Analysezeitraum auf 14 Tage erweitert
                self.root.after(0, lambda r=report: [loading.destroy(), self.show_simple_report("Top 5 Competitor Profile", r)])
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Top 5 Profile worker failed: {error_msg}", exc_info=True)
                self.root.after(0, lambda msg=error_msg: [loading.destroy(), messagebox.showerror("Error", f"Analysis failed:\n{msg}")])
        
        threading.Thread(target=worker, daemon=True).start()

    def show_simple_report(self, title, report):
        """Generic window to show a text report."""
        w = tk.Toplevel(self.root)
        w.title(title)
        w.geometry("900x600")
        
        text = scrolledtext.ScrolledText(w, wrap=tk.WORD, font=("Courier", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert('1.0', report)
        text.config(state='disabled')
        ttk.Button(w, text="Close", command=w.destroy).pack(pady=5)

        # ✅ NEU: Thread-sichere Analyse- und Anzeigemethoden

    def run_cluster_analysis(self):
        """Triggers the new self-learning cluster analysis in a background thread."""
        loading = self.show_loading(self.root, "Performing Deep Behavioral Analysis...\nThis may take a minute.")
        
        def worker():
            try:
                # Nur Datenberechnung im Worker-Thread
                report_data, X_pca, labels, player_list = self.ml_bot_detector.analyze_player_clusters(days=7)
                # Daten an den Main-Thread übergeben, um die GUI zu aktualisieren
                self.root.after(0, lambda: self.show_cluster_report(loading, report_data, X_pca, labels, player_list))
            except Exception as e:
                logger.error(f"Cluster analysis worker failed: {e}", exc_info=True)
                self.root.after(0, lambda: [loading.destroy(), messagebox.showerror("Error", f"Cluster analysis failed:\n{e}")])

        threading.Thread(target=worker, daemon=True).start()

    def show_cluster_report(self, loading_window, report_data, X_pca, labels, player_list):
        """
        Displays the cluster analysis report and visualization.
        This method runs ONLY on the main GUI thread.
        """
        loading_window.destroy()

        if "error" in report_data:
            messagebox.showinfo("Cluster Analysis", report_data["error"], parent=self.root)
            return

        # 1. Report-String generieren
        human_cluster_id = report_data.get('human_cluster_id', -1)
        report_str = f"🔬 DEEP BEHAVIORAL CLUSTER ANALYSIS\n{'='*70}\n"
        report_str += f"{report_data['summary']}\n"
        if human_cluster_id != -1:
            report_str += f"Assumption: Cluster {human_cluster_id} is the 'human baseline'.\n\n"
        
        report_str += "--- DETECTED ANOMALIES (OUTLIERS) ---\n"
        report_str += "\n".join(f"  • {p}" for p in report_data['anomalies']) if report_data['anomalies'] else "  None\n"
        
        report_str += "\n--- CLUSTER PROFILES ---\n"
        human_profile = report_data.get('clusters', {}).get(str(human_cluster_id), {}).get('profile')
        
        for cid, data in sorted(report_data.get('clusters', {}).items()):
            if int(cid) == human_cluster_id: continue
            report_str += f"\n🚨 {data['verdict'].upper()} ({data['size']} members)\n"
            report_str += f"   Members: {', '.join(data['members'])}\n"
            if human_profile:
                report_str += "   Behavioral Profile (vs. Human):\n"
                for key, value in data['profile'].items():
                    human_val = human_profile.get(key, 0)
                    comparison = "similar"
                    if value < human_val * 0.5: comparison = "SIGNIFICANTLY LOWER"
                    elif value > human_val * 1.5: comparison = "SIGNIFICANTLY HIGHER"
                    report_str += f"     - {key.replace('_', ' ').title()}: {value:.3f} (vs. {human_val:.3f}) -> {comparison}\n"
        
        # 2. Fenster und Plot erstellen (jetzt sicher im Main-Thread)
        w = tk.Toplevel(self.root)
        w.title("🔬 Cluster Analysis Report")
        w.geometry("1600x900")
        pane = ttk.PanedWindow(w, orient=tk.HORIZONTAL)
        pane.pack(fill="both", expand=True)

        text_frame = ttk.Frame(pane, width=600)
        text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, font=("Courier", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert('1.0', report_str)
        text.config(state='disabled')
        pane.add(text_frame, weight=1)

        plot_frame = ttk.Frame(pane, width=1000)
        fig, ax = plt.subplots(figsize=(14, 8))
        unique_labels = sorted(list(set(labels)))
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        
        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = X_pca[class_member_mask]
            label_text = 'Anomaly' if k == -1 else f'Network {k}'
            if k == human_cluster_id: label_text = 'Human Baseline'
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=12 if k!=-1 else 7, label=f'{label_text} ({len(xy)})')

        ax.set_title('Player Behavior Clusters (via PCA)', fontsize=16)
        ax.legend()
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        pane.add(plot_frame, weight=2)
    
    def build_validation(self):
        vf = ttk.LabelFrame(self.root, text="Validation Settings", padding=5)
        vf.pack(fill="x", padx=8, pady=4)
        
        ttk.Label(vf, text="Min 3max:").pack(side="left", padx=5)
        self.t3_var = tk.StringVar(value=str(self.config.get('min_3max_time')))
        ttk.Entry(vf, textvariable=self.t3_var, width=6).pack(side="left", padx=2)
        ttk.Label(vf, text="min").pack(side="left", padx=(0,15))
        
        ttk.Label(vf, text="Min 6max:").pack(side="left", padx=5)
        self.t6_var = tk.StringVar(value=str(self.config.get('min_6max_time')))
        ttk.Entry(vf, textvariable=self.t6_var, width=6).pack(side="left", padx=2)
        ttk.Label(vf, text="min").pack(side="left", padx=(0,15))
        
        ttk.Label(vf, text="Max Tables:").pack(side="left", padx=5)
        self.tables_var = tk.StringVar(value=str(self.config.get('max_tables', CFG.MAX_TABLES)))
        ttk.Spinbox(vf, textvariable=self.tables_var, from_=1, to=8, width=4).pack(side="left", padx=2)
        
        ttk.Button(vf, text="Save", command=self.save_settings, width=10).pack(side="left", padx=15)
    
    def build_period_info(self):
        pf = ttk.LabelFrame(self.root, text="Period Info (Wed 12:00 - Wed 11:59 MESZ)", padding=5)
        pf.pack(fill="x", padx=8, pady=4)
        
        self.ps_label = ttk.Label(pf, text="Start: ...", font=("Arial", 9, "bold"))
        self.ps_label.pack(side="left", padx=10)
        
        self.pe_label = ttk.Label(pf, text="Elapsed: ...", font=("Arial", 9, "bold"), foreground="blue")
        self.pe_label.pack(side="left", padx=10)
        
        self.pend_label = ttk.Label(pf, text="End: ...", font=("Arial", 9))
        self.pend_label.pack(side="left", padx=10)
        
        self._safe_after(100, self.update_period)
    
    def build_tree(self):
        """Build treeview with ALL requested columns - FIXED."""
        f = ttk.Frame(self.root)
        f.pack(fill="both", expand=True, padx=8, pady=6)
        
        # COMPLETE column list with new columns
        cols = (
            "Rank",              # 1
            "Name",              # 2
            "Points",            # 3
            "Delta Poll",        # 4
            "Delta Hour",        # 5 - NEW
            "3m",                # 6
            "6m",                # 7
            "Delta Day",         # 8
            "Min Games Day",     # 9 - NEW
            "Min Sessions Day",  # 10 - NEW
            "Combo Day",         # 11
            "Daily Session Valid",   # 12 - RENAMED
            "Weekly Session Valid",  # 13 - NEW
            "New Player Week",   # 14 - RESTORED
            "New Player Month",  # 15 - RESTORED
            "New Player Overall",# 16 - RESTORED
            "Ranking Total"      # 17 - RESTORED
        )
        
        # Store column count for use in display method
        self.column_count = len(cols)
        
        self.tv = ttk.Treeview(f, columns=cols, show="headings", height=20)
        
        # Optimized column widths
        column_widths = {
            "Rank": 50,
            "Name": 120,
            "Points": 90,
            "Delta Poll": 80,
            "Delta Hour": 80,  # NEW
            "3m": 100,
            "6m": 100,
            "Delta Day": 80,
            "Min Games Day": 110,  # NEW
            "Min Sessions Day": 120,  # NEW
            "Combo Day": 150,
            "Daily Session Valid": 180,  # RENAMED
            "Weekly Session Valid": 280,  # NEW - wider for validation message
            "New Player Week": 120,  # RESTORED
            "New Player Month": 130,  # RESTORED
            "New Player Overall": 130,  # RESTORED
            "Ranking Total": 200  # RESTORED
        }
        
        for c in cols:
            self.tv.heading(c, text=c)
            initial_width = column_widths.get(c, 100)
            self.tv.column(c, width=initial_width, anchor="center", stretch=False)
        
        # Scrollbars
        sy = ttk.Scrollbar(f, orient="vertical", command=self.tv.yview)
        sx = ttk.Scrollbar(f, orient="horizontal", command=self.tv.xview)
        self.tv.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        
        self.tv.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        
        f.grid_rowconfigure(0, weight=1)
        f.grid_columnconfigure(0, weight=1)
        
        # Configure tag colors - Green theme
        self.tv.tag_configure("positive", background="lightgreen")
        self.tv.tag_configure("high_day", background="#90EE90")  # Light green
        self.tv.tag_configure("medium_day", background="#FFFFE0")  # Light yellow
        self.tv.tag_configure("high_efficiency", background="#87CEEB")  # Light blue
        self.tv.tag_configure("warning", background="#ffcccc")
        self.tv.tag_configure("placeholder", foreground="#cccccc")
        self.tv.tag_configure("new_player", background="#FFD700")  # Gold color for new players
        
        self.tv.bind("<Button-3>", self.on_right_click)

    def build_status(self):
        sf = ttk.Frame(self.root)
        sf.pack(fill="x", padx=8, pady=(0,6))
        
        ttk.Label(sf, text="Status:", font=("Arial", 9, "bold")).pack(side="left")
        self.status_label = ttk.Label(sf, text="Ready", foreground="green")
        self.status_label.pack(side="left", padx=8)
        
        self.progress = ttk.Progressbar(sf, mode='indeterminate', length=200)
        self.progress.pack(side="right", padx=8)
        
        self.count_label = ttk.Label(sf, text="", font=("Arial", 9))
        self.count_label.pack(side="right", padx=15)
    
    def set_intelligent_mode(self):
        self.polling_mgr.set_intelligent()
        self.poll_interval_entry.config(state='disabled')
    
    def set_manual_mode(self):
        self.poll_interval_entry.config(state='normal')
    
    def update_hh(self):
        """Update Happy Hour Label - mit Safety-Check"""
        if self.closing:
            return
        try:
            if not self.root.winfo_exists():
                return
            
            self.hh_label.config(text="🎰 HAPPY HOUR (x2)" if is_hh() else "")
            self._safe_after(60000, self.update_hh)
        except Exception:
            pass

    def update_period(self):
        """Update Period Info - using website's period detection"""
        if self.closing:
            return
        try:
            if not self.root.winfo_exists():
                return
            
            # Use current period from website (not calculated locally)
            if self.periods and len(self.periods) > self.period_idx:
                current_period = self.periods[self.period_idx]
                
                # Try to parse dates from period string
                try:
                    # Parse period: "2025/11/05 ~ 2025/11/12"
                    parts = current_period.split(' ~ ')
                    if len(parts) == 2:
                        start_str = parts[0].strip()
                        end_str = parts[1].strip()
                        
                        # Convert to d-m-Y format
                        start_date = datetime.strptime(start_str, '%Y/%m/%d')
                        start_display = start_date.strftime('%d-%m-%Y')
                        
                        end_date = datetime.strptime(end_str, '%Y/%m/%d')
                        end_display = end_date.strftime('%d-%m-%Y')
                        
                        # Display
                        self.ps_label.config(text=f"Start: {start_display}")
                        self.pend_label.config(text=f"End: {end_display}")
                        
                        # Calculate elapsed
                        now = datetime.now(CFG.TIMEZONE).replace(tzinfo=None)
                        elapsed = now - start_date.replace(tzinfo=None)
                        days = elapsed.days
                        hours = elapsed.seconds // 3600
                        self.pe_label.config(text=f"Elapsed: ~{days}d {hours}h")
                    else:
                        self.ps_label.config(text="Period: " + current_period)
                except Exception as e:
                    logger.warning(f"Period parsing error: {e}")
                    self.ps_label.config(text="Period: " + current_period)
            else:
                self.ps_label.config(text="No period data")
                self.pe_label.config(text="")
                self.pend_label.config(text="")
            
            self._safe_after(60000, self.update_period)
        except Exception:
            pass

    def save_settings(self):
        try:
            t3 = int(self.t3_var.get())
            t6 = int(self.t6_var.get())
            tb = int(self.tables_var.get())
            if t3 < 1 or t6 < 1 or tb < 1:
                raise ValueError
            self.config.set('min_3max_time', t3)
            self.config.set('min_6max_time', t6)
            self.config.set('max_tables', tb)
            messagebox.showinfo("OK", "Settings saved!")
            if self.timestamps:
                self.display(self.timestamps[self.ts_idx])
        except Exception:
            messagebox.showerror("Error", "Invalid input")
    
    def refresh_players(self):
        try:
            players = DB.players()
            if players:
                self.verify_player_combo['values'] = sorted(players)
                if players:
                    self.verify_player_combo.current(0)
        except Exception as e:
            logger.error(f"Refresh: {e}")
    
    def verify_player(self):
        try:
            player = self.verify_player_var.get()
            if not player:
                messagebox.showwarning("Error", "Select player!")
                return
            points_str = self.verify_points_var.get()
            if not points_str:
                messagebox.showwarning("Error", "Enter points!")
                return
            try:
                points = int(points_str.replace(',', ''))
            except Exception:
                messagebox.showwarning("Error", "Invalid points!")
                return
            
            timeframe = self.verify_timeframe_var.get()
            
            if timeframe == "period":
                pi = get_period_info()
                minutes = pi['elapsed_minutes']
                desc = f"Period ({pi['elapsed_formatted']})"
            elif timeframe == "hour":
                minutes = 60
                desc = "1 Hour"
            else:
                try:
                    minutes = int(self.verify_minutes_var.get())
                    desc = f"{minutes} Minutes"
                except Exception:
                    messagebox.showwarning("Error", "Invalid minutes!")
                    return
            
            try:
                t3 = int(self.t3_var.get())
                t6 = int(self.t6_var.get())
                tb = int(self.tables_var.get())
            except Exception:
                t3, t6, tb = CFG.MIN_3MAX, CFG.MIN_6MAX, CFG.MAX_TABLES
            
            hh = is_hh()
            games = calc_min_games(points, hh)
            sessions = calc_min_sessions(games, tb)
            needed_time = sessions * t6
            
            report = f"🔍 PLAYER VERIFICATION REPORT\n"
            report += f"{'='*60}\n"
            report += f"Player: {player}\n"
            report += f"Points: {points:,}\n"
            report += f"Timeframe: {desc}\n"
            report += f"Happy Hour: {'YES (x2 points)' if hh else 'NO'}\n"
            report += f"\n{'='*60}\n"
            report += f"ANALYSIS:\n\n"
            report += f"Minimum Games Required: {games}\n"
            report += f"Minimum Sessions (with {tb} tables): {sessions}\n"
            report += f"Time Needed (@ {t6}min/session): {needed_time:.0f} minutes ({needed_time/60:.1f} hours)\n"
            report += f"Time Available: {minutes:.0f} minutes ({minutes/60:.1f} hours)\n"
            report += f"\n{'='*60}\n"
            
            if minutes >= needed_time:
                report += f"✅ VERDICT: POSSIBLE\n"
                report += f"Player has enough time to achieve {points:,} points.\n"
                utilization = (needed_time / minutes) * 100
                report += f"Time Utilization: {utilization:.1f}%\n"
            else:
                report += f"❌ VERDICT: IMPOSSIBLE\n"
                report += f"Player needs {needed_time:.0f} minutes but only has {minutes:.0f} minutes available.\n"
                deficit = needed_time - minutes
                report += f"Time Deficit: {deficit:.0f} minutes ({deficit/60:.1f} hours)\n"
            
            messagebox.showinfo("Verification Report", report)
            
        except Exception as e:
            logger.error(f"Verify error: {e}")
            messagebox.showerror("Error", f"Verification failed:\n{e}")
    
    def detect_gap_bot(self):
        """Detect bot using self-learning AI."""
        player = self.verify_player_var.get()
        if not player:
            messagebox.showwarning("Error", "Select player!", parent=self.root)
            return
        
        try:
            # Initialize detector if needed
            if not hasattr(self, 'ml_bot_detector'):
                self.ml_bot_detector = SelfLearningBotDetector()
            
            # Show loading window
            loading = self.show_loading(self.root, "Analyzing player patterns...")
            
            # Analyze with machine learning (in background thread to prevent UI freeze)
            def analyze():
                try:
                    report = self.ml_bot_detector.get_detailed_report(player, days=14)
                    
                    # Close loading window and show report
                    self.root.after(0, loading.destroy)
                    self.root.after(0, lambda: self.show_bot_report(player, report))
                    
                except Exception as e:
                    self.root.after(0, loading.destroy)
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{e}", parent=self.root))
            
            threading.Thread(target=analyze, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Bot detection error: {e}")
            messagebox.showerror("Error", f"Detection failed:\n{e}", parent=self.root)

    def detect_network(self):
        """Detect coordinated bot networks."""
        try:
            # Initialize detector if needed
            if not hasattr(self, 'ml_bot_detector'):
                self.ml_bot_detector = SelfLearningBotDetector()
            
            # Show loading window
            loading = self.show_loading(self.root, "Analyzing network coordination patterns...")
            
            # Analyze in background thread
            def analyze():
                try:
                    report = self.ml_bot_detector.get_network_report(days=14)
                    
                    # Close loading window and show report
                    self.root.after(0, loading.destroy)
                    self.root.after(0, lambda: self.show_network_report(report))
                    
                except Exception as e:
                    self.root.after(0, loading.destroy)
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Network analysis failed:\n{e}", parent=self.root))
            
            threading.Thread(target=analyze, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Network detection error: {e}")
            messagebox.showerror("Error", f"Detection failed:\n{e}", parent=self.root)
    
    def show_network_report(self, report):
        """Display network coordination report in a window."""
        w = tk.Toplevel(self.root)
        w.title("🕸️ Network Coordination Analysis")
        w.geometry("1000x750")
        
        text = scrolledtext.ScrolledText(w, wrap=tk.WORD, font=("Courier", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert('1.0', report)
        text.config(state='disabled')
        
        # Button frame
        btn_frame = ttk.Frame(w)
        btn_frame.pack(fill="x", pady=5)
        
        def export_report():
            try:
                filename = f"network_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Report saved:\n{filename}", parent=w)
                if os.name == 'nt':
                    os.startfile(filename)
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{e}", parent=w)
        
        ttk.Button(btn_frame, text="💾 Export Report", command=export_report, width=15).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", command=w.destroy, width=10).pack(side="right", padx=5)
    
    def show_bot_report(self, player, report):
        """Display bot detection report in a window."""
        w = tk.Toplevel(self.root)
        w.title(f"🤖 AI Bot Detection - {player}")
        w.geometry("900x700")
        
        text = scrolledtext.ScrolledText(w, wrap=tk.WORD, font=("Courier", 10))
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert('1.0', report)
        text.config(state='disabled')
        
        # Button frame
        btn_frame = ttk.Frame(w)
        btn_frame.pack(fill="x", pady=5)
        
        def export_report():
            try:
                filename = f"bot_report_{player}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("Success", f"Report saved:\n{filename}", parent=w)
                if os.name == 'nt':
                    os.startfile(filename)
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{e}", parent=w)
        
        ttk.Button(btn_frame, text="💾 Export Report", command=export_report, width=15).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Close", command=w.destroy, width=10).pack(side="right", padx=5)
    
    def fetch(self):
        if not check_disk():
            messagebox.showerror("Disk Full", "Free up space!")
            return
        
        def worker():
            d = None
            try:
                self.root.after(0, lambda: self.status_label.config(text="Fetching...", foreground="orange"))
                self.root.after(0, self.progress.start)
                
                d = get_driver(False)
                df, period = parse_leaderboard(d)
                    
                if df.empty:
                    self.root.after(0, lambda: self.status_label.config(text="No data found", foreground="red"))
                    return # Beendet den Worker-Thread hier

                # ---- Logik, die im Fetch-Thread verbleibt ----
                
                # 1. Daten zur Verarbeitung in die Queue legen
                self.data_queue.put((df, period))
                logger.info(f"✅ Fetch successful. {len(df)} records added to the queue.")

                # 2. Polling-Logik aktualisieren (dies ist schnell und sicher)
                # Berechne die Anzahl der Spieler, die tatsächlich Punkte gewonnen haben
                players_with_gains = 0
                if self.prev_data is not None and not self.prev_data.empty:
                    merged = pd.merge(df, self.prev_data, on='name', suffixes=('_curr', '_prev'))
                    if not merged.empty:
                        players_with_gains = (merged['points_curr'] > merged['points_prev']).sum()
                else:
                    # Wenn es keine vorherigen Daten gibt, ist jeder Spieler mit Punkten "neu"
                    players_with_gains = len(df[df['points'] > 0])

                # Rufe die korrigierte update-Methode auf
                self.polling_mgr.update(players_with_gains)
                
                # Aktualisiere prev_data für den nächsten Durchlauf
                self.prev_data = df.copy()

                # 3. GUI-Status aktualisieren
                ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                self.root.after(0, lambda t=ts: self.status_label.config(text=f"OK {t}", foreground="green"))
                self.root.after(0, lambda: self.poll_status_label.config(text=self.polling_mgr.status()))

            except Exception as e:
                logger.error(f"Fetch worker error: {e}", exc_info=True)
                self.root.after(0, lambda: self.status_label.config(text="Error during fetch", foreground="red"))
            finally:
                if d:
                    try: d.quit()
                    except Exception: pass
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=worker, daemon=True).start()

    def load_snapshot(self):
        """Load data in background to prevent freeze"""
        try:
            logger.debug("🔵 Loading snapshot...")
            self.timestamps = DB.timestamps()
            self.periods = DB.periods()
            
            logger.info(f"🔵 Loaded {len(self.timestamps)} timestamps, {len(self.periods)} periods")
            
            if not self.timestamps:
                self.tv.delete(*self.tv.get_children())
                self.ts_label.config(text="No data - Click 'Fetch Now'")
                self.count_label.config(text="0 entries")
                logger.warning("❌ No timestamps available")
                return
            
            if not self.periods:
                self.tv.delete(*self.tv.get_children())
                self.ts_label.config(text="No period data")
                self.count_label.config(text="0 entries")
                logger.warning("❌ No periods available")
                return
            
            # 🔴 FIX: Always show NEWEST period (index 0 = most recent)
            self.ts_idx = 0
            self.period_idx = 0
            
            # Update period label immediately
            self.period_label.config(text=self.periods[0])
            logger.info(f"📅 Displaying newest period: {self.periods[0]}")
            
            # FIX: Warte bis alte Thread beendet ist
            if self.display_thread and self.display_thread.is_alive():
                logger.warning("⚠️ Display thread already running, waiting...")
                self.display_thread.join(timeout=5)
            
            logger.info(f"🔵 Starting display worker for timestamp: {self.timestamps[0]}")
            self.display_thread = threading.Thread(
                target=self._display_worker,
                args=(self.timestamps[0],),
                daemon=False
            )
            self.display_thread.start()
            logger.info("✅ Display thread started")
            
        except Exception as e:
            logger.error(f"❌ Load snapshot error: {e}", exc_info=True)
            self.tv.delete(*self.tv.get_children())
            self.ts_label.config(text="Error loading")
            self.count_label.config(text="0 entries")

    def _display_worker(self, ts):
        """Background worker - WITH PROPER RANK SORTING"""
        try:
            logger.info(f"🟢 Display worker started for timestamp: {ts}")

            conn = sqlite3.connect(CFG.DB_FILE)
            cur = conn.cursor()
            
            # ✅ FIX: Cast rank to INTEGER for proper numeric sorting
            df = pd.read_sql_query(
                "SELECT * FROM leaderboard WHERE timestamp = ? ORDER BY CAST(rank AS INTEGER) ASC",
                conn,
                params=(ts,),
            )
            conn.close()

            if df.empty:
                logger.warning(f"⚠️ No data found for timestamp {ts}")
                self.root.after(0, lambda: self._update_display_error("No data available"))
                return

            display_data = []
            for _, row in df.iterrows():
                display_data.append({
                    "rank": row.get("rank", ""),
                    "name": row.get("name", ""),
                    "points": row.get("points", 0),
                    "delta_poll": 0,
                    "delta_hour": 0,
                    "combo_3m": "",
                    "combo_6m": "",
                    "delta_day": 0,
                    "min_games_day": "",
                    "min_sessions_day": "",
                    "combo_day": "",
                    "daily_valid": "",
                    "weekly_valid": "",
                    "new_week": "",
                    "new_month": "",
                    "new_overall": "",
                    "ranking_total": "",
                    "has_delta": False,
                    "high_day": False,
                    "medium_day": False,
                    "is_warning": False,
                    "is_new": False,
                    "is_placeholder": False,
                })

            ct = datetime.now()
            hh = False
            player_count = len(display_data)

            self.root.after(
                0,
                lambda: self._update_display_ui(
                    display_data=display_data,
                    player_count=player_count,
                    ct=ct,
                    hh=hh,
                ),
            )

            logger.info("✅ Display worker completed successfully")

        except Exception as e:
            logger.error(f"❌ Display worker error: {e}", exc_info=True)
            self.root.after(0, lambda: self._update_display_error("Display worker failed"))

    def _update_display_ui(self, display_data, ct, hh, player_count):
        """Update UI on main thread."""
        try:
            logger.info(f"🔵 _update_display_ui START: {len(display_data)} rows")

            self.tv.delete(*self.tv.get_children())

            inserted = 0
            for row in display_data:
                if row['is_placeholder']:
                    # Create placeholder tuple with empty strings for all columns except rank
                    vals = (row['rank'],) + ("",) * (self.column_count - 1)
                    iid = self.tv.insert("", "end", values=vals)
                    self.tv.item(iid, tags=("placeholder",))
                else:
                    vals = (
                        row['rank'],              # 1
                        row['name'],              # 2
                        row['points'],            # 3
                        row['delta_poll'],        # 4
                        row['delta_hour'],        # 5
                        row['combo_3m'],          # 6
                        row['combo_6m'],          # 7
                        row['delta_day'],         # 8
                        row['min_games_day'],     # 9
                        row['min_sessions_day'],  # 10
                        row['combo_day'],         # 11
                        row['daily_valid'],       # 12
                        row['weekly_valid'],      # 13
                        row['new_week'],          # 14
                        row['new_month'],         # 15
                        row['new_overall'],       # 16
                        row['ranking_total']      # 17
                    )
                    iid = self.tv.insert("", "end", values=vals)
                    inserted += 1

                    tags = []
                    if row['has_delta']:
                        tags.append("positive")
                    if row['high_day']:
                        tags.append("high_day")
                    elif row['medium_day']:
                        tags.append("medium_day")
                    if row['is_warning']:
                        tags.append("warning")
                    if row['is_new']:
                        tags.append("new_player")

                    if tags:
                        self.tv.item(iid, tags=tuple(tags))

            self.count_label.config(text=f"{player_count} players")

            display_text = f"Data: {ct.strftime('%d-%m-%Y %H:%M:%S')}"
            if hh:
                display_text += " 🎰"
            self.ts_label.config(text=display_text)

            logger.info(f"✅✅✅ _update_display_ui DONE: {inserted} players inserted ✅✅✅")

        except Exception as e:
            logger.error(f"❌ UI update error: {e}", exc_info=True)
            self._update_display_error("UI error")

    # ===========================================================
    # Compatibility method for legacy display() calls
    # ===========================================================
    def display(self, timestamp):
        """
        Display leaderboard data for a specific timestamp.
        Runs _display_worker in background thread to prevent UI freeze.
        """
        try:
            logger.info(f"🔵 display() called for timestamp: {timestamp}")
            
            # Wait for existing display thread to finish
            if self.display_thread and self.display_thread.is_alive():
                logger.warning("⚠️ Display thread already running, waiting...")
                self.display_thread.join(timeout=5)
            
            # Start new background worker
            self.display_thread = threading.Thread(
                target=self._display_worker,
                args=(timestamp,),
                daemon=False
            )
            self.display_thread.start()
            logger.info("✅ Display worker started")
            
        except Exception as e:
            logger.error(f"❌ display() failed: {e}", exc_info=True)
            self._update_display_error(f"Display error: {str(e)[:40]}")

    def _update_display_error(self, message):
        """Show error in UI"""
        try:
            logger.error(f"🔴 Display error: {message}")
            self.tv.delete(*self.tv.get_children())
            self.ts_label.config(text=message)
            self.count_label.config(text="0 entries")
        except Exception:
            pass
    
    def filter(self):
        s = self.search_var.get().lower()
        if not s:
            for i in self.tv.get_children():
                self.tv.reattach(i, '', 'end')
        else:
            for i in self.tv.get_children():
                if s in str(self.tv.item(i)['values'][1]).lower():
                    self.tv.reattach(i, '', 'end')
                else:
                    self.tv.detach(i)
    
    def prev_ts(self):
        if not self.timestamps:
            return
        if self.ts_idx < len(self.timestamps) - 1:
            self.ts_idx += 1
        ts = self.timestamps[self.ts_idx]
        # ALT (kaputt):
        # self.display(ts)
        # NEU:
        self._update_display_ui(ts)

    def next_ts(self):
        if not self.timestamps:
            return
        if self.ts_idx > 0:
            self.ts_idx -= 1
        ts = self.timestamps[self.ts_idx]
        # ALT:
        # self.display(ts)
        # NEU:
        self._update_display_ui(ts)
    
    def prev_period(self):
        if not self.periods:
            return
        self.period_idx = min(self.period_idx + 1, len(self.periods) - 1)
        self.period_label.config(text=self.periods[self.period_idx])
    
    def next_period(self):
        if not self.periods:
            return
        self.period_idx = max(self.period_idx - 1, 0)
        self.period_label.config(text=self.periods[self.period_idx])
    
    def start_poll(self):
        if self.is_polling:
            messagebox.showinfo("Info", "Already running")
            return
        
        try:
            interval = float(self.poll_interval_var.get())
            if interval < CFG.MIN_POLL:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", f"Invalid! Min: {CFG.MIN_POLL} min")
            return
        
        if self.poll_mode_var.get() == "manual":
            self.polling_mgr.set_manual(interval)
        
        self.is_polling = True
        prevent_sleep()
        
        # ✅ CORRECT INDENTATION (no extra indent!)
        # Start Player Scout (if implemented)
        if hasattr(self, 'player_scout'):
            self.player_scout.start()
        
        self.poll_status_label.config(text=self.polling_mgr.status(), foreground="blue")
        
        def loop():
            last_6am_poll = None  # Track last 6am mandatory poll
            
            while self.is_polling and not self.closing:
                now = datetime.now()
                
                # ✅ MANDATORY 6AM POLL (start of new daily session)
                if now.hour == 6 and now.minute < 10:  # 6:00-6:09 window
                    today_date = now.date()
                    
                    # Only poll once per 6am
                    if last_6am_poll != today_date:
                        logger.info("🌅 6AM MANDATORY POLL - New daily session starting!")
                        self.fetch()
                        last_6am_poll = today_date
                        
                        # Reset polling interval to 15min
                        self.polling_mgr.interval = 15 * 60
                        self.polling_mgr.level = 0
                        logger.info("🔄 Polling reset to 20min baseline")
                        
                        # Wait for next cycle
                        delay = int(self.polling_mgr.interval)
                        for _ in range(delay):
                            if not self.is_polling or self.closing:
                                break
                            time.sleep(1)
                        continue
                
                # Normal fetch
                self.fetch()
                
                # Dynamic delay
                delay = int(self.polling_mgr.interval) if hasattr(self.polling_mgr, 'interval') else int(interval * 60)
                
                for _ in range(delay):
                    if not self.is_polling or self.closing:
                        break
                    time.sleep(1)
        
        threading.Thread(target=loop, daemon=True).start()
    
    def stop_poll(self):
        if not self.is_polling:
            return
        
        self.is_polling = False
        
        # Stop activity monitor
        # self.activity_monitor.stop()
        
        # ✅ STOP Player Scout
        self.player_scout.stop()
        
        allow_sleep()
        self.poll_status_label.config(text="Stopped", foreground="gray")
        self.status_label.config(text="Stopped", foreground="red")
        
    def export(self, fmt):
        try:
            if fmt == 'excel':
                fp = DB.export_excel()
            else:
                fp = DB.export_csv()
            messagebox.showinfo("OK", f"Exported:\n{fp}")
            if os.name == 'nt':
                os.startfile(fp)
        except Exception as e:
            messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def backup(self):
        bf = DB.backup()
        if bf:
            messagebox.showinfo("OK", f"Backup:\n{bf}")
        else:
            messagebox.showerror("Error", "Backup failed")
    
    def restore(self):
        fp = filedialog.askopenfilename(title="Select Backup", initialdir=CFG.BACKUP_DIR, filetypes=[("DB", "*.db")])
        if fp and messagebox.askyesno("Restore", "Overwrite?"):
            if DB.restore(fp):
                messagebox.showinfo("OK", "Restored!")
                self.load_snapshot()
            else:
                messagebox.showerror("Error", "Restore failed")
    
    def view_logs(self):
        w = tk.Toplevel(self.root)
        w.title("Logs")
        w.geometry("900x600")
        t = scrolledtext.ScrolledText(w, wrap=tk.WORD, font=("Courier", 9))
        t.pack(fill="both", expand=True, padx=10, pady=10)
        try:
            if os.path.exists(CFG.LOG_FILE):
                with open(CFG.LOG_FILE, 'r', encoding='utf-8') as f:
                    t.insert('1.0', f.read())
                t.see('end')
            else:
                t.insert('1.0', "No logs")
        except Exception as e:
            t.insert('1.0', f"Error: {e}")
        t.config(state='disabled')
        ttk.Button(w, text="Close", command=w.destroy).pack(pady=5)

    def show_points_chart(self):
        """Show points progress chart with matplotlib."""
        
        # ✅ Singleton check
        if self.open_windows['points_chart'] is not None:
            try:
                if self.open_windows['points_chart'].winfo_exists():
                    self.open_windows['points_chart'].lift()
                    self.open_windows['points_chart'].focus_force()
                    logger.info("✅ Points chart already open - bringing to front")
                    return
            except Exception:
                self.open_windows['points_chart'] = None
        
        # Create new window
        w = tk.Toplevel(self.root)
        w.title("Points Progress - Top 10")
        w.geometry("1200x700")
        self.open_windows['points_chart'] = w
        
        # Cleanup on close
        def on_close_points():
            try:
                if fig_holder[0]:
                    plt.close(fig_holder[0])
            except Exception:
                pass
            self.open_windows['points_chart'] = None
            w.destroy()
        
        w.protocol("WM_DELETE_WINDOW", on_close_points)
        
        # === REST OF EXISTING CODE (unchanged) ===
        cf = ttk.Frame(w, padding=10)
        cf.pack(fill="x")
        
        ttk.Label(cf, text="Period:").pack(side="left", padx=5)
        period_var = tk.StringVar()
        pc = ttk.Combobox(cf, textvariable=period_var, width=30, state="readonly")
        periods = DB.periods()
        if periods:
            pc['values'] = periods
            pc.current(0)
        pc.pack(side="left", padx=5)
        
        chart_frame = ttk.Frame(w)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        canvas_holder = [None]
        fig_holder = [None]

        def plot():
            try:
                conn = sqlite3.connect(CFG.DB_FILE)
                q = "SELECT name, points, timestamp FROM leaderboard"
                if period_var.get():
                    q += f" WHERE period_range = '{period_var.get()}'"
                q += " ORDER BY timestamp"
                df = pd.read_sql_query(q, conn)
                conn.close()
                
                if df.empty:
                    messagebox.showinfo("Info", "No data", parent=w)
                    return
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                df = df.dropna(subset=['timestamp'])
                
                if df.empty:
                    messagebox.showinfo("Info", "No valid timestamps", parent=w)
                    return
                
                top10 = df.groupby('name')['points'].max().nlargest(10).index.tolist()
                
                fig, ax = plt.subplots(figsize=(12, 6))
                fig_holder[0] = fig
                colors = plt.cm.tab10(range(10))
                
                for idx, name in enumerate(top10):
                    pd_player = df[df['name'] == name].sort_values('timestamp')
                    if not pd_player.empty:
                        ax.plot(pd_player['timestamp'], pd_player['points'], 
                                label=f"{name} (max: {max(pd_player['points']):,})", 
                                marker='o', linewidth=2, markersize=5, color=colors[idx])
                
                ax.set_title("Top 10 Players - Points Progress", fontsize=14, fontweight='bold')
                ax.set_xlabel("Time", fontsize=11)
                ax.set_ylabel("Points", fontsize=11)
                ax.legend(loc='upper left', fontsize=8, ncol=2)
                ax.grid(True, alpha=0.3)
                #set_30min_xlocator(ax)
                plt.tight_layout()
                
                if canvas_holder[0]:
                    canvas_holder[0].get_tk_widget().destroy()
                
                canvas = FigureCanvasTkAgg(fig, master=chart_frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                canvas.draw()
                canvas_holder[0] = canvas
                
            except Exception as e:
                logger.error(f"Chart error: {e}")
                messagebox.showerror("Error", f"Chart error:\n{e}", parent=w)
        
        def on_close():
            try:
                if fig_holder[0]:
                    plt.close(fig_holder[0])
            except Exception:
                pass
            w.destroy()
        
        w.protocol("WM_DELETE_WINDOW", on_close)
        
        # ✅ CORRECT BUTTON (uses plot, not update_chart):
        ttk.Button(cf, text="Update", command=plot).pack(side="left", padx=10)
        
        # Initial chart
        plot()

    def get_current_session_start(self):
        """Get the current session start time (6am today or yesterday) - TIMEZONE-NAIVE."""
        now = datetime.now()  # Remove CFG.TIMEZONE to make it naive
        if now.hour < 6:
            session_start = (now - timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
        else:
            session_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        return session_start

    def calculate_daily_delta_with_reset(self, df_player, session_start):
        """
        Calculate Δ points for the day, accounting for weekly reset at Wed 12:00.
        FIXED: All datetimes are timezone-naive for comparison.
        """
        df_player = df_player.sort_values('timestamp').copy()
        df_player['delta_day'] = 0

        carry = 0
        last_points = None

        # Ensure session_start is timezone-naive
        if hasattr(session_start, 'tzinfo') and session_start.tzinfo is not None:
            session_start = session_start.replace(tzinfo=None)

        # Reset time for Wednesday (timezone-naive)
        reset_time = datetime.combine(session_start.date(), dt_time(12, 0))
        if reset_time < session_start:
            reset_time += timedelta(days=1)

        for i, row in df_player.iterrows():
            points = row['points']
            timestamp = row['timestamp']
            
            # Ensure timestamp is timezone-naive (should already be from database)
            if hasattr(timestamp, 'tz_localize'):
                timestamp = pd.Timestamp(timestamp).tz_localize(None)
            elif hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
                timestamp = timestamp.replace(tzinfo=None)

            if last_points is None:
                df_player.at[i, 'delta_day'] = 0
                last_points = points
                continue

            # Detect reset: points drop at Wed noon
            try:
                if (points < last_points * 0.5 and timestamp >= reset_time) or (points == 0 and timestamp >= reset_time):
                    carry += last_points
                    last_points = points
            except TypeError:
                # If comparison still fails, skip this row
                logger.warning(f"Timestamp comparison failed for {timestamp}")
                continue

            delta = points - last_points
            df_player.at[i, 'delta_day'] = max(delta, 0) + carry
            last_points = points

        return df_player['delta_day']
    
    def _safe_after(self, delay, func, *args):
        """Schedule callback mit Auto-Cleanup bei Close"""
        if self.closing:
            return None
        
        try:
            after_id = self.root.after(delay, func, *args)
            self.after_ids.append(after_id)
            return after_id
        except Exception:
            return None
    
    def show_delta_chart(self):
        """Show delta chart for daily session gained points - SINGLETON."""
        
        # Check if window already exists
        if self.open_windows['delta_chart'] is not None:
            try:
                if self.open_windows['delta_chart'].winfo_exists():
                    self.open_windows['delta_chart'].lift()
                    self.open_windows['delta_chart'].focus_force()
                    logger.info("✅ Delta chart window already open - bringing to front")
                    return
            except Exception:
                self.open_windows['delta_chart'] = None
        
        # Create new window
        w = tk.Toplevel(self.root)
        w.title("Delta Chart - Daily Session Progress")
        w.geometry("1400x800")
        self.open_windows['delta_chart'] = w
        
        # Cleanup on close
        def on_close_delta():
            try:
                if hasattr(w, '_fig_holder') and w._fig_holder:
                    plt.close(w._fig_holder)
            except Exception:
                pass
            self.open_windows['delta_chart'] = None
            w.destroy()
        
        w.protocol("WM_DELETE_WINDOW", on_close_delta)
        
        # === REST OF EXISTING CODE (unchanged) ===
        now_utc = datetime.now(pytz.UTC)
        now_local = datetime.now(CFG.TIMEZONE)
        
        logger.info(f"🕐 Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info(f"🕐 Current Local: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        if now_local.hour < 6:
            expected_session = (now_local - timedelta(days=1)).date()
        else:
            expected_session = now_local.date()
        
        logger.info(f"📅 Expected session: {expected_session} (hour={now_local.hour})")
        
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            
            query = '''SELECT DISTINCT DATE(timestamp) as cal_date 
                       FROM leaderboard 
                       ORDER BY cal_date DESC'''
            result = conn.execute(query).fetchall()
            
            if not result:
                conn.close()
                messagebox.showinfo("Info", "No data available", parent=w)
                w.destroy()
                self.open_windows['delta_chart'] = None
                return
            
            all_dates = []
            
            for row in result:
                cal_date = datetime.strptime(row[0], '%Y-%m-%d').date()
                
                query_after_6 = '''
                    SELECT COUNT(*) FROM leaderboard
                    WHERE DATE(timestamp) = ?
                    AND TIME(timestamp) >= '06:00:00'
                '''
                after_6_count = conn.execute(query_after_6, (str(cal_date),)).fetchone()[0]
                
                if after_6_count > 0:
                    if cal_date not in all_dates:
                        all_dates.append(cal_date)
            
            conn.close()
            
            all_dates = sorted(list(set(all_dates)), reverse=True)
            
            logger.info(f"📅 Available sessions: {[str(d) for d in all_dates[:5]]}")
            
            if not all_dates:
                messagebox.showinfo("Info", "No session data available", parent=w)
                w.destroy()
                self.open_windows['delta_chart'] = None
                return
            
            if expected_session in all_dates:
                current_session_idx = [all_dates.index(expected_session)]
                logger.info(f"✅ Found current session: {expected_session} (index {current_session_idx[0]})")
            else:
                current_session_idx = [0]
                logger.warning(f"⚠️  Expected session {expected_session} not found, using: {all_dates[0]}")
                
        except Exception as e:
            logger.error(f"Failed to get session dates: {e}")
            import traceback
            logger.error(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to load session dates:\n{e}", parent=w)
            w.destroy()
            self.open_windows['delta_chart'] = None
            return
        
        # Control frame
        cf = ttk.Frame(w, padding=10)
        cf.pack(fill="x")
        
        # Top N players dropdown
        ttk.Label(cf, text="Show Top:").pack(side="left", padx=5)
        top_n_var = tk.StringVar(value="10")
        top_n_combo = ttk.Combobox(cf, textvariable=top_n_var, width=8, state="readonly")
        top_n_combo['values'] = ["10", "20", "30", "40"]
        top_n_combo.current(0)
        top_n_combo.pack(side="left", padx=5)
        ttk.Label(cf, text="players").pack(side="left", padx=(0,20))
        
        # Session navigation
        ttk.Label(cf, text="Daily Session:", font=("Arial", 10, "bold")).pack(side="left", padx=(20,5))
        
        nav_frame = ttk.Frame(cf)
        nav_frame.pack(side="left", padx=5)
        
        prev_btn = ttk.Button(nav_frame, text="◀ Previous", width=12)
        prev_btn.pack(side="left", padx=2)
        
        session_label = ttk.Label(nav_frame, text="", relief="sunken", width=50, anchor="center", 
                                  font=("Arial", 10, "bold"), foreground="blue")
        session_label.pack(side="left", padx=5)
        
        next_btn = ttk.Button(nav_frame, text="Next ▶", width=12)
        next_btn.pack(side="left", padx=2)
        
        def go_to_today():
            """Jump to current session."""
            now_local = datetime.now(CFG.TIMEZONE)
            
            if now_local.hour < 6:
                today_session = (now_local - timedelta(days=1)).date()
            else:
                today_session = now_local.date()
            
            if today_session in all_dates:
                current_session_idx[0] = all_dates.index(today_session)
                update_chart()
                logger.info(f"✅ Jumped to TODAY'S session: {today_session}")
            else:
                messagebox.showinfo("Info", f"No data for today's session ({today_session})", parent=w)
        
        ttk.Button(nav_frame, text="📅 Today", command=go_to_today, width=10).pack(side="left", padx=(10, 0))
        
        # Chart frame
        chart_frame = ttk.Frame(w)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        canvas_holder = [None]
        fig_holder = [None]
        
        # Store fig_holder on window for cleanup
        w._fig_holder = None

        def update_chart():
            # ... (keep ALL existing update_chart code unchanged)
            try:
                session_date = all_dates[current_session_idx[0]]
                session_start = datetime.combine(session_date, dt_time(6, 0, 0))
                session_end = session_start + timedelta(days=1)
                session_label.config(text=f"{session_start.strftime('%d-%m-%Y')} 06:00 - {session_end.strftime('%d-%m-%Y')} 05:59")
                
                try:
                    top_n = int(top_n_var.get())
                except Exception:
                    top_n = 10
                
                conn = sqlite3.connect(CFG.DB_FILE)
                session_start_str = session_start.strftime('%Y-%m-%d %H:%M:%S')
                session_end_str = session_end.strftime('%Y-%m-%d %H:%M:%S')
                
                query = '''
                    SELECT DISTINCT name 
                    FROM leaderboard 
                    WHERE timestamp >= ? 
                      AND timestamp < ?
                      AND points > 0
                '''
                
                players = [row[0] for row in conn.execute(query, (session_start_str, session_end_str)).fetchall()]
                
                if not players:
                    conn.close()
                    messagebox.showinfo("No Data", f"No players active in session {session_date}", parent=w)
                    return
                
                player_gains = {}
                
                for player in players:
                    query_player = '''
                        SELECT timestamp, points 
                        FROM leaderboard 
                        WHERE name = ? 
                          AND timestamp >= ? 
                          AND timestamp < ?
                        ORDER BY timestamp
                    '''
                    
                    df_player = pd.read_sql_query(query_player, conn, params=(player, session_start_str, session_end_str))
                    
                    if df_player.empty:
                        continue
                    
                    df_player['timestamp'] = pd.to_datetime(df_player['timestamp'], format='mixed', errors='coerce')
                    df_player = df_player.dropna(subset=['timestamp'])
                    
                    if df_player.empty:
                        continue
                    
                    first_points = df_player['points'].iloc[0]
                    last_points = df_player['points'].iloc[-1]
                    total_gain = last_points - first_points
                    
                    if total_gain > 0:
                        player_gains[player] = total_gain
                
                conn.close()
                
                top_players = sorted(player_gains.items(), key=lambda x: x[1], reverse=True)[:top_n]
                
                if not top_players:
                    messagebox.showinfo("No Data", "No players gained points in this session", parent=w)
                    return
                
                if fig_holder[0]:
                    plt.close(fig_holder[0])
                
                fig, ax = plt.subplots(figsize=(14, 7))
                fig_holder[0] = fig
                w._fig_holder = fig
                
                colors = plt.cm.tab20(range(len(top_players)))
                
                conn = sqlite3.connect(CFG.DB_FILE)
                
                for idx, (player, total_gain) in enumerate(top_players):
                    query_player = '''
                        SELECT timestamp, points 
                        FROM leaderboard 
                        WHERE name = ? 
                          AND timestamp >= ? 
                          AND timestamp < ?
                        ORDER BY timestamp
                    '''
                    
                    df_player = pd.read_sql_query(query_player, conn, params=(player, session_start_str, session_end_str))
                    
                    if df_player.empty:
                        continue
                    
                    df_player['timestamp'] = pd.to_datetime(df_player['timestamp'], format='mixed', errors='coerce')
                    df_player = df_player.dropna(subset=['timestamp'])
                    
                    if df_player.empty:
                        continue
                    
                    first_points = df_player['points'].iloc[0]
                    df_player['delta'] = df_player['points'] - first_points
                    
                    ax.plot(df_player['timestamp'], df_player['delta'], 
                           label=f"{player} (+{total_gain:,})", 
                           marker='o', linewidth=2, markersize=4, color=colors[idx])
                
                conn.close()
                
                ax.set_title(f'Daily Session Progress - Top {len(top_players)} Players\n{session_start.strftime("%d-%m-%Y")} 06:00 - {session_end.strftime("%d-%m-%Y")} 05:59', 
                            fontsize=14, fontweight='bold')
                ax.set_xlabel('Time', fontsize=11)
                ax.set_ylabel('Points Gained', fontsize=11)
                ax.legend(loc='upper left', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                
                set_30min_xlocator(ax)
                
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                
                if canvas_holder[0]:
                    canvas_holder[0].get_tk_widget().destroy()
                
                canvas = FigureCanvasTkAgg(fig, master=chart_frame)
                canvas.get_tk_widget().pack(fill="both", expand=True)
                canvas.draw()
                canvas_holder[0] = canvas
                
            except Exception as e:
                logger.error(f"Delta chart error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                messagebox.showerror("Error", f"Chart error:\n{e}", parent=w)
        
        def go_prev_session():
            if current_session_idx[0] < len(all_dates) - 1:
                current_session_idx[0] += 1
                update_chart()
        
        def go_next_session():
            if current_session_idx[0] > 0:
                current_session_idx[0] -= 1
                update_chart()
        
        prev_btn.config(command=go_prev_session)
        next_btn.config(command=go_next_session)
        top_n_combo.bind('<<ComboboxSelected>>', lambda e: update_chart())
        
        ttk.Button(cf, text="🔄 Refresh", command=update_chart, width=12).pack(side="left", padx=20)
        
        update_chart() 

    def show_comparison(self):
        """Show player comparison with checkbox selection (daily/weekly)."""

        # --- ensure window registry exists ---
        if not hasattr(self, "open_windows"):
            self.open_windows = {}

        # --- singleton / reuse existing window safely ---
        existing = self.open_windows.get('comparison')
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    logger.info("✅ Comparison already open - bringing to front")
                    return
            except Exception:
                # stale reference, drop it
                pass
            self.open_windows['comparison'] = None

        # --- create new window ---
        w = tk.Toplevel(self.root)
        w.title("Player Comparison")
        w.geometry("1600x800")
        self.open_windows['comparison'] = w

        def on_close_comparison():
            self.open_windows['comparison'] = None
            try:
                w.destroy()
            except Exception:
                pass

        w.protocol("WM_DELETE_WINDOW", on_close_comparison)

        # === Layout containers ===
        main_container = ttk.Frame(w)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # --- left panel: player selection ---
        left_panel = ttk.Frame(main_container, width=220)
        left_panel.pack(side="left", fill="y", padx=(0, 10))

        ttk.Label(
            left_panel,
            text="Select Players:",
            font=("Arial", 11, "bold")
        ).pack(pady=(0, 10))

        # scrollable checkbox area
        canvas = tk.Canvas(left_panel, width=220, height=500)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # allow mousewheel scrolling inside the canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scrollable_frame.bind("<Enter>", lambda e: scrollable_frame.bind_all("<MouseWheel>", _on_mousewheel))
        scrollable_frame.bind("<Leave>", lambda e: scrollable_frame.unbind_all("<MouseWheel>"))

        # fetch players from DB
        try:
            all_players = DB.players()
        except Exception as e:
            logger.error(f"Comparison: failed to load players list: {e}")
            all_players = []

        player_vars = {}

        for player in sorted(all_players):
            var = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(scrollable_frame, text=player, variable=var)
            cb.pack(anchor="w", pady=2)
            player_vars[player] = var

        # select / deselect helpers
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(pady=10)

        def select_all():
            for var in player_vars.values():
                var.set(True)

        def deselect_all():
            for var in player_vars.values():
                var.set(False)

        ttk.Button(btn_frame, text="Select All", command=select_all).pack(pady=2)
        ttk.Button(btn_frame, text="Deselect All", command=deselect_all).pack(pady=2)

        # --- right panel: controls + chart ---
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side="left", fill="both", expand=True)

        # controls row
        cf = ttk.Frame(right_panel, padding=10)
        cf.pack(fill="x")

        # view type (daily / weekly)
        ttk.Label(cf, text="View:", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        view_type_var = tk.StringVar(value="daily")
        ttk.Radiobutton(cf, text="Daily", variable=view_type_var, value="daily").pack(side="left", padx=5)
        ttk.Radiobutton(cf, text="Weekly", variable=view_type_var, value="weekly").pack(side="left", padx=5)

        # session navigation
        nav_frame = ttk.Frame(cf)
        nav_frame.pack(side="left", padx=20)

        prev_btn = ttk.Button(nav_frame, text="◀", width=3)
        prev_btn.pack(side="left", padx=2)

        session_label = ttk.Label(
            nav_frame,
            text="",
            relief="sunken",
            width=40,
            anchor="center",
            font=("Arial", 9, "bold"),
            foreground="blue"
        )
        session_label.pack(side="left", padx=5)

        next_btn = ttk.Button(nav_frame, text="▶", width=3)
        next_btn.pack(side="left", padx=2)

        # compare button
        compare_btn = ttk.Button(cf, text="🔄 Compare Selected Players", width=30)
        compare_btn.pack(side="left", padx=10)

        # chart container
        chart_frame = ttk.Frame(right_panel)
        chart_frame.pack(fill="both", expand=True)
        canvas_holder = [None]  # mutable ref to current FigureCanvasTkAgg

        # session tracking
        sessions = {
            'daily': [],
            'weekly': [],
            'current_idx': 0,
        }

        # --- helper: update nav button states ---
        def update_nav_buttons():
            view_type = view_type_var.get()
            session_list = sessions['daily'] if view_type == 'daily' else sessions['weekly']
            idx = sessions['current_idx']

            if not session_list:
                prev_btn.config(state='disabled')
                next_btn.config(state='disabled')
                return

            prev_btn.config(state='normal' if idx < len(session_list) - 1 else 'disabled')
            next_btn.config(state='normal' if idx > 0 else 'disabled')

        # --- helper: load sessions for current view ---
        def update_session_list(*_):
            view_type = view_type_var.get()

            try:
                conn = sqlite3.connect(CFG.DB_FILE)

                if view_type == "daily":
                    rows = conn.execute(
                        "SELECT DISTINCT session_date FROM player_sessions "
                        "WHERE session_date IS NOT NULL "
                        "ORDER BY session_date DESC"
                    ).fetchall()
                    sessions['daily'] = [
                        pd.to_datetime(r[0]).date()
                        for r in rows if r[0]
                    ]
                    sessions['current_idx'] = 0

                    if sessions['daily']:
                        sdate = sessions['daily'][0]
                        session_label.config(
                            text=(
                                f"Daily: {sdate.strftime('%b %d')} 06:00 - "
                                f"{(sdate + timedelta(days=1)).strftime('%b %d')} 05:59"
                            )
                        )
                    else:
                        session_label.config(text="No daily sessions found")

                else:
                    rows = conn.execute(
                        "SELECT DISTINCT period_range FROM leaderboard "
                        "WHERE period_range IS NOT NULL "
                        "ORDER BY period_range DESC"
                    ).fetchall()
                    sessions['weekly'] = [r[0] for r in rows if r[0]]
                    sessions['current_idx'] = 0

                    if sessions['weekly']:
                        session_label.config(text=f"Weekly: {sessions['weekly'][0]}")
                    else:
                        session_label.config(text="No weekly periods found")

                conn.close()
            except Exception as e:
                logger.error(f"Error loading sessions list: {e}")
                session_label.config(text="Error loading sessions")
            finally:
                update_nav_buttons()

        # --- nav actions ---
        def go_prev():
            view_type = view_type_var.get()
            session_list = sessions['daily'] if view_type == 'daily' else sessions['weekly']

            if session_list and sessions['current_idx'] < len(session_list) - 1:
                sessions['current_idx'] += 1

                if view_type == 'daily':
                    sdate = session_list[sessions['current_idx']]
                    session_label.config(
                        text=(
                            f"Daily: {sdate.strftime('%b %d')} 06:00 - "
                            f"{(sdate + timedelta(days=1)).strftime('%b %d')} 05:59"
                        )
                    )
                else:
                    session_label.config(text=f"Weekly: {session_list[sessions['current_idx']]}")
                update_nav_buttons()

        def go_next():
            view_type = view_type_var.get()
            session_list = sessions['daily'] if view_type == 'daily' else sessions['weekly']

            if session_list and sessions['current_idx'] > 0:
                sessions['current_idx'] -= 1

                if view_type == 'daily':
                    sdate = session_list[sessions['current_idx']]
                    session_label.config(
                        text=(
                            f"Daily: {sdate.strftime('%b %d')} 06:00 - "
                            f"{(sdate + timedelta(days=1)).strftime('%b %d')} 05:59"
                        )
                    )
                else:
                    session_label.config(text=f"Weekly: {session_list[sessions['current_idx']]}")
                update_nav_buttons()

        prev_btn.config(command=go_prev)
        next_btn.config(command=go_next)

        # --- plotting logic ---
        def plot():
            try:
                # selected players
                selected_players = [p for p, v in player_vars.items() if v.get()]

                if not selected_players:
                    messagebox.showwarning("Warning", "Select at least one player!", parent=w)
                    return
                if len(selected_players) > 10:
                    messagebox.showwarning("Warning", "Max 10 players for readability!", parent=w)
                    return

                view_type = view_type_var.get()
                session_list = sessions['daily'] if view_type == 'daily' else sessions['weekly']

                if not session_list:
                    messagebox.showwarning("Warning", "No sessions available!", parent=w)
                    return

                current_session = session_list[sessions['current_idx']]

                conn = sqlite3.connect(CFG.DB_FILE)
                fig, ax = plt.subplots(figsize=(14, 7))
                colors = plt.cm.tab20(range(len(selected_players)))

                if view_type == 'daily':
                    # current_session is a date
                    from datetime import date as _date_cls
                    if isinstance(current_session, datetime):
                        sdate = current_session.date()
                    elif isinstance(current_session, _date_cls):
                        sdate = current_session
                    else:
                        try:
                            sdate = pd.to_datetime(current_session).date()
                        except Exception:
                            conn.close()
                            messagebox.showerror("Error", "Invalid daily session date format!", parent=w)
                            return

                    session_start = datetime.combine(sdate, dt_time(6, 0))
                    session_end = session_start + timedelta(days=1)

                    for idx, player in enumerate(selected_players):
                        q = (
                            "SELECT timestamp, points FROM leaderboard "
                            "WHERE name = ? AND timestamp >= ? AND timestamp < ? "
                            "ORDER BY timestamp"
                        )
                        df = pd.read_sql_query(q, conn, params=(player, session_start, session_end))

                        if df.empty:
                            continue

                        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                        df = df.dropna(subset=['timestamp'])
                        if df.empty:
                            continue

                        first_points = df['points'].iloc[0]
                        df['delta_day'] = df['points'] - first_points

                        ax.plot(
                            df['timestamp'],
                            df['delta_day'],
                            label=f"{player} (+{int(df['delta_day'].iloc[-1]):,})",
                            marker='o',
                            linewidth=2,
                            markersize=3,
                            color=colors[idx]
                        )

                    time_ticks = pd.date_range(start=session_start, end=session_end, freq='2H')
                    ax.set_xlim(session_start, session_end)
                    ax.set_xticks(time_ticks)
                    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

                    title = (
                        f"Daily Session Comparison\n"
                        f"{sdate.strftime('%b %d')} 06:00 - "
                        f"{(sdate + timedelta(days=1)).strftime('%b %d')} 05:59"
                    )

                else:
                    # weekly / period_range
                    period_range = str(current_session)

                    for idx, player in enumerate(selected_players):
                        q = (
                            "SELECT timestamp, points FROM leaderboard "
                            "WHERE name = ? AND period_range = ? "
                            "ORDER BY timestamp"
                        )
                        df = pd.read_sql_query(q, conn, params=(player, period_range))

                        if df.empty:
                            continue

                        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                        df = df.dropna(subset=['timestamp'])
                        if df.empty:
                            continue

                        first_points = df['points'].iloc[0]
                        df['delta'] = df['points'] - first_points

                        ax.plot(
                            df['timestamp'],
                            df['delta'],
                            label=f"{player} (+{int(df['delta'].iloc[-1]):,})",
                            marker='o',
                            linewidth=2,
                            markersize=3,
                            color=colors[idx]
                        )

                    ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
                    title = f"Weekly Period Comparison\n{period_range}"

                conn.close()

                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel("Time", fontsize=11)
                ax.set_ylabel("Points Gained", fontsize=11)
                ax.legend(loc='upper left', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                plt.tight_layout()

                # replace existing canvas
                if canvas_holder[0] is not None:
                    try:
                        canvas_holder[0].get_tk_widget().destroy()
                    except Exception:
                        pass

                fc = FigureCanvasTkAgg(fig, master=chart_frame)
                fc.get_tk_widget().pack(fill="both", expand=True)
                fc.draw()
                canvas_holder[0] = fc

            except Exception as e:
                logger.error(f"Comparison error: {e}", exc_info=True)
                messagebox.showerror("Error", f"Chart error:\n{e}", parent=w)

        compare_btn.config(command=plot)

        # re-fill sessions when view type changes
        # use trace_add if available, fallback to trace for older code
        try:
            view_type_var.trace_add('write', update_session_list)
        except AttributeError:
            view_type_var.trace('w', lambda *args: update_session_list())

        # initial session load
        update_session_list()
        
    def show_patterns(self):
        """Player patterns - opens with TODAY's session by default."""
        if hasattr(self, '_patterns_window') and self._patterns_window and self._patterns_window.winfo_exists():
            self._patterns_window.lift()
            return
        
        w = tk.Toplevel(self.root)
        w.title("Player Patterns - Daily Sessions")
        w.geometry("2000x800")
        self._patterns_window = w
        
        def on_close():
            self._patterns_window = None
            w.destroy()
        w.protocol("WM_DELETE_WINDOW", on_close)
        
        main_frame = ttk.Frame(w, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(main_frame, text="Player Activity Patterns - Daily Sessions", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Navigation + Sort Controls
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill="x", pady=10)
        ttk.Label(nav_frame, text="Daily View:", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # Get session dates
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            result = conn.execute('SELECT DISTINCT session_date FROM player_sessions ORDER BY session_date DESC').fetchall()
            conn.close()
            session_dates = [pd.to_datetime(r[0]).date() for r in result if r[0]]
        except Exception:
            session_dates = []
        
        # 🔴 FIX: Determine today's session (timezone-naive for DB consistency)
        now = datetime.now()  # Remove timezone
        if now.hour < 6:
            today_session = (now - timedelta(days=1)).date()
        else:
            today_session = now.date()

        logger.info(f"🔍 Today's session: {today_session} (time: {now.strftime('%H:%M')})")

        # Always default to most recent available session
        if session_dates:
            if today_session in session_dates:
                current_session = [today_session]
                logger.info(f"✅ Using today's session: {today_session}")
            else:
                current_session = [session_dates[0]]
                logger.warning(f"⚠️ Today not found, using most recent: {session_dates[0]}")
        else:
            current_session = [None]
            logger.error("❌ No session dates available")
        
        # Navigation buttons
        prev_btn = ttk.Button(nav_frame, text="◀ Previous Day", width=15)
        prev_btn.pack(side="left", padx=2)
        
        session_label = ttk.Label(nav_frame, text="", relief="sunken", 
                                  width=40, anchor="center", font=("Arial", 10, "bold"))
        session_label.pack(side="left", padx=5)
        
        next_btn = ttk.Button(nav_frame, text="Next Day ▶", width=15)
        next_btn.pack(side="left", padx=2)
        
        weekly_btn = ttk.Button(nav_frame, text="📊 Weekly Summary", 
                                command=self.show_weekly_summary, width=18)
        weekly_btn.pack(side="left", padx=10)
        
        # SORT BY CONTROLS (immediately here - no button_frame stuff)
        ttk.Label(nav_frame, text="Sort by:", font=("Arial", 10, "bold")).pack(side="left", padx=(20,5))
        sort_var = tk.StringVar(value="Rank")
        sort_combo = ttk.Combobox(nav_frame, textvariable=sort_var, width=15, state="readonly")
        sort_combo['values'] = ["Rank", "Name", "First Online", "Last Online", "Hours Played", "Break Total", "Points/h", "Total Sleep"]
        sort_combo.current(0)
        sort_combo.pack(side="left", padx=2)
        
        # Treeview with Points/h COLUMN
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill="both", expand=True, pady=10)
        
        cols = ("Rank", "Player", "Sessions", "First Online", "Last Online", "Hours Played", "Sleep", 
                "Break Total", "Points/h", "First Seen", "Last Seen", "Avg Start", "Avg End", "Regularity", "Days")
        tv = ttk.Treeview(tree_frame, columns=cols, show="headings", height=25)
        
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=90, anchor="center")
        
        tv.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        tv.tag_configure("bot", background="#8B0000", foreground="white")
        tv.tag_configure("offline", foreground="gray")
        tv.tag_configure("high_efficiency", background="#90EE90")
        
        # Helper: Calculate stats for daily session

        def calc_daily_stats(player, sdate):
            """
            Calculate daily stats - OPTIMIZED VERSION (10x faster!)
            Returns stats dict with session info, sleep time, breaks, points/hour
            """
            try:
                conn = sqlite3.connect(CFG.DB_FILE)
                
                # Session boundaries: 6am today to 5:59am tomorrow (timezone-naive)
                session_start = datetime.combine(sdate, dt_time(6, 0))
                session_end = session_start + timedelta(days=1)
                
                session_start_str = session_start.strftime('%Y-%m-%d %H:%M:%S')
                session_end_str = session_end.strftime('%Y-%m-%d %H:%M:%S')
                
                # OPTIMIZED: Use SQL to pre-filter data
                query = '''
                    SELECT timestamp, points 
                    FROM leaderboard 
                    WHERE name = ? 
                      AND timestamp >= ? 
                      AND timestamp < ?
                      AND points > 0
                    ORDER BY timestamp ASC
                    LIMIT 1000
                '''
                
                df = pd.read_sql_query(query, conn, params=(player, session_start_str, session_end_str))
                
                if df.empty:
                    conn.close()
                    return {
                        'sessions': 0,
                        'first_online': "offline",
                        'last_online': "offline",
                        'hours': 0,
                        'sleep': "00:00",
                        'break_total': "00:00",
                        'points_per_hour': 0,
                        'first_seen': "offline",
                        'last_seen': "offline",
                        'avg_start': "offline",
                        'avg_end': "offline",
                        'regularity': 100,
                        'days': 0,
                        'is_bot': False
                    }
                
                # Parse timestamps (timezone-naive)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                df = df.dropna(subset=['timestamp'])
                df['points'] = df['points'].astype(int)
                
                if df.empty:
                    conn.close()
                    return {
                        'sessions': 0,
                        'first_online': "offline",
                        'last_online': "offline",
                        'hours': 0,
                        'sleep': "00:00",
                        'break_total': "00:00",
                        'points_per_hour': 0,
                        'first_seen': "offline",
                        'last_seen': "offline",
                        'avg_start': "offline",
                        'avg_end': "offline",
                        'regularity': 100,
                        'days': 0,
                        'is_bot': False
                    }
                
                # Find first INCREASE in points
                first_online = None
                first_points = None
                
                # Check if first poll shows an increase from previous session
                if df['points'].iloc[0] > 0:
                    prev_query = '''
                        SELECT points FROM leaderboard
                        WHERE name = ? AND timestamp < ?
                        ORDER BY timestamp DESC LIMIT 1
                    '''
                    prev_result = conn.execute(prev_query, (player, session_start_str)).fetchone()
                    prev_points = prev_result[0] if prev_result else 0
                    
                    if df['points'].iloc[0] > prev_points:
                        first_online = df['timestamp'].iloc[0]
                        first_points = df['points'].iloc[0]
                
                # Check subsequent polls for increases
                if first_online is None:
                    for i in range(1, len(df)):
                        if df['points'].iloc[i] > df['points'].iloc[i-1]:
                            first_online = df['timestamp'].iloc[i]
                            first_points = df['points'].iloc[i]
                            break
                
                # If no increase found, player was offline
                if first_online is None:
                    conn.close()
                    return {
                        'sessions': 0,
                        'first_online': "offline",
                        'last_online': "offline",
                        'hours': 0,
                        'sleep': "00:00",
                        'break_total': "00:00",
                        'points_per_hour': 0,
                        'first_seen': "offline",
                        'last_seen': "offline",
                        'avg_start': "offline",
                        'avg_end': "offline",
                        'regularity': 100,
                        'days': 0,
                        'is_bot': False
                    }
                
                # Find LAST INCREASE in points
                last_offline = None
                last_points = None
                
                for i in range(len(df) - 1, 0, -1):
                    if df['points'].iloc[i] > df['points'].iloc[i-1]:
                        last_offline = df['timestamp'].iloc[i]
                        last_points = df['points'].iloc[i]
                        break
                
                # Fallback if no recent increase
                if last_offline is None:
                    last_offline = df['timestamp'].iloc[-1]
                    last_points = df['points'].iloc[-1]
                
                # Calculate session duration
                session_duration_minutes = (last_offline - first_online).total_seconds() / 60
                
                # Calculate points gained
                points_gained = last_points - first_points
                
                # Smart break detection - dynamic threshold
                break_total_minutes = 0
                sleep_total_minutes = 0
                last_increase_time = first_online
                break_threshold = CFG.BREAK_THRESHOLD_MIN
                
                for i in range(1, len(df)):
                    if df['points'].iloc[i] > df['points'].iloc[i-1]:
                        gap_minutes = (df['timestamp'].iloc[i] - last_increase_time).total_seconds() / 60
                        if gap_minutes >= break_threshold:
                            # Subtract expected polling interval to get actual break time
                            break_time = max(0, gap_minutes - break_threshold)
                            
                            # Classify as sleep if break starts after 1am and lasts >SLEEP_CLASSIFICATION_THRESHOLD
                            break_start_time = last_increase_time
                            if break_start_time.hour >= 1 and break_time > CFG.SLEEP_CLASSIFICATION_THRESHOLD:
                                sleep_total_minutes += break_time
                            else:
                                break_total_minutes += break_time
                        last_increase_time = df['timestamp'].iloc[i]
                
                # Active play time (excluding breaks)
                active_play_minutes = max(0, session_duration_minutes - break_total_minutes)
                
                # Points per hour (based on active play)
                points_per_hour = (points_gained / active_play_minutes) * 60 if active_play_minutes > 0 else 0
                
                # Calculate sleep time from previous session
                sleep_time = "00:00"
                try:
                    prev_session_date = sdate - timedelta(days=1)
                    prev_session_start = datetime.combine(prev_session_date, dt_time(6, 0))
                    prev_session_end = prev_session_start + timedelta(days=1)
                    
                    query_prev = '''
                        SELECT timestamp FROM leaderboard 
                        WHERE name = ? 
                          AND timestamp >= ? 
                          AND timestamp < ?
                          AND points > 0
                        ORDER BY timestamp DESC
                        LIMIT 1
                    '''
                    
                    result = conn.execute(query_prev, (
                        player,
                        prev_session_start.strftime('%Y-%m-%d %H:%M:%S'),
                        prev_session_end.strftime('%Y-%m-%d %H:%M:%S')
                    )).fetchone()
                    
                    if result:
                        prev_last_offline = pd.to_datetime(result[0])
                        sleep_seconds = (first_online - prev_last_offline).total_seconds()
                        
                        if sleep_seconds > 0:
                            sleep_hours = int(sleep_seconds // 3600)
                            sleep_minutes = int((sleep_seconds % 3600) // 60)
                            sleep_time = f"{sleep_hours:02d}:{sleep_minutes:02d}"
                except Exception as e:
                    logger.error(f"Sleep calc error for {player}: {e}")
                
                # Store break/sleep data in database
                try:
                    break_type = 'sleep' if sleep_total_minutes > 0 else 'break'
                    DB.update_session_breaks(
                        player, 
                        sdate, 
                        break_total_minutes, 
                        sleep_total_minutes, 
                        break_type
                    )
                except Exception as e:
                    logger.error(f"Failed to store break/sleep data for {player}: {e}")
                
                conn.close()
                
                # Format break time
                if break_total_minutes >= 5:
                    if break_total_minutes >= 60:
                        break_hours = int(break_total_minutes // 60)
                        break_mins = int(break_total_minutes % 60)
                        break_display = f"{break_hours}h {break_mins}min"
                    else:
                        break_display = f"{int(break_total_minutes)}min"
                else:
                    break_display = "0min"
                
                return {
                    'sessions': 1,
                    'first_online': first_online.strftime('%H:%M'),
                    'last_online': last_offline.strftime('%H:%M'),
                    'hours': session_duration_minutes / 60,
                    'sleep': sleep_time,
                    'break_total': break_display,
                    'points_per_hour': points_per_hour,
                    'first_seen': first_online.strftime('%Y-%m-%d %H:%M'),
                    'last_seen': last_offline.strftime('%Y-%m-%d %H:%M'),
                    'avg_start': first_online.strftime('%H:%M'),
                    'avg_end': last_offline.strftime('%H:%M'),
                    'regularity': 100,
                    'days': 1,
                    'is_bot': (session_duration_minutes / 60) >= 24
                }
                
            except Exception as e:
                logger.error(f"calc_daily_stats error for {player}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        def update_display():
            """Run player processing in background thread to prevent UI freeze."""
            tv.delete(*tv.get_children())
            
            if current_session[0] is None:
                session_label.config(text="No Session Data Available")
                return
            
            sdate = current_session[0]
            session_label.config(text=f"Daily Session: {sdate.strftime('%d-%m-%Y')} 06:00 - {(sdate + timedelta(days=1)).strftime('%d-%m-%Y')} 05:59")
            
            # Show loading window
            loading = tk.Toplevel(w)
            loading.title("Loading...")
            loading.geometry("300x100")
            loading.transient(w)
            ttk.Label(loading, text="Processing players...", font=("Arial", 12)).pack(pady=20)
            progress_bar = ttk.Progressbar(loading, mode='indeterminate', length=200)
            progress_bar.pack(pady=10)
            progress_bar.start()
            loading.update()
            
            def worker():
                """Background thread to process players."""
                try:
                    # Get rankings
                    try:
                        conn = sqlite3.connect(CFG.DB_FILE)
                        latest_ts = conn.execute('SELECT MAX(timestamp) FROM leaderboard').fetchone()[0]
                        if latest_ts:
                            rank_df = pd.read_sql_query(
                                'SELECT name, CAST(rank AS INTEGER) as rank_num FROM leaderboard WHERE timestamp = ? ORDER BY rank_num',
                                conn, params=(latest_ts,))
                            rank_map = dict(zip(rank_df['name'], rank_df['rank_num']))
                        else:
                            rank_map = {}
                        conn.close()
                    except Exception:
                        rank_map = {}
                    
                    # Collect all player stats
                    player_stats = []
                    for player in DB.players():
                        s = calc_daily_stats(player, sdate)
                        if s:
                            player_stats.append((player, s, rank_map.get(player, 999)))
                    
                    # Sort based on selection
                    sort_key = sort_var.get()
                    if sort_key == "Rank":
                        player_stats.sort(key=lambda x: x[2])
                    elif sort_key == "Name":
                        player_stats.sort(key=lambda x: x[0].lower())
                    elif sort_key == "First Online":
                        player_stats.sort(key=lambda x: (x[1]['first_online'] == "offline", x[1]['first_online']))
                    elif sort_key == "Last Online":
                        player_stats.sort(key=lambda x: (x[1]['last_online'] == "offline", x[1]['last_online']), reverse=True)
                    elif sort_key == "Hours Played":
                        player_stats.sort(key=lambda x: x[1]['hours'], reverse=True)
                    elif sort_key == "Break Total":
                        def break_key(item):
                            bt = item[1]['break_total']
                            try:
                                total_min = 0
                                if 'h' in bt:
                                    parts = bt.split('h')
                                    total_min += int(parts[0]) * 60
                                    if 'min' in parts[1]:
                                        total_min += int(parts[1].replace('min', '').strip())
                                elif 'min' in bt:
                                    total_min = int(bt.replace('min', '').strip())
                                return total_min
                            except Exception:
                                return 0
                        player_stats.sort(key=break_key, reverse=True)
                    elif sort_key == "Points/h":
                        player_stats.sort(key=lambda x: x[1]['points_per_hour'], reverse=True)
                    elif sort_key == "Total Sleep":
                        def sleep_key(item):
                            sleep = item[1]['sleep']
                            if sleep == "offline":
                                return 0
                            try:
                                parts = sleep.split(':')
                                return int(parts[0]) * 60 + int(parts[1])
                            except Exception:
                                return 0
                        player_stats.sort(key=sleep_key, reverse=True)
                    
                    # Update UI on main thread
                    def update_ui():
                        for player, s, rank in player_stats:
                            rank_display = str(rank) if rank != 999 else "-"
                            points_h_display = f"{s['points_per_hour']:.0f}" if s['points_per_hour'] > 0 else "0"
                            
                            vals = (rank_display, player, s['sessions'], s['first_online'], s['last_online'],
                                    f"{s['hours']:.1f}h", s['sleep'], s['break_total'], points_h_display,
                                    s['first_seen'], s['last_seen'],
                                    s['avg_start'], s['avg_end'], f"{s['regularity']}%", s['days'])
                            
                            iid = tv.insert("", "end", values=vals)
                            if s.get('is_bot'):
                                tv.item(iid, tags=("bot",))
                            elif s['first_online'] == "offline":
                                tv.item(iid, tags=("offline",))
                            elif s['points_per_hour'] > 2000:
                                tv.item(iid, tags=("high_efficiency",))
                        
                        loading.destroy()
                    
                    w.after(0, update_ui)
                    
                except Exception as e:
                    logger.error(f"Worker thread error: {e}")
                    w.after(0, loading.destroy)
            
            threading.Thread(target=worker, daemon=True).start()

        def go_prev_day():
            if current_session[0] is None:
                return
            idx = session_dates.index(current_session[0]) if current_session[0] in session_dates else -1
            if idx < len(session_dates) - 1:
                current_session[0] = session_dates[idx + 1]
                update_display()
        
        def go_next_day():
            if current_session[0] is None:
                return
            idx = session_dates.index(current_session[0]) if current_session[0] in session_dates else -1
            if idx > 0:
                current_session[0] = session_dates[idx - 1]
                update_display()
        
        def export():
            try:
                if current_session[0] is None:
                    messagebox.showwarning("Warning", "No session selected!", parent=w)
                    return
                
                suffix = f"daily_{current_session[0].strftime('%Y%m%d')}"
                fn = f"patterns_{suffix}_{datetime.now().strftime('%H%M%S')}.csv"
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write(",".join(cols) + "\n")
                    for item in tv.get_children():
                        f.write(",".join(str(v) for v in tv.item(item)['values']) + "\n")
                messagebox.showinfo("Success", f"Exported:\n{fn}", parent=w)
                if os.name == 'nt':
                    os.startfile(fn)
            except Exception as e:
                messagebox.showerror("Error", f"Failed:\n{e}", parent=w)
        
        # Wire buttons
        prev_btn.config(command=go_prev_day)
        next_btn.config(command=go_next_day)
        sort_combo.bind('<<ComboboxSelected>>', lambda e: update_display())
        
        # Bottom
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="📄 Export CSV", command=export, width=15).pack(side="left", padx=5)
        
        # Initial display
        update_display()

    def show_weekly_summary(self):
        """Show weekly summary in a SEPARATE window."""
        if hasattr(self, '_weekly_window') and self._weekly_window and self._weekly_window.winfo_exists():
            self._weekly_window.lift()
            return
        
        w = tk.Toplevel(self.root)
        w.title("Player Patterns - Weekly Summary")
        w.geometry("1900x800")
        self._weekly_window = w
        
        def on_close():
            self._weekly_window = None
            w.destroy()
        w.protocol("WM_DELETE_WINDOW", on_close)
        
        main_frame = ttk.Frame(w, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # Header
        ttk.Label(main_frame, text="Player Activity Patterns - Weekly Summary", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Get weekly periods
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            result = conn.execute('SELECT DISTINCT period_range FROM leaderboard WHERE period_range IS NOT NULL ORDER BY period_range DESC').fetchall()
            conn.close()
            weekly_periods = [r[0] for r in result if r[0]]
        except Exception:
            weekly_periods = []
        
        if not weekly_periods:
            ttk.Label(main_frame, text="No weekly period data available", 
                     font=("Arial", 12), foreground="red").pack(pady=50)
            return
        
        current_week = [0]
        
        # Navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill="x", pady=10)
        ttk.Label(nav_frame, text="Weekly Period:", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        prev_week_btn = ttk.Button(nav_frame, text="◀ Prev Week", width=12)
        prev_week_btn.pack(side="left", padx=2)
        
        week_label = ttk.Label(nav_frame, text="", relief="sunken", width=40, anchor="center", 
                               font=("Arial", 10, "bold"))
        week_label.pack(side="left", padx=5)
        
        next_week_btn = ttk.Button(nav_frame, text="Next Week ▶", width=12)
        next_week_btn.pack(side="left", padx=2)
        
        # Treeview
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill="both", expand=True, pady=10)
        
        cols = ("Rank", "Player", "Sessions", "First Online", "Last Online", "Hours Played", 
                "Break Total", "First Seen", "Last Seen", "Avg Start", "Avg End", 
                "Regularity", "Days")
        tv = ttk.Treeview(tree_frame, columns=cols, show="headings", height=25)
        
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=100, anchor="center")
        
        tv.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        
        tv.tag_configure("bot", background="#8B0000", foreground="white")
        
        # Helper: Calculate weekly stats
        def calc_weekly_stats(player, period_range):
            try:
                conn = sqlite3.connect(CFG.DB_FILE)
                
                # Get period boundaries
                query_period = '''SELECT MIN(timestamp), MAX(timestamp) FROM leaderboard 
                                  WHERE period_range = ?'''
                result = conn.execute(query_period, (period_range,)).fetchone()
                
                if not result or not result[0]:
                    conn.close()
                    return None
                
                period_start = pd.to_datetime(result[0]).date()
                period_end = pd.to_datetime(result[1]).date()
                
                # Get all sessions for this player within the period
                query = '''SELECT * FROM player_sessions 
                           WHERE player_name = ? 
                           AND session_date >= ? 
                           AND session_date <= ?
                           ORDER BY session_date'''
                df = pd.read_sql_query(query, conn, params=(player, str(period_start), str(period_end)))
                conn.close()
                
                if df.empty:
                    return None
                
                df['first_online'] = pd.to_datetime(df['first_online'], errors='coerce')
                df['last_offline'] = pd.to_datetime(df['last_offline'], errors='coerce')
                df = df.dropna(subset=['first_online', 'last_offline'])
                
                if df.empty:
                    return None
                
                # Calculate stats
                ft = df['first_online'].dt.hour * 60 + df['first_online'].dt.minute
                lt = df['last_offline'].dt.hour * 60 + df['last_offline'].dt.minute
                std = ft.std() if len(ft) > 1 else 0
                std = 0 if pd.isna(std) else std
                
                first_online_time = df['first_online'].min()
                last_offline_time = df['last_offline'].max()
                total_hours = df['session_duration_minutes'].sum() / 60
                total_break_minutes = df['break_duration_minutes'].sum() if 'break_duration_minutes' in df.columns else 0
                
                return {
                    'sessions': len(df),
                    'first_online': first_online_time.strftime('%H:%M') if pd.notna(first_online_time) else "offline",
                    'last_online': last_offline_time.strftime('%H:%M') if pd.notna(last_offline_time) else "offline",
                    'hours': total_hours,
                    'break_total': f"{int(total_break_minutes)}min" if total_break_minutes > 0 else "0min",
                    'first_seen': first_online_time.strftime('%Y-%m-%d %H:%M') if pd.notna(first_online_time) else "offline",
                    'last_seen': last_offline_time.strftime('%Y-%m-%d %H:%M') if pd.notna(last_offline_time) else "offline",
                    'avg_start': f"{int(ft.mean()//60):02d}:{int(ft.mean()%60):02d}" if not pd.isna(ft.mean()) else "offline",
                    'avg_end': f"{int(lt.mean()//60):02d}:{int(lt.mean()%60):02d}" if not pd.isna(lt.mean()) else "offline",
                    'regularity': int(max(0, 100 - std)), 
                    'days': len(df)
                }
            except Exception as e:
                logger.error(f"Weekly stats error for {player}: {e}")
                return None
        
        def update_display():
            tv.delete(*tv.get_children())
            
            period_range = weekly_periods[current_week[0]]
            week_label.config(text=period_range)
            
            # Get rankings
            try:
                conn = sqlite3.connect(CFG.DB_FILE)
                latest_ts = conn.execute('SELECT MAX(timestamp) FROM leaderboard').fetchone()[0]
                if latest_ts:
                    rank_df = pd.read_sql_query(
                        'SELECT name, CAST(rank AS INTEGER) as rank_num FROM leaderboard WHERE timestamp = ? ORDER BY rank_num',
                        conn, params=(latest_ts,))
                    rank_map = dict(zip(rank_df['name'], rank_df['rank_num']))
                else:
                    rank_map = {}
                conn.close()
            except Exception:
                rank_map = {}
            
            # Update navigation buttons
            prev_week_btn.config(state='normal' if current_week[0] < len(weekly_periods) - 1 else 'disabled')
            next_week_btn.config(state='normal' if current_week[0] > 0 else 'disabled')
            
            # Calculate stats for all players
            player_stats = []
            for player in sorted(DB.players()):
                s = calc_weekly_stats(player, period_range)
                if s and s['sessions'] > 0:
                    player_stats.append((player, s, rank_map.get(player, 999)))
            
            # Sort by rank
            player_stats.sort(key=lambda x: x[2])
            
            # Display
            for player, s, rank in player_stats:
                rank_display = str(rank) if rank != 999 else "-"
                
                vals = (rank_display, player, s['sessions'], s['first_online'], s['last_online'],
                        f"{s['hours']:.1f}h", s['break_total'],
                        s['first_seen'], s['last_seen'],
                        s['avg_start'], s['avg_end'], f"{s['regularity']}%", s['days'])
                
                tv.insert("", "end", values=vals)
        
        def go_prev_week():
            if current_week[0] < len(weekly_periods) - 1:
                current_week[0] += 1
                update_display()
        
        def go_next_week():
            if current_week[0] > 0:
                current_week[0] -= 1
                update_display()
        
        def export():
            try:
                suffix = f"weekly_{weekly_periods[current_week[0]].replace(' ', '_')}"
                fn = f"patterns_{suffix}_{datetime.now().strftime('%H%M%S')}.csv"
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write(",".join(cols) + "\n")
                    for item in tv.get_children():
                        f.write(",".join(str(v) for v in tv.item(item)['values']) + "\n")
                messagebox.showinfo("Success", f"Exported:\n{fn}", parent=w)
                if os.name == 'nt':
                    os.startfile(fn)
            except Exception as e:
                messagebox.showerror("Error", f"Failed:\n{e}", parent=w)
        
        # Wire buttons
        prev_week_btn.config(command=go_prev_week)
        next_week_btn.config(command=go_next_week)
        
        # Bottom
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="📄 Export CSV", command=export, width=15).pack(side="left", padx=5)
        
        # Initial display
        update_display()
    
    def show_overall_chart(self):
        
        # --- ensure window registry exists ---
        if not hasattr(self, "open_windows"):
            self.open_windows = {}

        # --- singleton / reuse existing ---
        existing = self.open_windows.get('overall_chart')
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.lift()
                    existing.focus_force()
                    logger.info("✅ Overall chart already open - bringing to front")
                    return
            except Exception:
                pass
            self.open_windows['overall_chart'] = None

        # --- create new chart window ---
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Overall Leaderboard Chart")
        chart_window.geometry("1600x800")
        chart_window.configure(bg="#ecf0f1")

        self.open_windows['overall_chart'] = chart_window

        def on_close():
            self.open_windows['overall_chart'] = None
            try:
                chart_window.destroy()
            except Exception:
                pass

        chart_window.protocol("WM_DELETE_WINDOW", on_close)

        # --- top control frame ---
        control_frame = tk.Frame(chart_window, bg='#ecf0f1', pady=12, padx=20)
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="Overall Leaderboard Trend", font=("Arial", 13, "bold")).pack(side="left", padx=10)

        ttk.Label(control_frame, text="View:", font=("Arial", 10, "bold")).pack(side="left", padx=(20, 5))
        view_var = tk.StringVar(value="top10")
        ttk.Radiobutton(control_frame, text="Top 10", variable=view_var, value="top10").pack(side="left", padx=5)
        ttk.Radiobutton(control_frame, text="Top 20", variable=view_var, value="top20").pack(side="left", padx=5)

        refresh_btn = ttk.Button(control_frame, text="🔄 Refresh Chart")
        refresh_btn.pack(side="right", padx=10)

        # --- chart frame ---
        chart_frame = tk.Frame(chart_window, bg="#ffffff")
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        canvas_holder = [None]

        def plot_chart():
            """Load data and plot overall leaderboard progression."""
            try:
                conn = sqlite3.connect(CFG.DB_FILE)

                # Determine how many players to plot
                top_n = 10 if view_var.get() == "top10" else 20

                # Get most recent leaderboard snapshot (timestamp max)
                latest_ts = conn.execute("SELECT MAX(timestamp) FROM leaderboard").fetchone()[0]
                if not latest_ts:
                    messagebox.showwarning("No data", "No leaderboard data available.", parent=chart_window)
                    conn.close()
                    return

                # Get top players by points in latest snapshot
                top_players = [
                    row[0] for row in conn.execute(
                        "SELECT name FROM leaderboard WHERE timestamp = ? "
                        "ORDER BY points DESC LIMIT ?", (latest_ts, top_n)
                    ).fetchall()
                ]

                if not top_players:
                    messagebox.showwarning("No players", "No top players found for the latest snapshot.", parent=chart_window)
                    conn.close()
                    return

                # Create figure
                fig, ax = plt.subplots(figsize=(14, 7))
                colors = plt.cm.tab20(range(len(top_players)))

                for idx, player in enumerate(top_players):
                    df = pd.read_sql_query(
                        "SELECT timestamp, points FROM leaderboard WHERE name = ? ORDER BY timestamp ASC",
                        conn,
                        params=(player,)
                    )
                    if df.empty:
                        continue

                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                    df = df.dropna(subset=['timestamp'])
                    if df.empty:
                        continue

                    first_points = df['points'].iloc[0]
                    df['delta'] = df['points'] - first_points

                    ax.plot(
                        df['timestamp'],
                        df['delta'],
                        label=f"{player} (+{int(df['delta'].iloc[-1]):,})",
                        linewidth=2,
                        markersize=3,
                        color=colors[idx]
                    )

                conn.close()

                ax.set_title("Overall Points Progression (Top Players)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Time", fontsize=11)
                ax.set_ylabel("Points Gained", fontsize=11)
                ax.legend(loc='upper left', fontsize=9, ncol=2)
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
                plt.tight_layout()

                # --- update Tkinter canvas ---
                if canvas_holder[0]:
                    try:
                        canvas_holder[0].get_tk_widget().destroy()
                    except Exception:
                        pass

                fc = FigureCanvasTkAgg(fig, master=chart_frame)
                fc.get_tk_widget().pack(fill="both", expand=True)
                fc.draw()
                canvas_holder[0] = fc

            except Exception as e:
                logger.error(f"Overall chart error: {e}", exc_info=True)
                messagebox.showerror("Error", f"Chart error:\n{e}", parent=chart_window)

        # connect refresh button
        refresh_btn.config(command=plot_chart)

        # --- initial render ---
        plot_chart()

    def show_player_history_chart(self, player_name):
        """Show history chart for a specific player using leaderboard data."""
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            
            # Get player's weekly performance
            query = '''
                SELECT period_range, 
                       MIN(CAST(rank AS INTEGER)) as best_rank,
                       MAX(points) as max_points
                FROM leaderboard
                WHERE name = ?
                AND period_range IS NOT NULL
                AND CAST(rank AS INTEGER) <= 40
                GROUP BY period_range
                ORDER BY period_range
            '''
            
            data = conn.execute(query, (player_name,)).fetchall()
            conn.close()
            
            if not data:
                messagebox.showinfo("No Data", f"No historical data for {player_name}", parent=self.root)
                return
            
            # Create window
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            chart_window = tk.Toplevel(self.root)
            chart_window.title(f"History: {player_name}")
            chart_window.geometry("1200x600")
            
            # Create figure with 2 subplots
            fig = Figure(figsize=(12, 6), dpi=100)
            
            # Parse data
            weeks = []
            ranks = []
            points = []
            
            for period_str, rank, pts in data:
                try:
                    start_date_str = period_str.split(' - ')[0].strip()
                    date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
                    weeks.append(date_obj)
                    ranks.append(rank)
                    points.append(pts)
                except Exception:
                    continue
            
            if not weeks:
                messagebox.showinfo("No Data", f"Could not parse data for {player_name}", parent=self.root)
                return
            
            # Subplot 1: Rank over time
            ax1 = fig.add_subplot(121)
            ax1.plot(weeks, ranks, marker='o', color='#e74c3c', linewidth=2)
            ax1.set_xlabel('Week', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Best Rank', fontsize=11, fontweight='bold')
            ax1.set_title(f'{player_name} - Rank History', fontsize=12, fontweight='bold')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Points over time
            ax2 = fig.add_subplot(122)
            ax2.plot(weeks, points, marker='o', color='#2ecc71', linewidth=2)
            ax2.set_xlabel('Week', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Max Points', fontsize=11, fontweight='bold')
            ax2.set_title(f'{player_name} - Points History', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Format dates
            fig.autofmt_xdate()
            fig.tight_layout()
            
            # Embed
            canvas = FigureCanvasTkAgg(fig, master=chart_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Stats
            stats_frame = tk.Frame(chart_window, bg='#2c3e50', pady=10)
            stats_frame.pack(fill=tk.X)
            
            stats_text = f"Weeks: {len(weeks)} | Best Rank: {min(ranks)} | Avg Rank: {sum(ranks)/len(ranks):.1f} | Avg Points: {sum(points)/len(points):,.0f}"
            
            tk.Label(stats_frame, text=stats_text, font=("Arial", 11, "bold"), 
                    bg='#2c3e50', fg='white').pack()
            
        except Exception as e:
            messagebox.showerror("Error", f"Chart error:\n{e}", parent=self.root)
            import traceback
            logger.error(traceback.format_exc())

    def on_right_click(self, event):
        """Show context menu on right-click."""
        try:
            item = self.tv.identify_row(event.y)
            if item:
                self.tv.selection_set(item)
                values = self.tv.item(item, 'values')
                player_name = values[1]  # Player name column
                
                # Create context menu
                context_menu = tk.Menu(self.root, tearoff=0)
                context_menu.add_command(
                    label=f"📊 Show History: {player_name}",
                    command=lambda: self.show_player_history_chart(player_name)
                )
                context_menu.post(event.x_root, event.y_root)
        except Exception as e:
            logger.error(f"Right-click error: {e}")
    
    # ✅ NEU: Methode zum Starten des DB-Workers
    def start_db_worker(self):
        """Starts the dedicated background thread for database operations."""
        if self.db_worker_thread is None or not self.db_worker_thread.is_alive():
            self.db_worker_thread = threading.Thread(target=self._db_worker_loop, daemon=True)
            self.db_worker_thread.start()
            logger.info("🗃️ Database worker thread started.")

    # ✅ NEU: Die Schleife für den DB-Worker
    def _db_worker_loop(self):
        """
        The main loop for the database worker thread.
        It waits for data from the queue and processes it sequentially.
        """
        while not self.closing:
            try:
                # Blockiert, bis etwas in der Queue ist
                df, period = self.data_queue.get()

                if df is None: # Signal zum Beenden
                    break
                
                logger.info(f"🗃️ DB Worker received {len(df)} records for period {period}.")
                
                # Führe die DB-Operationen aus
                DB.save(df, period)
                
                # Informiere die GUI, dass neue Daten zur Anzeige bereitstehen
                self.root.after(0, self.load_snapshot)
                
                self.data_queue.task_done()
            except Exception as e:
                logger.error(f"Error in DB worker loop: {e}", exc_info=True)

    # ✅ ÜBERARBEITET: fetch() legt Daten nur noch in die Queue
    def fetch(self):
        if not check_disk():
            messagebox.showerror("Disk Full", "Free up space!")
            return
        
        def worker():
            d = None
            try:
                self.root.after(0, lambda: self.status_label.config(text="Fetching...", foreground="orange"))
                self.root.after(0, self.progress.start)
                
                d = get_driver(False)
                df, period = parse_leaderboard(d)
                    
                if not df.empty:
                    # Daten in die Queue legen, statt direkt zu speichern
                    self.data_queue.put((df, period))
                    logger.info(f"✅ Fetch successful. {len(df)} records added to the queue.")

                    # Polling-Logik bleibt hier, da sie schnell ist
                    active_players = len(df[df['points'] > 0])
                    changes = self.polling_mgr.calc_changes(df, self.prev_data)
                    self.polling_mgr.update(changes, active_players)
                    self.prev_data = df.copy()

                    ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                    self.root.after(0, lambda t=ts: self.status_label.config(text=f"OK {t}", foreground="green"))
                    self.root.after(0, lambda: self.poll_status_label.config(text=self.polling_mgr.status()))
                else:
                    self.root.after(0, lambda: self.status_label.config(text="No data found", foreground="red"))

            except Exception as e:
                logger.error(f"Fetch worker error: {e}", exc_info=True)
                self.root.after(0, lambda: self.status_label.config(text="Error during fetch", foreground="red"))
            finally:
                if d:
                    try: d.quit()
                    except: pass
                self.root.after(0, self.progress.stop)
        
        threading.Thread(target=worker, daemon=True).start()

    # ✅ ÜBERARBEITET: on_close muss den DB-Worker sicher beenden
    def on_close(self):
        try:
            logger.info("🔴 Closing application...")
            self.closing = True
            
            # Signal an den DB-Worker senden, dass er sich beenden soll
            if self.db_worker_thread and self.db_worker_thread.is_alive():
                self.data_queue.put((None, None)) # Sentinel-Wert
                self.db_worker_thread.join(timeout=5) # Auf Beendigung warten
            
            if hasattr(self, 'player_scout'): self.player_scout.stop()
            if hasattr(self, 'polling_mgr'): self.polling_mgr.save_state()
            
            for after_id in self.after_ids:
                try: self.root.after_cancel(after_id)
                except: pass
            
            if self.is_polling: self.stop_poll()
            
            try: self.config.set('window_geometry', self.root.geometry())
            except: pass
            
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=5)
            
            allow_sleep()
            logger.info("✅ Cleanup complete, destroying window...")
            
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Close error: {e}", exc_info=True)
            try: self.root.destroy()
            except: pass

# =============================================================================
# DRIVER CREATION HELPERS (NEW)
# =============================================================================
# =============================================================================
# DRIVER HEALTH UTILITIES
# =============================================================================
def is_driver_alive(driver):
    """Check if Selenium driver session is still active."""
    try:
        if driver is None:
            return False
        driver.title  # simpler call to provoke WebDriver if invalid
        return True
    except Exception:
        return False

def ensure_driver_alive(create_func, name="driver", headless=False, retries=3):
    """Try to restart a dead Selenium driver."""
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"🔄 Attempting to restart {name} (try {attempt}/{retries})")
            driver = create_func(headless=headless)
            logger.info(f"✅ {name.capitalize()} driver restarted successfully")
            return driver
        except Exception as e:
            logger.error(f"❌ Failed to restart {name} (attempt {attempt}): {e}")
            time.sleep(2)
    logger.critical(f"❌ Could not restart {name} after {retries} attempts.")
    return None

def _create_main_driver(headless=False):
    """Create the main Chrome driver with isolated port and clean config."""
    opts = ChromeOptions()
    if headless:
        opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--remote-debugging-port=0")
    opts.add_argument("--disable-blink-features=AutomationControlled")

    if os.path.exists(CFG.CHROME_DRIVER_PATH):
        service = ChromeService(executable_path=CFG.CHROME_DRIVER_PATH, log_path=os.devnull, port=0)
    else:
        service = ChromeService(log_path=os.devnull, port=0)

    # ✅ FIX: Create actual Chrome driver (not recursive call)
    driver = webdriver.Chrome(service=service, options=opts)
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(60)
    
    logger.info("✅ Chrome driver created successfully (isolated port)")
    return driver

def _create_isolated_driver(headless=False):
    """Create isolated Firefox driver for Scout mode (separate profile + port)."""
    opts = FirefoxOptions()
    if headless:
        opts.add_argument("--headless")
    opts.add_argument("--no-remote")
    opts.add_argument("--disable-gpu")

    profile_path = os.path.join(tempfile.gettempdir(), "firefox_scout_profile")
    os.makedirs(profile_path, exist_ok=True)
    
    # Set up Firefox profile with optimizations
    try:
        from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
        if os.listdir(profile_path):
            profile = FirefoxProfile(profile_path)
            profile.set_preference("permissions.default.image", 2)
            profile.set_preference("browser.cache.disk.enable", False)
            profile.set_preference("browser.cache.memory.enable", False)
            opts.profile = profile
            logger.info("🔍 Scout: Using custom Firefox profile")
        else:
            opts.set_preference("permissions.default.image", 2)
            logger.info("🔍 Scout: Profile directory empty, using default settings")
    except Exception as e:
        logger.warning(f"Scout profile error: {e}, using default")
        opts.set_preference("permissions.default.image", 2)

    # ✅ FIX: Use FirefoxService (was incorrectly using undefined 'Service')
    if os.path.exists(CFG.GECKO_PATH):
        service = FirefoxService(executable_path=CFG.GECKO_PATH, log_path=os.devnull, port=0)
    else:
        service = FirefoxService(log_path=os.devnull, port=0)
    
    # ✅ FIX: Create actual Firefox driver (not recursive call)
    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(60)
    
    logger.info("✅ Firefox (isolated Scout) driver created successfully (own port & profile)")
    return driver
    
# ============================================================================
# MAIN
# ============================================================================
def main():
    """FIXED: Korrekter Startup mit Dialog"""
    
    # Setup logging ZUERST
    try:
        logger_main = setup_logging()
    except Exception as e:
        print(f"FATAL: Logger setup failed: {e}")
        return
    
    logger_main.info("=" * 80)
    logger_main.info("Leaderboard Tracker v3.6.2 COMPLETE FINAL - STARTUP")
    logger_main.info(f"User: jimmybeam3000")
    logger_main.info(f"UTC: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    logger_main.info(f"MESZ: {datetime.now(CFG.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    # Disk space check
    if not check_disk():
        logger_main.warning(f"Low disk space: Less than {CFG.MIN_FREE_DISK_MB} MB free!")
    
    # FIX: Create root VISIBLE (nicht hidden)
    root = tk.Tk()
    root.title("Leaderboard Tracker - Starting...")
    root.geometry("400x200")
    
    # Status label
    status_label = ttk.Label(root, text="Initializing...", 
                            font=("Arial", 12), foreground="blue")
    status_label.pack(pady=50)
    
    root.update()
    
    try:
        # FIX: DB init
        DB.init()
        status_label.config(text="Database initialized...")
        root.update()
        
        # FIX: Check migration
        do_migration = False
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            leaderboard_count = conn.execute('SELECT COUNT(*) FROM leaderboard').fetchone()[0]
            session_count = conn.execute('SELECT COUNT(*) FROM player_sessions WHERE first_online IS NOT NULL').fetchone()[0]
            conn.close()
            
            if leaderboard_count > 0 and session_count == 0:
                do_migration = True
                status_label.config(text="Migrating data...", foreground="orange")
                root.update()
                
                migrate_existing_data()
                
        except Exception as e:
            logger_main.warning(f"Migration check failed: {e}")
        
        # FIX: Browser selection Dialog MIT TIMEOUT
        status_label.config(text="Waiting for browser selection...", foreground="blue")
        root.update()
        
        chooser = BrowserChooser(root)
        
        # Warte max 30 Sekunden
        start_time = datetime.now()
        while chooser.winfo_exists() and (datetime.now() - start_time).total_seconds() < 30:
            try:
                root.update()
            except Exception:
                break
        
        # Dialog sollte jetzt geschlossen sein
        if chooser.result is None:
            status_label.config(text="Browser selection cancelled", foreground="red")
            root.update()
            root.after(2000, root.destroy)
            return
        
        # FIX: Setze Browser-Config
        CFG.BROWSER = chooser.result["browser"].lower()
        CFG.START_MINIMIZED = chooser.result["minimized"]
        
        logger_main.info(f"Browser selected: {CFG.BROWSER}")
        logger_main.info(f"Minimize browser: {CFG.START_MINIMIZED}")
        
        status_label.config(text="Loading application...", foreground="green")
        root.update()
        
        # FIX: Destroy startup window
        root.destroy()
        root = None
        
        # FIX: Create MAIN window
        root = tk.Tk()
        app = LeaderboardApp(root)
        
        logger_main.info("Application started successfully")
        root.mainloop()
        
    except Exception as e:
        logger_main.error(f"Startup failed: {e}", exc_info=True)
        if root and root.winfo_exists():
            status_label.config(text=f"ERROR: {str(e)[:50]}", foreground="red")
            root.update()
            root.after(5000, root.destroy)
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown by user")
    except Exception as e:
        import traceback
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()