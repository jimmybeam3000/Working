#!/usr/bin/env python3
# leaderboard_tracker_v3.6.1_migration.py
# FIXED: Timestamp conversion bug + Migration for existing data
# Version 3.6.1 FINAL WITH MIGRATION
# User: jimmybeam3000
# Date: 2025-10-18 11:50:32 UTC

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from datetime import datetime, time as dt_time, timedelta
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler
from math import ceil
import sqlite3, pandas as pd, numpy as np, random, time, threading, os, json, logging, shutil, pytz
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# ============================================================================
# PREVENT SLEEP MODE (Windows)
# ============================================================================
if os.name == 'nt':
    import ctypes
    
    # Constants for SetThreadExecutionState
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ES_AWAYMODE_REQUIRED = 0x00000040
    
    def prevent_sleep():
        """Prevent Windows from sleeping while polling."""
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            logger.info("Sleep prevention enabled (Windows)")
        except Exception as e:
            logger.warning(f"Could not prevent sleep: {e}")
    
    def allow_sleep():
        """Re-enable Windows sleep mode."""
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logger.info("Sleep prevention disabled")
        except:
            pass
    else:
        # Placeholder for Linux/Mac
        def prevent_sleep():
            logger.info("Sleep prevention not implemented for this OS")
        def allow_sleep():
            pass

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    URL = "https://pml.good-game-service.com/pm-leaderboard/group?groupId=1220&lang=en&timezone=UTC-8"
    DB_FILE, EXCEL_FILE, CSV_FILE = "leaderboard_data.db", "leaderboard_export.xlsx", "leaderboard_export.csv"
    BACKUP_DIR, CONFIG_FILE, LOG_FILE, PATTERNS_FILE = "backups", "config.json", "leaderboard_tracker.log", "player_patterns.json"
    GECKO_PATH = r"C:\Webdriver\bin\geckodriver.exe"
    
    MIN_FREE_DISK_MB = 500  # ← ADD THIS LINE
    
    QUIET_START, QUIET_END = dt_time(15, 0), dt_time(8, 0)
    QUIET_INTERVAL, ACTIVE_BASE, ACTIVE_JITTER = 30, 0.5, 5
    ACTIVITY_THRESHOLD, ESCALATION_INTERVAL = 10, 20
    MIN_POLL, DEESCALATION_TIME = 0.17, 60
    
    HH_START, HH_END, HH_INTERVAL = dt_time(23, 0), dt_time(0, 59), 60
    PERIOD_DAY, PERIOD_HOUR, PERIOD_MIN = 2, 12, 0
    TIMEZONE = pytz.timezone('Europe/Berlin')
    MAX_TABLES, MIN_3MAX, MIN_6MAX = 4, 8, 12
    POINTS = {'3max': {1:36, 2:24, 3:12, 4:0}, '6max': {1:72, 2:54, 3:36, 4:18}}

CFG = Config()

# ============================================================================
# LOGGING
# ============================================================================
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(CFG.LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console)
    return logger

logger = setup_logging()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
clean_points = lambda s: int("".join(c for c in str(s or '').replace(',','').split('.')[0] if c.isdigit())) if s else 0
def check_disk(p="C:\\"):
    d = shutil.disk_usage(p)
    free_mb = d[2] / (1024 * 1024)
    print(f"DEBUG: Free space: {free_mb:.2f} MB")
    return free_mb >= CFG.MIN_FREE_DISK_MB
    is_hh = lambda dt=None: (t := (dt or datetime.now()).time()) >= CFG.HH_START or t <= CFG.HH_END

def get_period_start(ref=None):
    ref = ref or datetime.now(CFG.TIMEZONE)
    ref = CFG.TIMEZONE.localize(ref) if ref.tzinfo is None else ref
    wd, t = ref.weekday(), ref.time()
    days_back = 7 if wd == CFG.PERIOD_DAY and t < dt_time(CFG.PERIOD_HOUR, CFG.PERIOD_MIN) else (wd - CFG.PERIOD_DAY) % 7
    ps = ref.replace(hour=CFG.PERIOD_HOUR, minute=CFG.PERIOD_MIN, second=0, microsecond=0) if days_back == 0 and wd == CFG.PERIOD_DAY and t >= dt_time(CFG.PERIOD_HOUR, CFG.PERIOD_MIN) else (ref - timedelta(days=days_back)).replace(hour=CFG.PERIOD_HOUR, minute=CFG.PERIOD_MIN, second=0, microsecond=0)
    return ps

    get_period_elapsed = lambda ct=None: ((ct or datetime.now(CFG.TIMEZONE)) - get_period_start(ct)).total_seconds() / 60

def get_period_info(ct=None):
    ct = ct or datetime.now(CFG.TIMEZONE)
    ps, em = get_period_start(ct), get_period_elapsed(ct)
    h, m, d = int(em // 60), int(em % 60), int(em // 1440)
    return {'start': ps, 'end': ps + timedelta(days=7, seconds=-1), 'elapsed_minutes': em, 
            'elapsed_formatted': f"{d}d {h%24}h {m}min" if d > 0 else f"{h}h {m}min", 'current': ct}

    calc_min_games = lambda pts, hh=False: max(1, pts // (72*(2 if hh else 1)) + ceil((pts % (72*(2 if hh else 1))) / (36*(2 if hh else 1)))) if pts > 0 else 0
    calc_min_sessions = lambda games, tables=CFG.MAX_TABLES: ceil(games / tables) if games > 0 else 0

def validate_period(pts, t3, t6, tables=CFG.MAX_TABLES, hh=False, ct=None, player_name=None):
    """
    Validate period with BREAKS and SLEEP subtracted.
    """
    if pts <= 0: 
        return True, "N/A", 0, 0, 0
    
    em = get_period_elapsed(ct)
    games = calc_min_games(pts, hh)
    sess = calc_min_sessions(games, tables)
    needed = sess * t6
    pi = get_period_info(ct)
    
    # CRITICAL: Subtract breaks and sleep from available time
    if player_name:
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            
            # Get period boundaries
            period_start = pi['start'].date()
            period_end = pi['end'].date()
            
            # Get all sessions in this period for this player
            query = '''SELECT session_duration_minutes, break_duration_minutes, last_offline
                       FROM player_sessions
                       WHERE player_name = ?
                       AND session_date >= ?
                       AND session_date <= ?'''
            df = pd.read_sql_query(query, conn, params=(player_name, str(period_start), str(period_end)))
            conn.close()
            
            if not df.empty:
                # Total breaks across all sessions
                total_breaks = df['break_duration_minutes'].sum()
                
                # Calculate total sleep time (time between sessions)
                total_sleep = 0
                df['last_offline'] = pd.to_datetime(df['last_offline'], errors='coerce')
                df = df.dropna(subset=['last_offline'])
                df = df.sort_values('last_offline')
                
                for i in range(len(df) - 1):
                    # Gap between this session's end and next session's start
                    # (This is handled by looking at gaps in activity)
                    pass
                
                # For now, we'll calculate sleep as: Period time - Total session time
                total_session_time = df['session_duration_minutes'].sum()
                total_sleep = max(0, em - total_session_time - total_breaks)
                
                # Subtract breaks from available time
                available_time = em - total_breaks
            else:
                available_time = em
        except Exception as e:
            logger.error(f"Period validation error: {e}")
            available_time = em
    else:
        available_time = em
    
    is_valid = available_time >= needed
    
    if is_valid:
        status = f"OK Valid (period: {pi['elapsed_formatted']}, available: {available_time:.0f}min, needs {needed:.0f}min)"
    else:
        status = f"!! Impossible! (period: {pi['elapsed_formatted']}, available: {available_time:.0f}min, needs {needed:.0f}min)"
    
    return is_valid, status, sess, needed, available_time

def validate_session(games, dm, t3, t6, tables=CFG.MAX_TABLES, hh=False, game_type="mixed"):
    """
    Validate if session duration is physically possible.
    FIXED: Now considers multi-tabling (games / tables = sessions)
    """
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
    """
    Calculate point combinations with exhaustive search.
    Returns 'IMPOSSIBLE' if no exact combination exists.
    """
    if delta <= 0:
        return ""
    
    mult = 2 if hh else 1
    rem = abs(int(delta))
    pts_labels = [(72,"6max-P1"), (54,"6max-P2"), (36,"6max-P3"), (36,"3max-P1"), 
                  (24,"3max-P2"), (18,"6max-P4"), (12,"3max-P3")]
    
    def find_combination(remaining, idx, path):
        """Recursive backtracking to find exact combination."""
        if remaining == 0:
            return path
        if remaining < 0 or idx >= len(pts_labels):
            return None
        
        pts, lbl = pts_labels[idx]
        max_count = remaining // (pts * mult)
        
        # Try from max down to 0
        for count in range(max_count, -1, -1):
            new_path = path + ([f"{count}x{lbl}"] if count > 0 else [])
            result = find_combination(remaining - count * pts * mult, idx + 1, new_path)
            if result is not None:
                return result
        return None
    
    result = find_combination(rem, 0, [])
    
    if result is None:
        return "⚠ IMPOSSIBLE"
    
    if not result or all(c.startswith("0x") for c in result):
        return "⚠ IMPOSSIBLE"
    
    games = sum(int(x.split('x')[0]) for x in result if 'x' in x and not x.startswith('0x'))
    return f"[{games} games] {', '.join([c for c in result if not c.startswith('0x')])}"

def calc_3max_only_combo(delta, hh=False):
        """Calculate minimum 3-max game combinations. OPTIMIZED."""
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
    """Calculate minimum 6-max game combinations. OPTIMIZED."""
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

# ============================================================================
# INTELLIGENT POLLING
# ============================================================================
class IntelligentPolling:
    def __init__(self):
        self.activity, self.changes = deque(maxlen=100), deque(maxlen=10)
        self.interval, self.level, self.consecutive = CFG.QUIET_INTERVAL * 60, 0, 0
        self.last_activity = None
        self.manual_mode = False
        self.manual_interval = 30
    
    def is_quiet(self, dt=None):
        t = (dt or datetime.now()).time()
        return t >= CFG.QUIET_START or t < CFG.QUIET_END if CFG.QUIET_START > CFG.QUIET_END else CFG.QUIET_START <= t < CFG.QUIET_END
    
    def is_happy_hour(self, dt=None):
        """Check if it's Happy Hour (23:00-00:59)."""
        return is_hh(dt)
    
    def calc_changes(self, curr, prev):
        if prev is None or prev.empty: return 0
        return sum(1 for _, r in curr.iterrows() if not prev[prev['name']==r['name']].empty and r['points'] != prev[prev['name']==r['name']].iloc[0]['points'])
    
    def set_manual(self, minutes):
        self.manual_mode = True
        self.manual_interval = max(CFG.MIN_POLL, minutes * 60)
        self.interval = self.manual_interval
        logger.info(f"Manual mode: {minutes} min")
    
    def set_intelligent(self):
        self.manual_mode = False
        logger.info("Intelligent mode activated")
    
    def update(self, changes):
        if self.manual_mode:
            return self.manual_interval
        
        self.changes.append(changes)
        self.last_activity = datetime.now()
        avg = sum(self.changes) / len(self.changes) if self.changes else 0
        logger.info(f"Activity: {changes} changes, avg: {avg:.1f}, level: {self.level}")
        
        # PRIORITY 1: Happy Hour - always 1 minute
        if self.is_happy_hour():
            self.interval, self.level = 60, 4  # 60 seconds = 1 minute, level 4 = HH mode
            logger.info("🎰 HAPPY HOUR MODE -> 1min polling")
            return self.interval
        
        # PRIORITY 2: Quiet hours
        if self.is_quiet():
            self.interval, self.level = CFG.QUIET_INTERVAL * 60, 0
            return self.interval
        
        # PRIORITY 3: Normal activity-based polling
        if changes >= CFG.ACTIVITY_THRESHOLD:
            self.consecutive += 1
            if self.level == 0:
                self.level, self.interval = 1, CFG.ESCALATION_INTERVAL * 60
                logger.info(f"Activity spike! -> {CFG.ESCALATION_INTERVAL}min")
            elif self.consecutive >= 2:
                if self.level == 1 and avg >= CFG.ACTIVITY_THRESHOLD:
                    self.level, self.interval = 2, self.interval * 0.5
                    logger.info(f"Sustained! -> {self.interval/60:.1f}min")
                elif self.level == 2 and avg >= CFG.ACTIVITY_THRESHOLD * 1.5:
                    self.interval = max(CFG.MIN_POLL * 60, self.interval * 0.5)
                    logger.info(f"PEAK! -> {self.interval/60:.1f}min")
        else:
            if self.last_activity and (datetime.now() - self.last_activity).total_seconds()/60 > CFG.DEESCALATION_TIME:
                if self.level > 0:
                    self.level -= 1
                    self.interval = min(CFG.ESCALATION_INTERVAL * 60, self.interval * 2)
                    logger.info(f"Declined -> {self.interval/60:.1f}min")
            self.consecutive = 0
        
        if not self.is_quiet() and self.level == 0:
            self.interval = max(CFG.ACTIVE_BASE * 60 + random.randint(-CFG.ACTIVE_JITTER, CFG.ACTIVE_JITTER) * 60, CFG.MIN_POLL * 60)
        
        return self.interval
    
    def status(self):
        if self.manual_mode:
            return f"Manual Mode ({self.manual_interval/60:.1f}min)"
        if self.is_happy_hour():
            return f"🎰 HAPPY HOUR (polling every 1min)"
        if self.is_quiet(): 
            return f"Quiet Hours (polling every {CFG.QUIET_INTERVAL}min)"
        level = ["Normal", "Active", "Very Active", "PEAK"][min(self.level, 3)]
        return f"{level} (polling every {self.interval/60:.1f}min)"

# ============================================================================
# PATTERN DETECTOR (BOT DETECTION)
# ============================================================================
class PatternDetector:
    def __init__(self):
        self.patterns = self.load()
        self.sessions = defaultdict(list)
    
    def load(self):
        try:
            return json.load(open(CFG.PATTERNS_FILE)) if os.path.exists(CFG.PATTERNS_FILE) else {}
        except: 
            return {}
    
    def save(self):
        try:
            json.dump(self.patterns, open(CFG.PATTERNS_FILE, 'w'), indent=4)
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
    
    def calculate_bot_score(self, player):
        return 0, "LOW", []
    
    def stats(self, player):
        if player not in self.patterns:
            return None
        d = self.patterns[player]
        if not d['starts']:
            return {'sessions': 0, 'first_online': None, 'last_offline': None, 'regularity': 0, 'is_bot': False, 'bot_score': 0, 'bot_confidence': 'LOW', 'bot_warnings': [], 'total_tracked_days': 0}
        start_times = [s['time'] for s in d['starts']]
        earliest_start = min(start_times)
        if d.get('ends'):
            end_times = [e['time'] for e in d['ends']]
            latest_end = max(end_times)
        else:
            latest_end = None
        reg = max(0, 100 - (np.std(start_times) * 2)) if len(start_times) >= 3 else 0
        bot_score, bot_confidence, bot_warnings = self.calculate_bot_score(player)
        return {
            'sessions': d['sessions'],
            'first_online': f"{int(earliest_start//60):02d}:{int(earliest_start%60):02d}",
            'last_offline': f"{int(latest_end//60):02d}:{int(latest_end%60):02d}" if latest_end else "N/A",
            'regularity': int(reg),
            'total_tracked_days': len(set(s['date'] for s in d['starts'])),
            'is_bot': bot_score >= 50,
            'bot_score': bot_score,
            'bot_confidence': bot_confidence,
            'bot_warnings': bot_warnings
        }

# ============================================================================
# SELF-LEARNING BOT DETECTOR
# ============================================================================
class SelfLearningBotDetector:
    """
    Machine learning bot detector that learns from player behavior patterns.
    Uses anomaly detection and clustering to identify suspicious patterns.
    """
    
    def __init__(self):
        self.player_profiles = {}  # player -> feature vector
        self.normal_baseline = None  # "normal" player baseline
        self.detection_history = defaultdict(list)  # player -> [(timestamp, score)]
        self.learning_data_file = "bot_learning_data.json"
        self.load_learning_data()
    
    def load_learning_data(self):
        """Load previously learned patterns."""
        try:
            if os.path.exists(self.learning_data_file):
                with open(self.learning_data_file, 'r') as f:
                    data = json.load(f)
                    self.player_profiles = data.get('profiles', {})
                    self.normal_baseline = data.get('baseline', None)
                logger.info(f"Loaded learning data: {len(self.player_profiles)} profiles")
        except Exception as e:
            logger.error(f"Failed to load learning data: {e}")
    
    def save_learning_data(self):
        """Save learned patterns for future use."""
        try:
            data = {
                'profiles': self.player_profiles,
                'baseline': self.normal_baseline,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.learning_data_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved learning data")
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def extract_features(self, player_name, days=14):
        """
        Extract behavioral features for machine learning.
        Returns feature vector with 15+ dimensions.
        """
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            
            cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Feature 1-3: Session timing patterns
            query_sessions = '''
                SELECT first_online, last_offline, session_duration_minutes
                FROM player_sessions
                WHERE player_name = ?
                AND first_online >= ?
                ORDER BY session_date DESC
            '''
            df_sessions = pd.read_sql_query(query_sessions, conn, params=(player_name, cutoff))
            
            if df_sessions.empty:
                conn.close()
                return None
            
            # Convert to datetime
            df_sessions['first_online'] = pd.to_datetime(df_sessions['first_online'])
            df_sessions['last_offline'] = pd.to_datetime(df_sessions['last_offline'])
            
            # Extract hour of day
            start_hours = df_sessions['first_online'].dt.hour
            
            features = {}
            
            # F1: Start time regularity (low std = bot)
            features['start_time_std'] = start_hours.std() if len(start_hours) > 1 else 24
            
            # F2: Start time entropy (high entropy = human randomness)
            hour_counts = start_hours.value_counts()
            hour_probs = hour_counts / hour_counts.sum()
            features['start_time_entropy'] = -sum(hour_probs * np.log2(hour_probs + 1e-10))
            
            # F3: Session duration consistency (low variance = bot)
            features['session_duration_cv'] = (
                df_sessions['session_duration_minutes'].std() / 
                df_sessions['session_duration_minutes'].mean()
            ) if df_sessions['session_duration_minutes'].mean() > 0 else 0
            
            # Feature 4-6: Point gain patterns
            query_points = '''
                SELECT timestamp, points
                FROM leaderboard
                WHERE name = ?
                AND timestamp >= ?
                ORDER BY timestamp
            '''
            df_points = pd.read_sql_query(query_points, conn, params=(player_name, cutoff))
            
            if not df_points.empty:
                df_points['timestamp'] = pd.to_datetime(df_points['timestamp'])
                df_points = df_points.sort_values('timestamp')
                
                # Calculate point deltas
                df_points['delta'] = df_points['points'].diff()
                df_points = df_points[df_points['delta'] > 0]  # Only increases
                
                if len(df_points) > 2:
                    # F4: Point gain regularity (low CV = bot)
                    features['point_gain_cv'] = (
                        df_points['delta'].std() / df_points['delta'].mean()
                    ) if df_points['delta'].mean() > 0 else 0
                    
                    # F5: Time between gains regularity
                    time_diffs = df_points['timestamp'].diff().dt.total_seconds() / 60
                    time_diffs = time_diffs[time_diffs > 0]
                    features['time_between_gains_cv'] = (
                        time_diffs.std() / time_diffs.mean()
                    ) if len(time_diffs) > 1 and time_diffs.mean() > 0 else 0
                    
                    # F6: Point gain predictability (autocorrelation)
                    if len(df_points['delta']) > 10:
                        autocorr = df_points['delta'].autocorr()
                        features['point_gain_autocorr'] = autocorr if not pd.isna(autocorr) else 0
                    else:
                        features['point_gain_autocorr'] = 0
                else:
                    features['point_gain_cv'] = 0
                    features['time_between_gains_cv'] = 0
                    features['point_gain_autocorr'] = 0
            else:
                features['point_gain_cv'] = 0
                features['time_between_gains_cv'] = 0
                features['point_gain_autocorr'] = 0
            
            # Feature 7-9: Gap management (if rank 1)
            query_gaps = '''
                SELECT l1.timestamp, l1.points,
                       (SELECT points FROM leaderboard l2 
                        WHERE l2.timestamp = l1.timestamp 
                        AND CAST(l2.rank AS INTEGER) = 2
                        LIMIT 1) as second_place_points
                FROM leaderboard l1
                WHERE l1.name = ?
                AND l1.timestamp >= ?
                AND CAST(l1.rank AS INTEGER) = 1
                ORDER BY l1.timestamp
            '''
            df_gaps = pd.read_sql_query(query_gaps, conn, params=(player_name, cutoff))
            
            if not df_gaps.empty and len(df_gaps) > 5:
                df_gaps['gap'] = df_gaps['points'] - df_gaps['second_place_points']
                df_gaps = df_gaps.dropna(subset=['gap'])
                
                if not df_gaps.empty:
                    # F7: Gap consistency (low CV = bot managing lead)
                    features['gap_cv'] = (
                        df_gaps['gap'].std() / df_gaps['gap'].mean()
                    ) if df_gaps['gap'].mean() > 0 else 999
                    
                    # F8: Gap range (narrow range = bot)
                    gap_range = df_gaps['gap'].max() - df_gaps['gap'].min()
                    features['gap_range_ratio'] = (
                        gap_range / df_gaps['gap'].mean()
                    ) if df_gaps['gap'].mean() > 0 else 999
                    
                    # F9: Gap mean (is there a target gap?)
                    features['gap_mean'] = df_gaps['gap'].mean()
                else:
                    features['gap_cv'] = 999
                    features['gap_range_ratio'] = 999
                    features['gap_mean'] = 0
            else:
                features['gap_cv'] = 999
                features['gap_range_ratio'] = 999
                features['gap_mean'] = 0
            
            # Feature 10-12: Daily consistency
            query_daily = '''
                SELECT session_date, total_points_gained, session_duration_minutes
                FROM player_sessions
                WHERE player_name = ?
                AND session_date >= date(?, '-{} days')
                ORDER BY session_date DESC
            '''.format(days)
            df_daily = pd.read_sql_query(query_daily, conn, params=(player_name, datetime.now().strftime('%Y-%m-%d')))
            
            if not df_daily.empty and len(df_daily) > 3:
                # F10: Daily points consistency
                features['daily_points_cv'] = (
                    df_daily['total_points_gained'].std() / 
                    df_daily['total_points_gained'].mean()
                ) if df_daily['total_points_gained'].mean() > 0 else 0
                
                # F11: Daily duration consistency
                features['daily_duration_cv'] = (
                    df_daily['session_duration_minutes'].std() / 
                    df_daily['session_duration_minutes'].mean()
                ) if df_daily['session_duration_minutes'].mean() > 0 else 0
                
                # F12: Target-lock detection (cluster around specific values)
                points_rounded = (df_daily['total_points_gained'] / 1000).round() * 1000
                features['daily_target_clustering'] = (points_rounded.value_counts().max() / len(points_rounded))
            else:
                features['daily_points_cv'] = 0
                features['daily_duration_cv'] = 0
                features['daily_target_clustering'] = 0
            
            conn.close()
            
            # Feature 13-15: Weekend vs weekday patterns
            if not df_sessions.empty:
                df_sessions['is_weekend'] = df_sessions['first_online'].dt.dayofweek >= 5
                
                weekend_sessions = df_sessions[df_sessions['is_weekend']]
                weekday_sessions = df_sessions[~df_sessions['is_weekend']]
                
                # F13: Weekend play difference (humans play more on weekends)
                features['weekend_ratio'] = (
                    len(weekend_sessions) / len(df_sessions)
                ) if len(df_sessions) > 0 else 0.29  # baseline ~2/7
                
                # F14: Weekend duration difference
                if len(weekend_sessions) > 0 and len(weekday_sessions) > 0:
                    weekend_avg = weekend_sessions['session_duration_minutes'].mean()
                    weekday_avg = weekday_sessions['session_duration_minutes'].mean()
                    features['weekend_duration_ratio'] = weekend_avg / weekday_avg if weekday_avg > 0 else 1
                else:
                    features['weekend_duration_ratio'] = 1
                
                # F15: Play frequency (bots play daily)
                unique_days = df_sessions['first_online'].dt.date.nunique()
                features['play_frequency'] = unique_days / days
            else:
                features['weekend_ratio'] = 0.29
                features['weekend_duration_ratio'] = 1
                features['play_frequency'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error for {player_name}: {e}")
            return None
    
    def calculate_bot_score(self, features):
        """
        Calculate bot probability from features using learned baseline.
        Returns score 0-100 (higher = more likely bot).
        """
        if not features:
            return 0
        
        bot_score = 0
        reasons = []
        
        # Rule 1: Very low start time variance (< 30 min std)
        if features['start_time_std'] < 0.5:
            bot_score += 20
            reasons.append(f"Extremely regular start times (std={features['start_time_std']:.1f}h)")
        
        # Rule 2: Low start time entropy (< 2 bits)
        if features['start_time_entropy'] < 2.0:
            bot_score += 15
            reasons.append(f"Predictable start times (entropy={features['start_time_entropy']:.2f})")
        
        # Rule 3: Very consistent session durations (CV < 0.1)
        if features['session_duration_cv'] < 0.1:
            bot_score += 15
            reasons.append(f"Identical session durations (CV={features['session_duration_cv']:.2f})")
        
        # Rule 4: Very consistent point gains (CV < 0.15)
        if features['point_gain_cv'] < 0.15:
            bot_score += 10
            reasons.append(f"Robotic point gains (CV={features['point_gain_cv']:.2f})")
        
        # Rule 5: Gap management (CV < 0.15)
        if features['gap_cv'] < 0.15 and features['gap_cv'] < 999:
            bot_score += 20
            reasons.append(f"Maintains exact gap (CV={features['gap_cv']:.2f}, mean={features['gap_mean']:.0f})")
        
        # Rule 6: Daily target lock (clustering > 0.6)
        if features['daily_target_clustering'] > 0.6:
            bot_score += 15
            reasons.append(f"Hits daily target repeatedly ({features['daily_target_clustering']:.0%})")
        
        # Rule 7: No weekend difference (ratio 0.25-0.35 is human)
        if not (0.2 < features['weekend_ratio'] < 0.4):
            bot_score += 10
            reasons.append(f"Unnatural weekend pattern (ratio={features['weekend_ratio']:.2f})")
        
        # Rule 8: Plays every single day (frequency > 0.95)
        if features['play_frequency'] > 0.95:
            bot_score += 10
            reasons.append(f"Plays every day ({features['play_frequency']:.0%})")
        
        # Rule 9: High point gain autocorrelation (> 0.7)
        if features['point_gain_autocorr'] > 0.7:
            bot_score += 10
            reasons.append(f"Predictable point pattern (autocorr={features['point_gain_autocorr']:.2f})")
        
        return min(bot_score, 100), reasons
    
    def analyze_player(self, player_name, days=14):
        """
        Full analysis with self-learning.
        Returns: (bot_score, confidence, reasons, features)
        """
        # Extract features
        features = self.extract_features(player_name, days)
        
        if not features:
            return 0, 0, ["Insufficient data"], {}
        
        # Calculate bot score
        bot_score, reasons = self.calculate_bot_score(features)
        
        # Update learning data
        self.player_profiles[player_name] = {
            'features': features,
            'last_analyzed': datetime.now().isoformat(),
            'bot_score': bot_score
        }
        
        # Record detection history
        self.detection_history[player_name].append({
            'timestamp': datetime.now().isoformat(),
            'score': bot_score
        })
        
        # Save learning data
        self.save_learning_data()
        
        # Confidence based on data quality
        confidence = min(100, len(self.player_profiles.get(player_name, {}).get('features', {})) * 6)
        
        return bot_score, confidence, reasons, features
    
    def get_detailed_report(self, player_name, days=14):
        """Generate comprehensive bot detection report."""
        bot_score, confidence, reasons, features = self.analyze_player(player_name, days)
        
        report = f"🤖 SELF-LEARNING BOT DETECTOR\n"
        report += f"{'='*70}\n"
        report += f"Player: {player_name}\n"
        report += f"Analysis Period: Last {days} days\n"
        report += f"Bot Score: {bot_score}/100\n"
        report += f"Confidence: {confidence}%\n"
        report += f"\n"
        
        if bot_score >= 70:
            report += "🚨 VERDICT: HIGHLY SUSPICIOUS - Likely Bot\n"
        elif bot_score >= 50:
            report += "⚠️  VERDICT: SUSPICIOUS - Possible Bot\n"
        elif bot_score >= 30:
            report += "🟡 VERDICT: QUESTIONABLE - Some Bot-like Behavior\n"
        else:
            report += "✅ VERDICT: NORMAL - Human-like Behavior\n"
        
        report += f"\n{'='*70}\n"
        report += "DETECTED PATTERNS:\n\n"
        
        if reasons:
            for i, reason in enumerate(reasons, 1):
                report += f"{i}. {reason}\n"
        else:
            report += "No suspicious patterns detected.\n"
        
        report += f"\n{'='*70}\n"
        report += "BEHAVIORAL FINGERPRINT:\n\n"
        
        if features:
            report += f"Start Time Regularity:  {features['start_time_std']:.2f}h std (lower = bot)\n"
            report += f"Start Time Randomness:  {features['start_time_entropy']:.2f} bits (lower = bot)\n"
            report += f"Session Duration CV:    {features['session_duration_cv']:.3f} (lower = bot)\n"
            report += f"Point Gain CV:          {features['point_gain_cv']:.3f} (lower = bot)\n"
            
            if features['gap_cv'] < 999:
                report += f"Gap Management CV:      {features['gap_cv']:.3f} (lower = bot)\n"
                report += f"Target Gap:             {features['gap_mean']:.0f} points\n"
            
            report += f"Daily Target Lock:      {features['daily_target_clustering']:.0%} (higher = bot)\n"
            report += f"Weekend Play Ratio:     {features['weekend_ratio']:.2f} (0.29 = normal)\n"
            report += f"Play Frequency:         {features['play_frequency']:.0%} (100% = bot)\n"
        
        return report

# ============================================================================
# CONFIG MANAGER
# ============================================================================
class ConfigMgr:
    def __init__(self):
        self.cfg = {'dark_mode': False, 'window_geometry': '1900x900', 'autopoll_interval': CFG.QUIET_INTERVAL,
                    'hh_interval': CFG.HH_INTERVAL, 'min_3max_time': CFG.MIN_3MAX, 'min_6max_time': CFG.MIN_6MAX,
                    'max_tables': CFG.MAX_TABLES, 'last_period': None, 'intelligent_polling': True, 'pattern_detection': True}
        try:
            if os.path.exists(CFG.CONFIG_FILE):
                self.cfg.update(json.load(open(CFG.CONFIG_FILE)))
        except: pass
    
    def save(self):
        try: json.dump(self.cfg, open(CFG.CONFIG_FILE, 'w'), indent=4)
        except Exception as e: logger.error(f"Config save: {e}")
    
    get = lambda self, k, d=None: self.cfg.get(k, d)
    def set(self, k, v): self.cfg[k] = v; self.save()

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
        
        # Enhanced player_sessions table - FIXED SCHEMA
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
            UNIQUE(player_name, session_date)
        )''')
        
        # Create indexes
        for idx in ['timestamp', 'name', 'period_range']: 
            try:
                c.execute(f'CREATE INDEX IF NOT EXISTS idx_{idx} ON leaderboard({idx})')
            except: pass
        
        try:
            c.execute('CREATE INDEX IF NOT EXISTS idx_session_player ON player_sessions(player_name)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_session_date ON player_sessions(session_date)')
        except: pass
        
        # Check and add missing leaderboard columns
        c.execute('PRAGMA table_info(leaderboard)')
        cols = [r[1] for r in c.fetchall()]
        if 'period_range' not in cols: 
            try: c.execute("ALTER TABLE leaderboard ADD COLUMN period_range TEXT")
            except: pass
        if 'is_happy_hour' not in cols: 
            try: c.execute("ALTER TABLE leaderboard ADD COLUMN is_happy_hour INTEGER DEFAULT 0")
            except: pass
        
        # Check and add missing player_sessions columns
        c.execute('PRAGMA table_info(player_sessions)')
        session_cols = [r[1] for r in c.fetchall()]
        
        missing_cols = {
            'first_online': 'DATETIME',
            'last_offline': 'DATETIME',
            'first_points': 'INTEGER DEFAULT 0',
            'last_points': 'INTEGER DEFAULT 0',
            'total_points_gained': 'INTEGER DEFAULT 0',
            'session_duration_minutes': 'REAL DEFAULT 0'
        }
        
        for col_name, col_type in missing_cols.items():
            if col_name not in session_cols:
                try: 
                    c.execute(f"ALTER TABLE player_sessions ADD COLUMN {col_name} {col_type}")
                except: pass
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    @staticmethod
    def has_data():
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            c = conn.cursor()
            count = c.execute('SELECT COUNT(*) FROM leaderboard').fetchone()[0]
            conn.close()
            return count > 0
        except: return False
    
    @staticmethod
    def update_player_session(player_name, session_date, timestamp, points):
        """
        Update or create player session data.
        FIXED: first_online = FIRST poll with points > 0, NULL if points = 0
        """
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            c = conn.cursor()
            
            # Convert timestamp to Python datetime if it's a pandas Timestamp
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
            
            # Check if session exists
            c.execute('''SELECT id, first_online, last_offline, first_points, last_points 
                         FROM player_sessions 
                         WHERE player_name = ? AND session_date = ?''', 
                      (player_name, str(session_date)))
            
            existing = c.fetchone()
            
            if existing:
                # Update existing session
                session_id, first_online, last_offline, first_points, last_points = existing
                
                # CRITICAL FIX: Convert bytes to int if needed
                if isinstance(first_points, bytes):
                    try:
                        import struct
                        first_points = struct.unpack('<Q', first_points)[0]
                    except:
                        first_points = 0
                
                if isinstance(last_points, bytes):
                    try:
                        import struct
                        last_points = struct.unpack('<Q', last_points)[0]
                    except:
                        last_points = 0
                
                # Convert to datetime objects
                first_online = pd.to_datetime(first_online) if first_online else None
                last_offline = pd.to_datetime(last_offline) if last_offline else None
                
                # Convert pandas Timestamp to Python datetime
                if first_online and hasattr(first_online, 'to_pydatetime'):
                    first_online = first_online.to_pydatetime()
                if last_offline and hasattr(last_offline, 'to_pydatetime'):
                    last_offline = last_offline.to_pydatetime()
                
                # CRITICAL: first_online logic
                if points > 0:
                    # Player has points - set first_online if not set yet, or if this is earlier
                    if first_online is None:
                        first_online = timestamp
                        first_points = points
                    elif timestamp < first_online:
                        first_online = timestamp
                        first_points = points
                else:
                    # Player has 0 points - keep first_online as NULL (offline)
                    first_online = None
                    first_points = 0
                
                # FIXED: Only update last_offline if points INCREASED
                if points > last_points:
                    last_offline = timestamp
                    last_points = points
                # If points stayed the same or decreased to 0, keep old last_offline
                
                # Calculate duration and points gained
                if first_online and last_offline:
                    duration = (last_offline - first_online).total_seconds() / 60
                    points_gained = last_points - first_points
                else:
                    duration = 0
                    points_gained = 0
                
                c.execute('''UPDATE player_sessions 
                             SET first_online = ?, 
                                 last_offline = ?, 
                                 first_points = ?, 
                                 last_points = ?, 
                                 total_points_gained = ?,
                                 session_duration_minutes = ?
                             WHERE id = ?''',
                          (first_online, last_offline, first_points, last_points, 
                           points_gained, duration, session_id))
            else:
                # Create new session
                if points > 0:
                    # Player came online with points - set first_online
                    c.execute('''INSERT INTO player_sessions 
                                 (player_name, session_date, first_online, last_offline, 
                                  first_points, last_points, total_points_gained, session_duration_minutes)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                              (player_name, str(session_date), timestamp, timestamp, 
                               points, points, 0, 0))
                else:
                    # Player has 0 points - create session with NULL first_online (offline)
                    c.execute('''INSERT INTO player_sessions 
                                 (player_name, session_date, first_online, last_offline, 
                                  first_points, last_points, total_points_gained, session_duration_minutes)
                                 VALUES (?, ?, NULL, NULL, 0, 0, 0, 0)''',
                              (player_name, str(session_date)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating player session for {player_name}: {e}")
    
    @staticmethod
    def save(df, period=None):
        if df is None or df.empty: return
        
        df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.now()), errors='coerce').fillna(datetime.now())
        df['period_range'] = period or ""
        df['points'] = df['points'].apply(clean_points)
        ft, hh = df['timestamp'].iloc[0] if len(df) > 0 else datetime.now(), is_hh(df['timestamp'].iloc[0] if len(df) > 0 else None)
        df['is_happy_hour'] = 1 if hh else 0
        df['min_games'] = df['points'].apply(lambda p: calc_min_games(int(p), hh))
        df['combinations'] = df['points'].apply(lambda p: calc_combos(int(p), hh))
        
        # Save to leaderboard table
        df.to_sql('leaderboard', sqlite3.connect(CFG.DB_FILE), if_exists='append', index=False)
        
        # Update player sessions
        for _, row in df.iterrows():
            try:
                player_name = row['name']
                timestamp = row['timestamp']
                points = row['points']
                
                # Determine session date (day when 6am starts)
                if timestamp.hour < 6:
                    session_date = (timestamp - timedelta(days=1)).date()
                else:
                    session_date = timestamp.date()
                
                DB.update_player_session(player_name, session_date, timestamp, points)
            except Exception as e:
                logger.error(f"Error tracking session for {row.get('name', 'unknown')}: {e}")
        
        logger.info(f"Saved {len(df)} records (HH: {hh})")
    
    @staticmethod
    def load(ts=None, period=None):
        conn = sqlite3.connect(CFG.DB_FILE)
        try:
            if period:
                q = 'SELECT * FROM leaderboard WHERE period_range = ? AND timestamp = (SELECT MAX(timestamp) FROM leaderboard WHERE period_range = ?) ORDER BY CAST(rank AS INTEGER)'
                return pd.read_sql_query(q, conn, params=(period, period))
            elif ts:
                return pd.read_sql_query('SELECT * FROM leaderboard WHERE timestamp = ? ORDER BY CAST(rank AS INTEGER)', conn, params=(ts,))
            return pd.read_sql_query('SELECT * FROM leaderboard WHERE timestamp = (SELECT MAX(timestamp) FROM leaderboard) ORDER BY CAST(rank AS INTEGER)', conn)
        except Exception as e: 
            logger.error(f"Load: {e}")
            return pd.DataFrame()
        finally: conn.close()
    
    @staticmethod
    def timestamps():
        conn = sqlite3.connect(CFG.DB_FILE)
        r = [r[0] for r in conn.execute('SELECT DISTINCT timestamp FROM leaderboard ORDER BY timestamp DESC').fetchall()]
        conn.close()
        return r
    
    @staticmethod
    def periods():
        conn = sqlite3.connect(CFG.DB_FILE)
        r = [r[0] for r in conn.execute('SELECT DISTINCT period_range FROM leaderboard WHERE period_range IS NOT NULL AND period_range != "" ORDER BY period_range DESC').fetchall()]
        conn.close()
        return r
    
    @staticmethod
    def players():
        conn = sqlite3.connect(CFG.DB_FILE)
        r = [r[0] for r in conn.execute('SELECT DISTINCT name FROM leaderboard ORDER BY name').fetchall()]
        conn.close()
        return r
    
    @staticmethod
    def player_history(name, period=None):
        conn = sqlite3.connect(CFG.DB_FILE)
        q = 'SELECT timestamp, points, rank FROM leaderboard WHERE name = ?' + (' AND period_range = ?' if period else '') + ' ORDER BY timestamp'
        r = pd.read_sql_query(q, conn, params=(name, period) if period else (name,))
        conn.close()
        return r
    
    @staticmethod
    def backup():
        try:
            os.makedirs(CFG.BACKUP_DIR, exist_ok=True)
            bf = os.path.join(CFG.BACKUP_DIR, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            shutil.copy2(CFG.DB_FILE, bf)
            logger.info(f"Backup: {bf}")
            return bf
        except Exception as e: 
            logger.error(f"Backup: {e}")
            return None
    
    @staticmethod
    def restore(bf):
        try: 
            shutil.copy2(bf, CFG.DB_FILE)
            logger.info(f"Restored: {bf}")
            return True
        except Exception as e: 
            logger.error(f"Restore: {e}")
            return False
    
    @staticmethod
    def export_excel():
        df = pd.read_sql_query('SELECT * FROM leaderboard ORDER BY timestamp DESC, CAST(rank AS INTEGER)', sqlite3.connect(CFG.DB_FILE))
        df.to_excel(CFG.EXCEL_FILE, index=False, engine='openpyxl')
        return CFG.EXCEL_FILE
    
    @staticmethod
    def export_csv():
        df = pd.read_sql_query('SELECT * FROM leaderboard ORDER BY timestamp DESC, CAST(rank AS INTEGER)', sqlite3.connect(CFG.DB_FILE))
        df.to_csv(CFG.CSV_FILE, index=False)
        return CFG.CSV_FILE
    
    @staticmethod
    def get_player_sessions(player_name, start_date=None, end_date=None):
        conn = sqlite3.connect(CFG.DB_FILE)
        try:
            if start_date and end_date:
                query = '''SELECT * FROM player_sessions WHERE player_name = ? AND session_date BETWEEN ? AND ? ORDER BY session_date DESC'''
                df = pd.read_sql_query(query, conn, params=(player_name, start_date, end_date))
            else:
                query = '''SELECT * FROM player_sessions WHERE player_name = ? ORDER BY session_date DESC'''
                df = pd.read_sql_query(query, conn, params=(player_name,))
            return df
        finally: conn.close()

# ============================================================================
# ONE-TIME DATA MIGRATION
# ============================================================================
def migrate_existing_data():
    """
    FIXED: Only track sessions when points ACTUALLY INCREASE.
    """
    try:
        conn = sqlite3.connect(CFG.DB_FILE)
        
        # Get all unique player names
        players = [r[0] for r in conn.execute('SELECT DISTINCT name FROM leaderboard').fetchall()]
        
        print(f"\n{'='*60}")
        print(f"🔄 MIGRATING EXISTING DATA (FIXED)")
        print(f"{'='*60}")
        print(f"Found {len(players)} players to process...")
        
        migrated_sessions = 0
        
        for idx, player in enumerate(players, 1):
            print(f"[{idx}/{len(players)}] Processing: {player}...", end=' ')
            
            # Get all records for this player, sorted by timestamp
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
            
            # Track previous points to detect changes
            prev_points = None
            sessions_processed = 0
            
            for _, row in df.iterrows():
                timestamp = row['timestamp']
                points = row['points']
                
                # CRITICAL: Only process if points CHANGED
                if prev_points is None or points != prev_points:
                    # Determine session date
                    if timestamp.hour < 6:
                        session_date = (timestamp - timedelta(days=1)).date()
                    else:
                        session_date = timestamp.date()
                    
                    # Update or create session
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
# SELENIUM
# ============================================================================
def get_driver(headless=True, retry_count=0, max_retries=3):
    """Create Firefox driver with automatic restart on failure."""
    try:
        opts = Options()
        if headless: 
            opts.add_argument("--headless")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        
        # Increase timeouts to handle sleep mode wake-up
        opts.set_preference("dom.max_script_run_time", 60)
        opts.set_preference("dom.max_chrome_script_run_time", 60)
        opts.page_load_strategy = 'normal'
        
        svc = Service(CFG.GECKO_PATH) if os.path.exists(CFG.GECKO_PATH) else Service()
        d = webdriver.Firefox(service=svc, options=opts)
        d.set_page_load_timeout(60)  # Increased from 30
        d.set_script_timeout(60)
        
        logger.info("Firefox driver created successfully")
        return d
        
    except Exception as e:
        logger.error(f"Driver creation failed (attempt {retry_count + 1}/{max_retries}): {e}")
        
        # Kill any zombie Firefox processes
        try:
            if os.name == 'nt':  # Windows
                os.system('taskkill /F /IM firefox.exe /T >nul 2>&1')
                os.system('taskkill /F /IM geckodriver.exe /T >nul 2>&1')
            else:  # Linux/Mac
                os.system('pkill -9 firefox')
                os.system('pkill -9 geckodriver')
            time.sleep(2)
        except:
            pass
        
        if retry_count < max_retries:
            logger.info(f"Retrying driver creation in 3 seconds...")
            time.sleep(3)
            return get_driver(headless, retry_count + 1, max_retries)
        else:
            raise Exception(f"Failed to create driver after {max_retries} attempts")

def parse_leaderboard(driver):
    try:
        driver.get(CFG.URL)
        time.sleep(8)
        period = None
        try: period = driver.find_element(By.CSS_SELECTOR, "div.activated-set-title").text.strip()
        except: pass
        table = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.playerRankingTable, table[class*='ranking']")))
        rows, parsed, ct = table.find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr"), [], datetime.now()
        for tr in rows:
            try:
                cols = tr.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4: continue
                rk, nm, pts, pr = (cols[0].text.strip(), cols[1].text.strip(), cols[3].text.strip(), cols[4].text.strip()) if len(cols) == 5 else (cols[0].text.strip(), cols[1].text.strip(), cols[2].text.strip(), cols[3].text.strip())
                parsed.append({'rank': rk, 'name': nm, 'points': clean_points(pts), 'prize': pr, 'timestamp': ct, 'period_range': period})
            except: continue
        return pd.DataFrame(parsed) if parsed else pd.DataFrame(columns=["rank", "name", "points", "prize", "timestamp", "period_range"]), period
    except Exception as e: 
        logger.error(f"Parse: {e}")
        return pd.DataFrame(columns=["rank", "name", "points", "prize", "timestamp", "period_range"]), None

# ============================================================================
# GUI APPLICATION
# ============================================================================
class LeaderboardApp:
    def __init__(self, root):
        print("DEBUG: __init__ started")  # ADD THIS
        self.root, self.config = root, ConfigMgr()
        print("DEBUG: ConfigMgr loaded")  # ADD THIS
        
        self.closing, self.is_polling = False, False
        self.timestamps, self.ts_idx, self.periods, self.period_idx = [], 0, [], 0
        self.search_var = tk.StringVar()
        self.polling_mgr, self.pattern_detector = IntelligentPolling(), PatternDetector()
        self.prev_data = None
        
        print("DEBUG: Setting title and geometry")  # ADD THIS
        root.title("Leaderboard Tracker v3.6.1 FINAL - jimmybeam3000")
        root.geometry(self.config.get('window_geometry', '1900x900'))
        
        print("DEBUG: Initializing DB")  # ADD THIS
        DB.init()
        
        print("DEBUG: Building UI")  # ADD THIS
        self.build_ui()
        
        print("DEBUG: Checking for data")
            
        if DB.has_data():
            print("DEBUG: Loading snapshot...")
            self.load_snapshot()
            print(f"✅ Loaded {len(self.timestamps)} snapshots, {len(self.periods)} periods")
            # Don't block with messagebox - show in status bar instead
        else:
            print("⚠ No data yet")
            self.tv.delete(*self.tv.get_children())
            self.ts_label.config(text="No data")
            self.count_label.config(text="0 entries")

        print("DEBUG: Setting close protocol")
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        print("DEBUG: Init complete!")

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
        fm.add_command(label="Excel", command=lambda: self.export('excel'))
        fm.add_command(label="CSV", command=lambda: self.export('csv'))
        fm.add_separator()
        fm.add_command(label="Backup", command=self.backup)
        fm.add_command(label="Restore", command=self.restore)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.on_close)
        vm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="View", menu=vm)
        vm.add_command(label="Logs", command=self.view_logs)
        vm.add_command(label="Player Patterns", command=self.show_patterns)
        cm = tk.Menu(mb, tearoff=0)
        mb.add_cascade(label="Charts", menu=cm)
        cm.add_command(label="Points Progress", command=self.show_points_chart)
        cm.add_command(label="Delta Chart", command=self.show_delta_chart)  # this works only if show_delta_chart is a class method!
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
        self.update_hh()
    
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
    
    def build_verification_controls(self):
        vf = ttk.LabelFrame(self.root, text="🔍 Player Verification", padding=5)
        vf.pack(fill="x", padx=8, pady=4)
        ttk.Label(vf, text="Player:").pack(side="left", padx=5)
        self.verify_player_var = tk.StringVar()
        self.verify_player_combo = ttk.Combobox(vf, textvariable=self.verify_player_var, width=20, state="readonly")
        self.verify_player_combo.pack(side="left", padx=2)
        ttk.Label(vf, text="Points:").pack(side="left", padx=(15,5))
        self.verify_points_var = tk.StringVar()
        ttk.Entry(vf, textvariable=self.verify_points_var, width=8).pack(side="left", padx=2)
        ttk.Label(vf, text="Timeframe:").pack(side="left", padx=(15,5))
        self.verify_timeframe_var = tk.StringVar(value="period")
        ttk.Radiobutton(vf, text="Period", variable=self.verify_timeframe_var, value="period").pack(side="left", padx=5)
        ttk.Radiobutton(vf, text="Hour", variable=self.verify_timeframe_var, value="hour").pack(side="left", padx=5)
        ttk.Radiobutton(vf, text="Custom", variable=self.verify_timeframe_var, value="custom").pack(side="left", padx=5)
        ttk.Label(vf, text="Minutes:").pack(side="left", padx=(10,5))
        self.verify_minutes_var = tk.StringVar(value="60")
        self.verify_minutes_entry = ttk.Entry(vf, textvariable=self.verify_minutes_var, width=6, state='disabled')
        self.verify_minutes_entry.pack(side="left", padx=2)
        self.verify_timeframe_var.trace('w', lambda *args: self.verify_minutes_entry.config(state='normal' if self.verify_timeframe_var.get() == 'custom' else 'disabled'))
        ttk.Button(vf, text="🔍 Verify", command=self.verify_player, width=12).pack(side="left", padx=15)
        ttk.Button(vf, text="🤖 Bot Check", command=self.detect_gap_bot, width=12).pack(side="left", padx=2)
        ttk.Button(vf, text="🔄 Refresh", command=self.refresh_players, width=12).pack(side="left", padx=2)
        self.refresh_players()
    
    def refresh_players(self):
        try:
            players = DB.players()
            if players:
                self.verify_player_combo['values'] = sorted(players)
                if players: self.verify_player_combo.current(0)
        except Exception as e: logger.error(f"Refresh: {e}")
    
    def verify_player(self):  # ← REMOVE the emoji box
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
            except:
                messagebox.showwarning("Error", "Invalid points!")
                return
            
            # ... rest of verify_player code ...
            
            messagebox.showinfo("Verification", report)
        except Exception as e:
            logger.error(f"Verify: {e}")
            messagebox.showerror("Error", f"Failed:\n{e}")

    def detect_gap_bot(self):  # ← This must be at the SAME indentation level as verify_player
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
    
    def check_hh_in_period(self, start_time, end_time):
        if start_time.tzinfo is None: start_time = CFG.TIMEZONE.localize(start_time)
        if end_time.tzinfo is None: end_time = CFG.TIMEZONE.localize(end_time)
        current = start_time
        while current <= end_time:
            if is_hh(current): return True
            current += timedelta(hours=1)
        return False

    def get_session_start(self, dt):
        """Get the 6am session start for a given datetime."""
        if dt.hour < 6:
            return dt.replace(hour=6, minute=0, second=0, microsecond=0) - timedelta(days=1)
        else:
            return dt.replace(hour=6, minute=0, second=0, microsecond=0)
    
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
        self.update_period()
    
    def build_tree(self):
        f = ttk.Frame(self.root)
        f.pack(fill="both", expand=True, padx=8, pady=6)
        session_info_frame = ttk.Frame(f)
        session_info_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,5))
        self.session_info_label = ttk.Label(session_info_frame, text="", font=("Arial", 9, "bold"), foreground="blue")
        self.session_info_label.pack(side="left", padx=10)
        
        # UPDATED: Added 3m and 6m columns after Combo Poll
        cols = ("Rank", "Name", "Points", "Delta Poll", "Combo Poll", "3m", "6m",
                "Delta Day", "Combo Day", "Points/h", "Active Play", 
                "Prize", "Min Games", "Min Sessions", "Period Valid", "Session Valid", "Time")
        self.tv = ttk.Treeview(f, columns=cols, show="headings", height=20)
        for c in cols:
            self.tv.heading(c, text=c)
            self.tv.column(c, width=len(c)*10, anchor="center")
        
        sy = ttk.Scrollbar(f, orient="vertical", command=self.tv.yview)
        sx = ttk.Scrollbar(f, orient="horizontal", command=self.tv.xview)
        self.tv.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.tv.grid(row=1, column=0, sticky="nsew")
        sy.grid(row=1, column=1, sticky="ns")
        sx.grid(row=2, column=0, sticky="ew")
        f.grid_rowconfigure(1, weight=1)
        f.grid_columnconfigure(0, weight=1)
    
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
        messagebox.showinfo("Mode", "Intelligent polling activated!")
    
    def set_manual_mode(self):
        self.poll_interval_entry.config(state='normal')
        messagebox.showinfo("Mode", "Manual mode activated!")
    
    def update_hh(self):
        if self.closing: return
        try:
            self.hh_label.config(text="🎰 HAPPY HOUR (x2)" if is_hh() else "")
            self.root.after(60000, self.update_hh)
        except: pass
    
    def update_period(self):
        try:
            i = get_period_info()
            self.ps_label.config(text=f"Start: {i['start'].strftime('%d.%m %H:%M')}")
            self.pe_label.config(text=f"Elapsed: {i['elapsed_formatted']}")
            self.pend_label.config(text=f"End: {i['end'].strftime('%d.%m %H:%M')}")
            if not self.closing: self.root.after(60000, self.update_period)
        except: pass
    
    def save_settings(self):
        try:
            t3, t6, tb = int(self.t3_var.get()), int(self.t6_var.get()), int(self.tables_var.get())
            if t3 < 1 or t6 < 1 or tb < 1: raise ValueError
            self.config.set('min_3max_time', t3)
            self.config.set('min_6max_time', t6)
            self.config.set('max_tables', tb)
            messagebox.showinfo("OK", "Settings saved!")
            if self.timestamps: self.display(self.timestamps[self.ts_idx])
        except: messagebox.showerror("Error", "Invalid input")
    
    def fetch(self):
        if not check_disk():
            messagebox.showerror("Disk Full", "Free up space!")
            return
            
        def worker():
            d = None
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    self.root.after(0, lambda: self.status_label.config(text="Fetching...", foreground="orange"))
                    self.root.after(0, self.progress.start)
                    
                    # Create driver with timeout protection
                    try:
                        d = get_driver(True)
                    except Exception as driver_error:
                        logger.error(f"Driver creation failed (attempt {retry_count + 1}/{max_retries}): {driver_error}")
                        retry_count += 1
                        time.sleep(5)  # Wait before retry
                        continue
                    
                    df, period = parse_leaderboard(d)
                    
                    if df.empty:
                        self.root.after(0, lambda: self.status_label.config(text="No data", foreground="red"))
                        break
                    
                    alerts = self.pattern_detector.detect(df, self.prev_data)
                    changes = self.polling_mgr.calc_changes(df, self.prev_data)
                    self.polling_mgr.update(changes)
                    self.prev_data = df.copy()
                    
                    if alerts:
                        for alert in alerts:
                            self.root.after(0, lambda a=alert: messagebox.showinfo("Alert", a))
                    
                    DB.save(df, period)
                    self.root.after(0, self.load_snapshot)
                    if period: self.root.after(0, lambda p=period: self.period_label.config(text=p))
                    
                    ts = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                    self.root.after(0, lambda t=ts: self.status_label.config(text=f"OK {t}", foreground="green"))
                    self.root.after(0, lambda: self.poll_status_label.config(text=self.polling_mgr.status()))
                    
                    # Success - break retry loop
                    break
                    
                except TimeoutException as timeout_error:
                    logger.error(f"Timeout (attempt {retry_count + 1}/{max_retries}): {timeout_error}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        self.root.after(0, lambda: self.status_label.config(text="Timeout - Max retries", foreground="red"))
                        
                except Exception as e:
                    logger.error(f"Fetch error (attempt {retry_count + 1}/{max_retries}): {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        self.root.after(0, lambda: self.status_label.config(text="Error - Max retries", foreground="red"))
                
                finally:
                    # Always cleanup driver
                    if d:
                        try: 
                            d.quit()
                        except: 
                            pass
                        d = None
            
            self.root.after(0, self.progress.stop)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def load_snapshot(self):
        try:
            print("DEBUG: Getting timestamps...")
            self.timestamps = DB.timestamps()
            print(f"DEBUG: Got {len(self.timestamps)} timestamps")
            
            print("DEBUG: Getting periods...")
            self.periods = DB.periods()
            print(f"DEBUG: Got {len(self.periods)} periods")
            
            if not self.timestamps:
                self.tv.delete(*self.tv.get_children())
                self.ts_label.config(text="No data")
                self.count_label.config(text="0 entries")
                return
            
            self.ts_idx, self.period_idx = 0, 0
            
            print(f"DEBUG: Displaying first timestamp: {self.timestamps[0]}")
            self.display(self.timestamps[0])
            
            if self.periods: 
                self.period_label.config(text=self.periods[0])
            
            print("DEBUG: load_snapshot complete!")
        except Exception as e: 
            logger.error(f"Load: {e}")
            print(f"DEBUG ERROR: {e}")
    
    def display(self, ts):
        """UPDATED: 3m/6m show game combinations, not time. FIXED: Bytes conversion."""
        if not ts: return
        df = DB.load(ts=ts)
        if df.empty:
            self.tv.delete(*self.tv.get_children())
            self.ts_label.config(text=str(ts))
            self.count_label.config(text="0 entries")
            return
        
        try: 
            t3, t6, tb = int(self.t3_var.get()), int(self.t6_var.get()), int(self.tables_var.get())
        except: 
            t3, t6, tb = CFG.MIN_3MAX, CFG.MIN_6MAX, CFG.MAX_TABLES
        
        prev_pts, prev_ts = {}, None
        if len(self.timestamps) > self.ts_idx + 1:
            dfp = DB.load(ts=self.timestamps[self.ts_idx + 1])
            if not dfp.empty:
                prev_pts = dict(zip(dfp['name'], dfp['points']))
                try: 
                    prev_ts = pd.to_datetime(self.timestamps[self.ts_idx + 1])
                    if prev_ts.tzinfo is None: 
                        prev_ts = CFG.TIMEZONE.localize(prev_ts)
                except: 
                    pass
        
        ct = pd.to_datetime(ts) if isinstance(ts, str) else ts
        if ct.tzinfo is None: 
            ct = CFG.TIMEZONE.localize(ct)
        
        session_start = self.get_session_start(ct)
        session_end = session_start + timedelta(days=1)
        self.session_info_label.config(
            text=f"📅 Session: {session_start.strftime('%b %d')} 06:00 - {session_end.strftime('%b %d')} 05:59"
        )
        
        hh = bool(df['is_happy_hour'].iloc[0]) if 'is_happy_hour' in df.columns and len(df) > 0 else is_hh(ct)
        self.tv.delete(*self.tv.get_children())
        
        
        # PRE-CALCULATE ALL DELTA DAYS IN ONE BATCH QUERY
        session_date = session_start.date()
        try:
            conn = sqlite3.connect(CFG.DB_FILE)
            query_all = '''SELECT player_name, first_points, session_duration_minutes, break_duration_minutes
                           FROM player_sessions WHERE session_date = ?'''
            session_data = {}
            for row in conn.execute(query_all, (str(session_date),)):
                player_name = row[0]
                first_pts = row[1]
                if isinstance(first_pts, bytes):
                    import struct
                    try: first_pts = struct.unpack('<Q', first_pts)[0]
                    except: first_pts = 0
                else:
                    first_pts = int(first_pts) if first_pts else 0
                session_data[player_name] = {
                    'first_points': first_pts,
                    'duration': float(row[2]) if row[2] else 0,
                    'breaks': float(row[3]) if row[3] else 0
                }
            conn.close()
        except Exception as e:
            logger.error(f"Batch session query error: {e}")
            session_data = {}
        
        for _, r in df.iterrows():
            nm, pts = r.get("name", ""), r.get("points", 0)
            prv, dlt_poll = prev_pts.get(nm, pts), pts - prev_pts.get(nm, pts)
            
            # Calculate Delta Day using pre-fetched data
            if nm in session_data:
                sd = session_data[nm]
                dlt_day = pts - sd['first_points']
                session_duration = sd['duration']
                break_duration = sd['breaks']
                active_play_minutes = max(0, session_duration - break_duration)
                points_per_hour = (dlt_day / active_play_minutes) * 60 if active_play_minutes > 0 else 0
            else:
                dlt_day = 0
                session_duration = 0
                break_duration = 0
                active_play_minutes = 0
                points_per_hour = 0
            
            dm = (ct - prev_ts).total_seconds() / 60 if prev_ts else 0
            
            # Calculate combinations
            combos_poll = calc_combos(dlt_poll, hh) if dlt_poll > 0 else ""
            combos_day = calc_combos(dlt_day, hh) if dlt_day > 0 else ""
            
            # CRITICAL: Calculate 3m and 6m as GAME COMBINATIONS (not time)
            combo_3m = calc_3max_only_combo(dlt_poll, hh) if dlt_poll > 0 else ""
            combo_6m = calc_6max_only_combo(dlt_poll, hh) if dlt_poll > 0 else ""
            
            # Calculate min games and sessions
            mgt = r.get("min_games", calc_min_games(pts, hh))
            mgd = calc_min_games(dlt_poll, hh) if dlt_poll > 0 else 0
            mst = calc_min_sessions(mgt, tb)
            msd = calc_min_sessions(mgd, tb) if dlt_poll > 0 else 0
            
            # Period validation with breaks subtracted
            pv, pr, _, _, _ = validate_period(pts, t3, t6, tb, hh, ct, player_name=nm)
            
            # Determine game type
            game_type = "mixed"
            if combos_poll:
                if "6max" in combos_poll and "3max" not in combos_poll:
                    game_type = "6max"
                elif "3max" in combos_poll and "6max" not in combos_poll:
                    game_type = "3max"
            
            # Validate session
            if dlt_poll > 0 and dm > 0:
                dv, dr, _, _ = validate_session(mgd, dm, t3, t6, tb, hh, game_type)
            else:
                dv, dr = True, "N/A"

            try: 
                ts_val = r.get("timestamp", "")
                if pd.notna(ts_val):
                    td_obj = pd.to_datetime(ts_val)
                    time_full = td_obj.strftime("%H:%M:%S - %d.%m.%Y")
                else:
                    time_full = ""
            except: 
                time_full = ""

            # Format displays
            delta_poll_display = f"{dlt_poll:+d}" if dlt_poll != 0 else "0"
            if dlt_poll > 0 and hh: 
                delta_poll_display += " 🎰"

            delta_day_display = f"{dlt_day:+d}" if dlt_day != 0 else "0"
            if dlt_day > 0 and hh: 
                delta_day_display += " 🎰"
            
            points_h_display = f"{points_per_hour:.0f}" if points_per_hour > 0 else "0"
            active_play_display = f"{active_play_minutes/60:.1f}h" if active_play_minutes > 0 else "0h"

            # UPDATED: 3m and 6m show game combinations
            vals = (r.get("rank", ""), nm, f"{pts:,}", 
                    delta_poll_display, combos_poll,
                    combo_3m, combo_6m,  # Game combinations for 3-max only / 6-max only
                    delta_day_display, combos_day,
                    points_h_display, active_play_display,
                    r.get("prize", ""), f"{mgt}", f"{mst}", pr, 
                    dr, time_full)

            iid = self.tv.insert("", "end", values=vals)
            
            # Color coding
            if dlt_poll > 0: 
                self.tv.item(iid, tags=("positive",))
            
            if dlt_day > 1000:
                self.tv.item(iid, tags=("high_day",))
            elif dlt_day > 500:
                self.tv.item(iid, tags=("medium_day",))
            
            if points_per_hour > 2000:
                self.tv.item(iid, tags=("high_efficiency",))
            
            if not pv or (not dv and dlt_poll > 0): 
                self.tv.item(iid, tags=("warning",))
        
        # Configure tags
        self.tv.tag_configure("positive", background="lightgreen")
        self.tv.tag_configure("warning", background="#ffcccc")
        self.tv.tag_configure("high_day", background="#90EE90")
        self.tv.tag_configure("medium_day", background="#FFFFE0")
        self.tv.tag_configure("high_efficiency", background="#87CEEB")
        
        self.count_label.config(text=f"{len(df)} entries")

        # Auto-adjust columns
        for i, col in enumerate(self.tv["columns"]):
            max_width = len(col) * 9
            for item in self.tv.get_children():
                try:
                    val = str(self.tv.item(item)["values"][i])
                    val_width = len(val) * 9
                    max_width = max(max_width, val_width)
                except:
                    pass
            final_width = min(max_width + 20, 400)
            self.tv.column(col, width=final_width)
        
        try:
            dt = pd.to_datetime(ts) if isinstance(ts, str) else ts
            display_text = dt.strftime("%d.%m.%Y %H:%M:%S")
            if hh: 
                display_text += " 🎰"
            self.ts_label.config(text=display_text)
        except: 
            self.ts_label.config(text=str(ts))
    
    def filter(self):
        s = self.search_var.get().lower()
        if not s:
            for i in self.tv.get_children(): self.tv.reattach(i, '', 'end')
        else:
            for i in self.tv.get_children():
                if s in str(self.tv.item(i)['values'][1]).lower(): self.tv.reattach(i, '', 'end')
                else: self.tv.detach(i)
    
    def prev_ts(self):
        if not self.timestamps: return
        self.ts_idx = min(self.ts_idx + 1, len(self.timestamps) - 1)
        self.display(self.timestamps[self.ts_idx])
    
    def next_ts(self):
        if not self.timestamps: return
        self.ts_idx = max(self.ts_idx - 1, 0)
        self.display(self.timestamps[self.ts_idx])
    
    def prev_period(self):
        if not self.periods: return
        self.period_idx = min(self.period_idx + 1, len(self.periods) - 1)
        self.period_label.config(text=self.periods[self.period_idx])
    
    def next_period(self):
        if not self.periods: return
        self.period_idx = max(self.period_idx - 1, 0)
        self.period_label.config(text=self.periods[self.period_idx])
    
    def start_poll(self):
        if self.is_polling:
            messagebox.showinfo("Info", "Already running")
            return
        try:
            interval = float(self.poll_interval_var.get())
            if interval < CFG.MIN_POLL: raise ValueError
        except:
            messagebox.showerror("Error", f"Invalid! Min: {CFG.MIN_POLL} min")
            return
        
        if self.poll_mode_var.get() == "manual":
            self.polling_mgr.set_manual(interval)
        
        self.is_polling = True
        prevent_sleep()  # ← ADD THIS LINE
        self.poll_status_label.config(text=self.polling_mgr.status(), foreground="blue")
        
        def loop():
            while self.is_polling and not self.closing:
                self.fetch()
                delay = int(self.polling_mgr.interval) if hasattr(self.polling_mgr, 'interval') else int(interval * 60)
                for _ in range(delay):
                    if not self.is_polling or self.closing: break
                    time.sleep(1)
        
        threading.Thread(target=loop, daemon=True).start()
    
    def stop_poll(self):
        if not self.is_polling: return
        self.is_polling = False
        allow_sleep()  # ← ADD THIS LINE
        self.poll_status_label.config(text="Stopped", foreground="gray")
        self.status_label.config(text="Stopped", foreground="red")
    
    def export(self, fmt):
        try:
            fp = DB.export_excel() if fmt == 'excel' else DB.export_csv()
            messagebox.showinfo("OK", f"Exported:\n{fp}")
            if os.name == 'nt': os.startfile(fp)
        except Exception as e: messagebox.showerror("Error", f"Export failed:\n{e}")
    
    def backup(self):
        bf = DB.backup()
        if bf: messagebox.showinfo("OK", f"Backup:\n{bf}")
        else: messagebox.showerror("Error", "Backup failed")
    
    def restore(self):
        fp = filedialog.askopenfilename(title="Select Backup", initialdir=CFG.BACKUP_DIR, filetypes=[("DB", "*.db")])
        if fp and messagebox.askyesno("Restore", "Overwrite?"):
            if DB.restore(fp): messagebox.showinfo("OK", "Restored!"); self.load_snapshot()
            else: messagebox.showerror("Error", "Restore failed")
    
    def view_logs(self):
        w = tk.Toplevel(self.root)
        w.title("Logs")
        w.geometry("900x600")
        t = scrolledtext.ScrolledText(w, wrap=tk.WORD, font=("Courier", 9))
        t.pack(fill="both", expand=True, padx=10, pady=10)
        try:
            if os.path.exists(CFG.LOG_FILE):
                t.insert('1.0', open(CFG.LOG_FILE, 'r', encoding='utf-8').read())
                t.see('end')
            else: t.insert('1.0', "No logs")
        except Exception as e: t.insert('1.0', f"Error: {e}")
        t.config(state='disabled')
        ttk.Button(w, text="Close", command=w.destroy).pack(pady=5)

    def show_patterns(self):
        """Player patterns - opens with TODAY's session by default."""
        if hasattr(self, '_patterns_window') and self._patterns_window and self._patterns_window.winfo_exists():
            self._patterns_window.lift()
            return
        
        w = tk.Toplevel(self.root)
        w.title("Player Patterns - Daily Sessions")
        w.geometry("2000x800")  # Increased width for new column
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
        except:
            session_dates = []
        
        # Determine today's session
        now = datetime.now(CFG.TIMEZONE)
        if now.hour < 6:
            today_session = (now - timedelta(days=1)).date()
        else:
            today_session = now.date()
        
        current_session = [today_session if today_session in session_dates else (session_dates[0] if session_dates else None)]
        
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
        
        # SORT BY CONTROLS - UPDATED
        ttk.Label(nav_frame, text="Sort by:", font=("Arial", 10, "bold")).pack(side="left", padx=(20,5))
        sort_var = tk.StringVar(value="Rank")
        sort_combo = ttk.Combobox(nav_frame, textvariable=sort_var, width=15, state="readonly")
        sort_combo['values'] = ["Rank", "Name", "First Online", "Last Online", "Hours Played", "Break Total", "Points/h", "Total Sleep"]
        sort_combo.current(0)
        sort_combo.pack(side="left", padx=2)
        
        # Treeview with NEW Points/h COLUMN
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill="both", expand=True, pady=10)
        
        # UPDATED: Added Points/h column
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
            Calculate daily stats by querying leaderboard data directly.
            INCLUDES: Sleep time and break detection (10+ min gaps without point increases).
            FIXED: Timezone handling for break detection.
            """
            try:
                conn = sqlite3.connect(CFG.DB_FILE)
                
                # Session boundaries: 6am today to 5:59am tomorrow
                session_start = datetime.combine(sdate, dt_time(6, 0))
                session_end = session_start + timedelta(days=1)
                
                # Ensure timezone awareness
                if session_start.tzinfo is None:
                    session_start = CFG.TIMEZONE.localize(session_start)
                if session_end.tzinfo is None:
                    session_end = CFG.TIMEZONE.localize(session_end)
                
                # Query ALL polls for this player in this session
                query = '''
                    SELECT timestamp, points 
                    FROM leaderboard 
                    WHERE name = ? 
                      AND timestamp >= ? 
                      AND timestamp < ?
                    ORDER BY timestamp ASC
                '''
                
                df = pd.read_sql_query(query, conn, params=(
                    player, 
                    session_start.strftime('%Y-%m-%d %H:%M:%S'),
                    session_end.strftime('%Y-%m-%d %H:%M:%S')
                ))
                
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
                
                # CRITICAL: Ensure all timestamps are timezone-naive (remove timezone info)
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                
                # CRITICAL: Find first INCREASE in points
                first_online = None
                first_points = None
                
                # Check first poll vs previous session
                if df['points'].iloc[0] > 0:
                    prev_query = '''
                        SELECT points FROM leaderboard
                        WHERE name = ? AND timestamp < ?
                        ORDER BY timestamp DESC LIMIT 1
                    '''
                    prev_result = conn.execute(prev_query, (
                        player, 
                        session_start.strftime('%Y-%m-%d %H:%M:%S')
                    )).fetchone()
                    
                    prev_points = prev_result[0] if prev_result else 0
                    
                    if df['points'].iloc[0] > prev_points:
                        first_online = df['timestamp'].iloc[0]
                        first_points = df['points'].iloc[0]
                
                # Check subsequent polls for point increases
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
                
                # Get last poll with points
                # CRITICAL: Find LAST INCREASE in points (not just last poll with points)
                last_offline = None
                last_points = None
                
                # Start from the end and work backwards to find last increase
                for i in range(len(df) - 1, 0, -1):
                    if df['points'].iloc[i] > df['points'].iloc[i-1]:
                        last_offline = df['timestamp'].iloc[i]
                        last_points = df['points'].iloc[i]
                        break
                
                # If no increase found in the loop, check if first poll was an increase
                if last_offline is None and df['points'].iloc[0] > 0:
                    # Check against previous session
                    prev_query = '''
                        SELECT points FROM leaderboard
                        WHERE name = ? AND timestamp < ?
                        ORDER BY timestamp DESC LIMIT 1
                    '''
                    prev_result = conn.execute(prev_query, (
                        player, 
                        session_start.strftime('%Y-%m-%d %H:%M:%S')
                    )).fetchone()
                    
                    prev_points = prev_result[0] if prev_result else 0
                    
                    if df['points'].iloc[0] > prev_points:
                        last_offline = df['timestamp'].iloc[0]
                        last_points = df['points'].iloc[0]
                
                # If still no increase found, player was never active
                if last_offline is None:
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

                # === SMART BREAK DETECTION ===
                # Only count breaks if we were actively polling
                break_total_minutes = 0
                last_increase_time = first_online
                last_increase_idx = 0
                
                for i in range(1, len(df)):
                    current_time = df['timestamp'].iloc[i]
                    current_points = df['points'].iloc[i]
                    prev_points = df['points'].iloc[i-1]
                    
                    # If points increased, check gap from last increase
                    if current_points > prev_points:
                        gap_minutes = (current_time - last_increase_time).total_seconds() / 60
                        
                        # Count polls between last increase and this increase
                        polls_in_gap = i - last_increase_idx
                        
                        # CRITICAL: Only count as break if:
                        # 1. Gap >= 10 minutes
                        # 2. We had at least 2 polls in the gap (proof of active monitoring)
                        if gap_minutes >= 10 and polls_in_gap >= 2:
                            # Estimate polling interval from gap
                            avg_poll_interval = gap_minutes / polls_in_gap if polls_in_gap > 0 else 0
                            
                            # Break time = gap minus expected polling overhead
                            # (Conservative: assume player was idle between polls)
                            expected_poll_time = avg_poll_interval * 1.5  # 50% buffer for polling variance
                            break_time = max(0, gap_minutes - expected_poll_time)
                            
                            # Only count significant breaks (>5 min after overhead)
                            if break_time >= 5:
                                break_total_minutes += break_time
                        
                        last_increase_time = current_time
                        last_increase_idx = i
                
                # === SLEEP CALCULATION ===
                # Sleep = time from previous session's last_offline to this session's first_online
                sleep_time = "00:00"
                
                try:
                    prev_session_date = sdate - timedelta(days=1)
                    prev_session_start = datetime.combine(prev_session_date, dt_time(6, 0))
                    prev_session_end = prev_session_start + timedelta(days=1)
                    
                    # Get last poll with points from previous session
                    query_prev = '''
                        SELECT timestamp, points
                        FROM leaderboard 
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
                        
                        # Remove timezone info from both to make them comparable
                        if hasattr(prev_last_offline, 'tz_localize'):
                            prev_last_offline = prev_last_offline.tz_localize(None)
                        if hasattr(first_online, 'tz_localize'):
                            first_online_naive = first_online.tz_localize(None) if first_online.tzinfo else first_online
                        else:
                            first_online_naive = first_online
                        
                        sleep_seconds = (first_online_naive - prev_last_offline).total_seconds()
                        
                        if sleep_seconds > 0:
                            sleep_hours = int(sleep_seconds // 3600)
                            sleep_minutes = int((sleep_seconds % 3600) // 60)
                            sleep_time = f"{sleep_hours:02d}:{sleep_minutes:02d}"
                except Exception as e:
                    logger.error(f"Sleep calc error for {player}: {e}")
                
                conn.close()
                
                # Session duration
                session_duration_minutes = (last_offline - first_online).total_seconds() / 60
                
                # Active play time (excluding breaks)
                active_play_minutes = max(0, session_duration_minutes - break_total_minutes)
                
                # Points gained
                points_gained = last_points - first_points
                
                # Points/hour (based on active play, not total duration)
                if active_play_minutes > 0:
                    points_per_hour = (points_gained / active_play_minutes) * 60
                else:
                    points_per_hour = 0
                
                # Format break time (only show if significant)
                if break_total_minutes >= 5:  # Only show breaks ≥5 min
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
                return None
        
        def update_display():
            """Run player processing in background thread to prevent UI freeze."""
            
            # Clear immediately
            tv.delete(*tv.get_children())
            
            if current_session[0] is None:
                session_label.config(text="No Session Data Available")
                return
            
            sdate = current_session[0]
            session_label.config(text=f"Daily Session: {sdate.strftime('%b %d')} 06:00 - {(sdate + timedelta(days=1)).strftime('%b %d')} 05:59")
            
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
                    except:
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
                        player_stats.sort(key=lambda x: x[2])  # Ascending (1, 2, 3...)
                    elif sort_key == "Name":
                        player_stats.sort(key=lambda x: x[0].lower())  # Case-insensitive alphabetical
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
                                # Parse "9h 29min" or "45min"
                                total_min = 0
                                if 'h' in bt:
                                    parts = bt.split('h')
                                    total_min += int(parts[0]) * 60
                                    if 'min' in parts[1]:
                                        total_min += int(parts[1].replace('min', '').strip())
                                elif 'min' in bt:
                                    total_min = int(bt.replace('min', '').strip())
                                return total_min
                            except:
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
                                # Parse "12:39" format (hours:minutes)
                                parts = sleep.split(':')
                                return int(parts[0]) * 60 + int(parts[1])
                            except:
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
            if current_session[0] is None: return
            idx = session_dates.index(current_session[0]) if current_session[0] in session_dates else -1
            if idx < len(session_dates) - 1:
                current_session[0] = session_dates[idx + 1]
                update_display()
        
        def go_next_day():
            if current_session[0] is None: return
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
                if os.name == 'nt': os.startfile(fn)
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
        except:
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
            except:
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
                if os.name == 'nt': os.startfile(fn)
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
        
    def show_points_chart(self):
        """Show points progress chart with matplotlib - FIXED."""
        w = tk.Toplevel(self.root)
        w.title("Points Progress - Top 10")
        w.geometry("1200x700")
        
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
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                top10 = df.groupby('name')['points'].max().nlargest(10).index.tolist()
                
                fig, ax = plt.subplots(figsize=(12, 6))
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
                ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
                fig.autofmt_xdate()
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
        
        ttk.Button(cf, text="Update", command=plot).pack(side="left", padx=10)
        plot()  # Initial plot

    def calculate_reset_aware_delta(df, points_col='points'):
        """
        Calculates a reset-aware delta for a DataFrame.
        Ensures deltas never go negative and resets correctly for daily totals.
        """
        df = df.sort_values('timestamp').copy()
        delta_day = []
        last_points = None

        for points in df[points_col]:
            if last_points is None:
                delta = 0
            else:
                # Prevent negative delta after a reset
                delta = max(points - last_points, 0)
            delta_day.append(delta)
            last_points = points

        df['delta_day'] = delta_day
        df['delta_day'] = df['delta_day'].cumsum()  # cumulative per day
        return df

    # ==================== DELTA CALCULATION ====================
    def calculate_daily_delta_with_reset(self, df_player, session_start):
        """
        Calculate Δ points for the day, accounting for weekly reset at Wed 12:00.
        Ensures no negative values and smooth continuity across resets.
        """
        df_player = df_player.sort_values('timestamp').copy()
        df_player['delta_day'] = 0

        carry = 0
        last_points = None

        # Reset time for Wednesday
        reset_time = datetime.combine(session_start.date(), dt_time(12, 0))
        if reset_time < session_start:
            reset_time += timedelta(days=1)

        for i, row in df_player.iterrows():
            points = row['points']

            if last_points is None:
                df_player.at[i, 'delta_day'] = 0
                last_points = points
                continue

            # Detect reset: points drop at Wed noon
            if (points < last_points * 0.5 and row['timestamp'] >= reset_time) or (points == 0 and row['timestamp'] >= reset_time):
                carry += last_points
                last_points = points

            delta = points - last_points
            df_player.at[i, 'delta_day'] = max(delta, 0) + carry
            last_points = points

        return df_player['delta_day']

    # ==================== DELTA CHART ====================
    def show_delta_chart(self):
        if not hasattr(self, 'player_dfs') or not self.player_dfs:
            messagebox.showerror("Fehler", "Keine Spielerdaten geladen.")
            return

        session_start = self.get_current_session_start()
        session_end = session_start + timedelta(days=1)

        plt.figure(figsize=(10, 5))
        plt.title("Δ Punkte pro Spieler (Tagesverlauf)")
        plt.xlabel("Zeit")
        plt.ylabel("Δ Punkte")
        plt.grid(True)

        for player, df in self.player_dfs.items():
            df_player = df[
                (df['timestamp'] >= session_start) &
                (df['timestamp'] < session_end)
            ].copy()

            if df_player.empty:
                continue

            df_player['delta_day'] = self.calculate_daily_delta_with_reset(df_player, session_start)
            plt.plot(df_player['timestamp'], df_player['delta_day'], label=player)

        plt.legend()
        plt.tight_layout()
        plt.show()

    # ==================== COMPARISON CHART ====================
    def show_comparison(self):
        if not hasattr(self, 'player_dfs') or not self.player_dfs:
            messagebox.showerror("Fehler", "Keine Spielerdaten geladen.")
            return

        session_start = self.get_current_session_start()
        session_end = session_start + timedelta(days=1)

        comparison_data = []

        for player, df in self.player_dfs.items():
            df_player = df[
                (df['timestamp'] >= session_start) &
                (df['timestamp'] < session_end)
            ].copy()

            if df_player.empty:
                continue

            df_player['delta_day'] = self.calculate_daily_delta_with_reset(df_player, session_start)
            total_delta = df_player['delta_day'].iloc[-1] if not df_player.empty else 0
            comparison_data.append((player, total_delta))

        if not comparison_data:
            messagebox.showinfo("Info", "Keine Daten für diesen Zeitraum gefunden.")
            return

        comparison_data.sort(key=lambda x: x[1], reverse=True)
        players, deltas = zip(*comparison_data)

        plt.figure(figsize=(8, 5))
        plt.bar(players, deltas)
        plt.title("Tagesvergleich: Δ Punkte pro Spieler")
        plt.ylabel("Gesamt Δ Punkte")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

    # ==================== WINDOW CLOSE ====================
    def on_close(self):
        """Handle window close."""
        try:
            self.config.set('window_geometry', self.root.geometry())
        except:
            pass

        try:
            self.config.set('min_3max_time', int(self.t3_var.get()))
            self.config.set('min_6max_time', int(self.t6_var.get()))
        except:
            pass

        self.closing, self.is_polling = True, False
        allow_sleep()  # ← Ensures system can sleep if needed
        time.sleep(0.5)

        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass


# ============================================================================
# MAIN
# ============================================================================
def main():
    if not check_disk(): 
        print(f"WARNING: Less than {CFG.MIN_FREE_DISK_MB} MB free disk space!")
    
    logger.info("=" * 80)
    logger.info("Leaderboard Tracker v3.6.1 FINAL WITH MIGRATION - STARTUP")
    logger.info(f"User: jimmybeam3000")
    logger.info(f"UTC: {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"MESZ: {datetime.now(CFG.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Intelligent Polling: ENABLED")
    logger.info(f"Pattern Detection: ENABLED")
    logger.info(f"Enhanced Session Tracking: ENABLED")
    
    # ONE-TIME MIGRATION - FORCE RE-RUN FOR BUGGY DATA
    try:
        conn = sqlite3.connect(CFG.DB_FILE)
        
        # Check if we need to re-migrate (detect the bug)
        # If >80% of first_online times are 06:00-06:10, we have the bug
        query = '''SELECT COUNT(*) FROM player_sessions 
                   WHERE first_online IS NOT NULL 
                   AND strftime('%H:%M', first_online) BETWEEN '06:00' AND '06:10' '''
        buggy_count = conn.execute(query).fetchone()[0]
        
        query_total = '''SELECT COUNT(*) FROM player_sessions WHERE first_online IS NOT NULL'''
        total_count = conn.execute(query_total).fetchone()[0]
        
        leaderboard_count = conn.execute('SELECT COUNT(*) FROM leaderboard').fetchone()[0]
        conn.close()
        
        # If >80% of sessions have 06:00-06:10 first_online, we have the bug
        needs_migration = False
        
        if total_count == 0 and leaderboard_count > 0:
            print("\n" + "="*60)
            print("🔄 FIRST RUN - No session data")
            print("="*60)
            needs_migration = True
        elif total_count > 0 and buggy_count > (total_count * 0.8):
            print("\n" + "="*60)
            print("🐛 BUG DETECTED: Incorrect first_online timestamps")
            print("="*60)
            print(f"Found {buggy_count}/{total_count} sessions with 06:00-06:10 first_online")
            print("This indicates the bug where first_online = session boundary instead of actual first activity")
            print("Running corrective migration to fix all timestamps...")
            print("="*60 + "\n")
            needs_migration = True
        
        if needs_migration:
            # CRITICAL: Clear all existing session data
            print("🗑️  Clearing corrupted session data...")
            conn = sqlite3.connect(CFG.DB_FILE)
            conn.execute('DELETE FROM player_sessions')
            conn.commit()
            conn.close()
            print("✅ Cleared\n")
            
            # Run migration
            migrate_existing_data()
            
            print("\n✅ Migration complete! Starting application...\n")
        else:
            logger.info(f"Session data OK: {total_count} sessions tracked, {buggy_count} at 06:00-06:10 (normal)")
            
    except Exception as e:
        logger.warning(f"Migration check: {e}")
    def clean_selenium_temp():
        """Auto-clean temp on startup."""
        try:
            temp_path = r"C:\Users\test\AppData\Local\Temp"
            freed = 0
            
            for item in os.listdir(temp_path):
                # Delete Firefox/Selenium junk
                if any(x in item.lower() for x in ['rust_mozprofile', 'tmp', 'webdriver', 'firefox', 'gecko']):
                    try:
                        path = os.path.join(temp_path, item)
                        if os.path.isfile(path):
                            size = os.path.getsize(path)
                            os.unlink(path)
                            freed += size
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                    except:
                        pass
            
            if freed > 0:
                logger.info(f"Cleaned {freed/(1024**2):.2f} MB of temp files")
        except Exception as e:
            logger.warning(f"Temp clean: {e}")

    # Add this line in main() before root = tk.Tk()
    clean_selenium_temp()
    
    root = tk.Tk()
    app = LeaderboardApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()
    except Exception as e:
        logger.error(f"Fatal: {e}")

if __name__ == "__main__":
    main()