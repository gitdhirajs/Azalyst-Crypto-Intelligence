"""
Purged walk-forward backtester with dynamic model retraining.
Uses TimeSeriesSplit with purge gap and model version replay.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import FEATURES_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR

log = logging.getLogger("azalyst.backtest_advanced")

class PurgedWalkForward:
    def __init__(self, n_splits=5, purge_hours=3, long_threshold=0.62,
                 stop_loss=1.0, take_profit=2.0, cycle_minutes=5):
        self.n_splits = n_splits
        self.purge = pd.Timedelta(hours=purge_hours)
        self.thr = long_threshold
        self.sl = stop_loss
        self.tp = take_profit
        self.cycle = cycle_minutes

    def prepare_data(self):
        path = FEATURES_DIR / 'latest_features.csv'
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "scan_time" not in df.columns:
            return pd.DataFrame()
        df['scan_time'] = pd.to_datetime(df['scan_time'], utc=True)
        df = df.sort_values('scan_time').reset_index(drop=True)
        # We need labels to train/validate
        df = df.dropna(subset=['label'])
        return df

    def _train_model(self, X, y):
        model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                              eval_metric='logloss', random_state=42, tree_method='hist')
        model.fit(X, y)
        return model

    def run(self) -> Dict:
        df = self.prepare_data()
        if df.empty: return {'status': 'no_data'}

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        results = []
        equity_curve = []
        final_models = []
        
        cols_path = MODELS_DIR / 'feature_columns.joblib'
        if not cols_path.exists():
             return {'status': 'no_columns'}
        used_cols = joblib.load(cols_path)
        
        med_path = MODELS_DIR / 'feature_medians.joblib'
        medians = joblib.load(med_path) if med_path.exists() else {}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            # Purge overlap to prevent label leakage
            if fold > 0:
                test_start = df.iloc[test_idx].scan_time.min()
                purge_cut = test_start - self.purge
                train_idx = df.index[df.scan_time <= purge_cut].to_numpy()

            if len(train_idx) < 100:
                continue

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx].copy()

            # Feature construction
            available = [c for c in used_cols if c in train_df.columns]
            X_train = train_df[available].apply(pd.to_numeric, errors="coerce").fillna(medians)
            X_test = test_df[available].apply(pd.to_numeric, errors="coerce").fillna(medians)
            
            # Ensure all required columns are present
            for c in used_cols:
                if c not in X_train.columns: X_train[c] = medians.get(c, 0.0)
                if c not in X_test.columns: X_test[c] = medians.get(c, 0.0)
            
            X_train = X_train[used_cols]
            X_test = X_test[used_cols]
            y_train = train_df['label'].astype(int)

            model = self._train_model(X_train, y_train)
            final_models.append(model)

            test_df['ml_probability'] = model.predict_proba(X_test)[:, 1]
            
            # Simulate trades
            fold_res = self._simulate_fold(test_df)
            results.append(fold_res)
            equity_curve.extend(fold_res.get('pnl_list', []))

        # Aggregate
        if not equity_curve:
            return {'status': 'no_trades_in_backtest'}
            
        sharpe = self._compute_sharpe(equity_curve)
        mdd = self._compute_mdd(equity_curve)
        
        final_report = {
            'status': 'ok',
            'n_folds': len(results),
            'total_trades': sum(f['n_trades'] for f in results),
            'total_pnl': sum(f['pnl_sum'] for f in results),
            'sharpe': round(sharpe, 3),
            'max_drawdown': round(mdd, 4),
            'fold_metrics': results
        }
        
        with open(REPORTS_DIR / "backtest_advanced_report.json", "w") as f:
            json.dump(final_report, f, indent=2, default=str)
            
        return final_report

    def _simulate_fold(self, df):
        # Implementation uses horizon return as proxy (Phase 5 will upgrade this to OHLV bars)
        trades = df[df['ml_probability'] >= self.thr].copy()
        if trades.empty:
            return {'n_trades': 0, 'pnl_sum': 0.0, 'pnl_list': []}
            
        pnl_list = trades['future_return_pct'].tolist()
        return {
            'n_trades': len(trades),
            'pnl_sum': sum(pnl_list),
            'pnl_list': pnl_list
        }

    def _compute_sharpe(self, pnl_list):
        if not pnl_list or len(pnl_list) < 2: return 0
        arr = np.array(pnl_list)
        std = arr.std()
        if std == 0: return 0
        # Annualized Sharpe (assuming cycles represent trade frequency)
        avg = arr.mean()
        return (avg / std) * np.sqrt(len(pnl_list)) 

    def _compute_mdd(self, pnl_list):
        if not pnl_list: return 0
        cum = np.cumsum(pnl_list)
        peaks = np.maximum.accumulate(cum)
        drawdowns = peaks - cum
        return float(drawdowns.max()) if len(drawdowns) > 0 else 0
