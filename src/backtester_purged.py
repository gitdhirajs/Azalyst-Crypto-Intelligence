"""
Purged walk-forward backtester with dynamic model retraining, including OOS R².
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.config import FEATURES_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR

log = logging.getLogger("azalyst.backtest_purged")


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
        path = FEATURES_DIR / "latest_features.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "scan_time" not in df.columns:
            return pd.DataFrame()
        df["scan_time"] = pd.to_datetime(df["scan_time"], utc=True)
        df = df.sort_values("scan_time").reset_index(drop=True)
        df = df.dropna(subset=["label"])
        return df

    def _train_model(self, X, y):
        model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                              eval_metric="logloss", random_state=42, tree_method="hist")
        model.fit(X, y)
        return model

    def run(self) -> Dict:
        df = self.prepare_data()
        if df.empty:
            return {"status": "no_data"}

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        results = []
        equity_curve = []
        all_y_true = []
        all_y_pred = []

        cols_path = MODELS_DIR / "feature_columns.joblib"
        if not cols_path.exists():
            return {"status": "no_columns"}
        used_cols = joblib.load(cols_path)

        med_path = MODELS_DIR / "feature_medians.joblib"
        medians = joblib.load(med_path) if med_path.exists() else {}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            if fold > 0:
                test_start = df.iloc[test_idx].scan_time.min()
                purge_cut = test_start - self.purge
                train_idx = df.index[df.scan_time <= purge_cut].to_numpy()

            if len(train_idx) < 100:
                continue

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx].copy()

            available = [c for c in used_cols if c in train_df.columns]
            X_train = train_df[available].apply(pd.to_numeric, errors="coerce").fillna(medians)
            X_test = test_df[available].apply(pd.to_numeric, errors="coerce").fillna(medians)

            for c in used_cols:
                if c not in X_train.columns:
                    X_train[c] = medians.get(c, 0.0)
                if c not in X_test.columns:
                    X_test[c] = medians.get(c, 0.0)

            X_train = X_train[used_cols]
            X_test = X_test[used_cols]
            y_train = train_df["label"].astype(int)

            model = self._train_model(X_train, y_train)
            probas = model.predict_proba(X_test)[:, 1]
            test_df["ml_probability"] = probas

            trades = test_df[test_df["ml_probability"] >= self.thr]
            if not trades.empty:
                pnl = trades["future_return_pct"].values / 100.0
                equity_curve.extend(pnl.tolist())
                all_y_true.extend(trades["future_return_pct"].values / 100.0)
                all_y_pred.extend(trades["ml_probability"].values * 0.005)

            fold_res = {
                "fold": fold,
                "n_trades": len(trades),
                "pnl_sum": float(trades["future_return_pct"].sum()) if not trades.empty else 0.0,
            }
            results.append(fold_res)

        if not equity_curve:
            return {"status": "no_trades"}

        sharpe = self._compute_sharpe(equity_curve)
        mdd = self._compute_mdd(equity_curve)
        oos_r2 = r2_score(all_y_true, all_y_pred) if len(all_y_true) > 1 else 0.0

        total_trades = sum(r["n_trades"] for r in results)
        final_report = {
            "status": "ok",
            "n_splits": len(results),
            "n_trades": total_trades,
            "win_rate": (np.array(equity_curve) > 0).mean() if equity_curve else 0.0,
            "profit_factor": self._profit_factor(equity_curve),
            "sharpe_annualized": round(sharpe, 3),
            "max_drawdown_pct": round(mdd * 100, 4),
            "total_return_pct": round(np.sum(equity_curve) * 100, 4),
            "oos_r2": round(oos_r2, 4),
            "folds": results,
        }

        with open(REPORTS_DIR / "backtest_purged.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, default=str)

        with open(LOGS_DIR / "backtest_purged_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**final_report, "timestamp": datetime.now(timezone.utc).isoformat()}, default=str) + "\n")

        return final_report

    def _compute_sharpe(self, pnl_list):
        if len(pnl_list) < 2:
            return 0
        arr = np.array(pnl_list)
        return np.mean(arr) / np.std(arr) * np.sqrt(len(pnl_list)) if np.std(arr) > 0 else 0

    def _compute_mdd(self, pnl_list):
        if not pnl_list:
            return 0
        cum = np.cumsum(pnl_list)
        peaks = np.maximum.accumulate(cum)
        return max(peaks - cum) / peaks.max() if peaks.max() > 0 else 1.0

    def _profit_factor(self, pnl_list):
        wins = [x for x in pnl_list if x > 0]
        losses = [abs(x) for x in pnl_list if x < 0]
        if not losses:
            return 999.0 if wins else 0.0
        return sum(wins) / sum(losses) if losses and sum(losses) != 0 else 0.0
