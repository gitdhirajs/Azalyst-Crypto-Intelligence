import logging
import sys
from src.pipeline import run_scheduled_pipeline

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

if __name__ == "__main__":
    print("Starting Institutional Scan Cycle...")
    results = run_scheduled_pipeline(
        train_main_model=True,  # Retrain with new features
        train_hourly_pattern_model=False,
        force_main=True,
        show_progress=True,
        run_engines=True,
        run_backtest=True
    )
    print("\nInstitutional Scan Complete!")
    print(f"Signals Found: {results['fused_signal_count']}")
    print(f"Tier-A Signals: {results['tier_a_count']}")
    if results['backtest']:
        print(f"Backtest Win Rate: {results['backtest'].get('win_rate', 0)*100:.1f}%")
