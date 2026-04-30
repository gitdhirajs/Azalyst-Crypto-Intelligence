import json
import shutil
from pathlib import Path

# Mock config
REPORTS_DIR = Path(r"c:\Users\Administrator\Downloads\files\Azalyst-Crypto-Scanner\reports")

def inject_educational_frames(signals):
    manifest_path = Path(r"D:\Azalyst Bernd Skorupinski\_audit\manifest.json")
    if not manifest_path.exists():
        print("Manifest not found")
        return

    frames_dir = REPORTS_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    lessons = manifest.get("lessons", [])
    
    for s in signals:
        keywords = ["Candle", "Zone"]
        if s.get("direction") == "LONG":
            keywords.append("Demand")
        else:
            keywords.append("Supply")
            
        matching_frame = None
        for L in lessons:
            if any(k.lower() in L.get("rel_path", "").lower() for k in keywords):
                frs = L.get("frames", [])
                if frs:
                    pick = frs[len(frs)//2]
                    src = Path(L.get("abs_dir")) / pick["file"]
                    if src.exists():
                        dest_name = f"{s['symbol']}_edu.jpg"
                        dest = frames_dir / dest_name
                        shutil.copy2(src, dest)
                        matching_frame = f"reports/frames/{dest_name}"
                        print(f"Injected frame for {s['symbol']}: {matching_frame}")
                        break
        
        if matching_frame:
            s["edu_frame"] = matching_frame

# Test signals
test_signals = [
    {"symbol": "BTCUSDT", "direction": "LONG"},
    {"symbol": "ETHUSDT", "direction": "SHORT"}
]

inject_educational_frames(test_signals)
print(json.dumps(test_signals, indent=2))
