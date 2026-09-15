"""Which learned weights moved furthest from the hand-tuned prior."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from engine.weights import Weights

a = Weights.default().to_dict()
b = Weights.load(sys.argv[1] if len(sys.argv) > 1 else "backend/weights_learned.json").to_dict()
rows = sorted(((b[k] - a[k], k) for k in a if k in b), key=lambda r: -abs(r[0]))
print(f"{'feature':26s}{'prior':>9s}{'learned':>10s}{'change':>10s}")
for delta, k in rows[:20]:
    print(f"{k:26s}{a[k]:>9.3f}{b[k]:>10.3f}{delta:>+10.3f}")