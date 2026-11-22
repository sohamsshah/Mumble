"""Start/stop the ScreenCaptureKit source twice (no frames kept) to surface start errors."""
import sys, time, traceback
sys.path.insert(0, ".")
from vigil.capture import sck
from vigil.core import SessionClock
sck._ensure_cg_session()
wins = sck.list_windows()
print("windows:", [(w.window_id, w.app, w.title) for w in wins[:4]], flush=True)
for attempt in (1, 2):
    src = sck.SckWindowSource(window_id=wins[0].window_id)
    n = {"f": 0, "a": 0}
    try:
        src.start(SessionClock(), lambda f: n.__setitem__("f", n["f"] + 1), lambda a: n.__setitem__("a", n["a"] + 1))
        time.sleep(2)
        src.stop()
        print(f"attempt {attempt}: OK frames={n['f']} audio={n['a']} stats={src.stats} err={src.last_error}", flush=True)
    except Exception:
        print(f"attempt {attempt}: FAILED", flush=True); traceback.print_exc()
