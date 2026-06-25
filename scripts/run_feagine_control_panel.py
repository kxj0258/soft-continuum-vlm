from __future__ import annotations

import sys
import time
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from soft_continuum_vlm.ui.tk_control_panel import TkFeagineControlPanel


def main() -> int:
    try:
        panel = TkFeagineControlPanel()
    except Exception as exc:
        print(f"[FAIL] failed to open control panel: {exc}")
        return 1

    autoclose_seconds = float(os.environ.get("FEAGINE_PANEL_AUTOCLOSE_SECONDS", "0") or "0")
    start_time = time.monotonic()

    try:
        while not panel.should_quit():
            panel.update()
            print(f"[ACTION] {panel.action()}")
            if autoclose_seconds > 0.0 and time.monotonic() - start_time >= autoclose_seconds:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        panel.close()

    print("[OK] control panel closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
