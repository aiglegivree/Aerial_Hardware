# -*- coding: utf-8 -*-
"""
Presentation entry point — choose vision or position lap.

Set MODE below and run `python run.py`. Each lap module is fully
self-contained (owns its own radio connection, threads, and emergency-stop
listener) — this file is just a dispatcher.
"""

MODE = "vision"   # "vision" or "position"


if __name__ == '__main__':
    if MODE == "vision":
        import vision_triangulation_lap
        vision_triangulation_lap.main()
    elif MODE == "position":
        import position_lap
        position_lap.main()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (expected 'vision' or 'position')")
