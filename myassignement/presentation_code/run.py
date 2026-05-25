# -*- coding: utf-8 -*-
"""
Presentation entry point — choose a lap mode.

Set MODE below and run `python run.py`. Each lap module is fully
self-contained (owns its own radio connection, threads, and emergency-stop
listener) — this file is just a dispatcher.

  vision                  → IBVS + centre triangulation
  position                → known gate poses from gates_info.csv
  vision_no_triangulation → IBVS only (search → approach → transit)
"""

MODE = "vision"   # "vision" | "position" | "vision_no_triangulation"


if __name__ == '__main__':
    if MODE == "vision":
        import vision_triangulation_lap
        vision_triangulation_lap.main()
    elif MODE == "position":
        import position_lap
        position_lap.main()
    elif MODE == "vision_no_triangulation":
        import vision_no_triangulation_lap
        vision_no_triangulation_lap.main()
    else:
        raise ValueError(
            f"Unknown MODE: {MODE!r} "
            "(expected 'vision', 'position', or 'vision_no_triangulation')"
        )
