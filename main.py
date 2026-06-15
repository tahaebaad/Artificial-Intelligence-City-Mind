"""
CityMind  -  Main Runner  (main.py)
====================================
Entry point for the CityMind project.

Usage:
  python main.py              — runs the 20-step simulation in the terminal
  python main.py --gui        — opens the CustomTkinter GUI
  python main.py --help       — shows usage

This file counts as 1 of the 10 allowed files.
"""

import sys
import traceback


def run_terminal():
    """Run the 20-step simulation and print results to terminal."""
    from simulation_engine import SimulationEngine

    engine = SimulationEngine(rows=10, cols=10, seed=42)
    logs = engine.run(steps=20)

    print("\n" + "=" * 60)
    print("  CityMind — Full Simulation Log")
    print("=" * 60)
    for line in logs:
        print("  -", line)
    print("\nSimulation complete.")


def run_gui():
    """Launch the CustomTkinter GUI with clear startup errors."""
    try:
        from citymind_gui import CityMindGUI
    except ModuleNotFoundError as exc:
        if exc.name == "customtkinter":
            print("[GUI ERROR] Missing dependency: customtkinter")
            print("Install it with:")
            print("  pip install customtkinter")
            raise SystemExit(1)
        raise

    try:
        app = CityMindGUI()
        app.mainloop()
    except Exception as exc:
        print("[GUI ERROR] Failed to start CityMind GUI.")
        print("Reason:", exc)
        print("\nDetailed traceback:\n")
        traceback.print_exc()
        raise SystemExit(1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        run_gui()
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("CityMind — Urban Intelligence System")
        print("Usage:")
        print("  python main.py          Run 20-step terminal simulation")
        print("  python main.py --gui    Open the CustomTkinter GUI")
    else:
        run_terminal()


if __name__ == "__main__":
    main()
