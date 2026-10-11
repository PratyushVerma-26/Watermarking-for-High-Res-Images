"""
run_dashboard.py
================
Cross-platform launcher for the Streamlit Watermarking Dashboard.
Ensures correct Python environment, encoding flags, and port configuration.
"""

import sys
import os
import subprocess

def main():
    # Set UTF-8 encoding for Windows terminals
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"
    
    # Path to app.py
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Launch Streamlit via python -m streamlit run app.py
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless=false"
    ]
    
    print("=" * 60)
    print("🚀 Starting Adaptive Watermarking Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error starting dashboard: {e}")

if __name__ == "__main__":
    main()
