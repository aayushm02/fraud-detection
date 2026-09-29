"""Launch the fraud detection API server and optionally the dashboard.

Usage:
    python scripts/serve.py                    # Start API on :8000
    python scripts/serve.py --dashboard        # Also start Streamlit dashboard
"""
import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='Launch fraud detection services')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API host')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--dashboard', action='store_true', help='Also launch Streamlit dashboard')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    dashboard_proc = None
    
    if args.dashboard:
        print('Starting Streamlit dashboard on port 8501...')
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'src', 'fraud_detection', 'dashboard', 'app.py'
        )
        dashboard_proc = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', dashboard_path,
             '--server.port', '8501', '--server.headless', 'true'],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    
    try:
        import uvicorn
        print(f'Starting FastAPI server on {args.host}:{args.port}...')
        uvicorn.run(
            'fraud_detection.api.app:app',
            host=args.host,
            port=args.port,
            reload=args.reload
        )
    finally:
        if dashboard_proc:
            dashboard_proc.terminate()


if __name__ == '__main__':
    main()
