#!/bin/bash
# Double-click this file on Mac to start the ORB signal generator
cd "$(dirname "$0")"
echo "Starting ORB Signal Generator..."
echo "Risk: 3% per instrument | Account: \$100,000"
echo ""
python3 signal_generator.py --risk=3
echo ""
echo "Session complete. Press any key to close."
read -n 1
