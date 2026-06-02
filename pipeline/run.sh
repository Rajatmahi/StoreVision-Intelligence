#!/bin/bash
# run.sh - Runs the complete detection pipeline

echo "Starting Store Intelligence Detection Pipeline"
echo "----------------------------------------------"

# Run unified detection script (line crossing + zone analytics)
echo "Processing clips and emitting events..."
export PYTHONPATH=$(pwd)
python pipeline/detect.py

echo "Pipeline execution completed."
