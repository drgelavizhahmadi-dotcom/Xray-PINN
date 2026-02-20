#!/bin/bash
source .venv/bin/activate
python -m mdss_uncertainty.batch_processor
echo "Batch complete. Results in data/results/"
