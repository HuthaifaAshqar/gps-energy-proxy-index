#!/usr/bin/env bash
set -e
for s in 01_prepare_segments 02_tune_kalman 03_smooth_and_features 04_validation; do python scripts/$s.py; done
python scripts/05_montecarlo.py all
python scripts/06_regimes_spatial_vehicle.py
python scripts/07_figures.py
