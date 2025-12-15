#!/usr/bin/env bash

for pkg in {80..100}; do
  python ./synthetic_data_generator_ko/run_generate_ko.py --package="$pkg" --n_random=100 --max_workers=10 > /dev/null 2>&1
  if [ "$pkg" -ne 100 ]; then
    sleep 2m
  fi
done