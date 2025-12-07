#!/usr/bin/env bash

for pkg in {1..100}; do
  python ./synthetic_data_generator_ko/run_generate_ko.py --package="$pkg" --n_random=10000 --max_workers=12 > /dev/null 2>&1
  if [ "$pkg" -ne 100 ]; then
    sleep 2m
  fi
done