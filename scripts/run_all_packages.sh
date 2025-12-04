#!/usr/bin/env bash

for pkg in {108..110}; do
  python ./synthetic_data_generator_ko/run_generate_ko.py --package="$pkg" --n_random=10000 --max_workers=7 > /dev/null 2>&1
  if [ "$pkg" -ne 110 ]; then
    sleep 3m
  fi
done