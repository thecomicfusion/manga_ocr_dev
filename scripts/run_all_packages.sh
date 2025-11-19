#!/usr/bin/env bash

for pkg in {0..10}; do
  python ./synthetic_data_generator_ko/run_generate_ko.py --package="$pkg" --n_random=10000 --max_workers=10
  if [ "$pkg" -ne 10 ]; then
    sleep 3devm
  fi
done