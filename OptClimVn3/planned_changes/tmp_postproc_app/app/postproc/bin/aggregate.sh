#!/bin/bash
# Post-processing aggregation wrapper
# Modify line below to setup your Python environment

. /path/to/venv/bin/activate  # <-- MODIFY THIS

python "$OPTCLIMTOP/scripts/aggregate.py" --config "$AGGREGATE_CONFIG" --output "$OUTPUT_FILE"
