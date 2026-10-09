#!/bin/bash
# AortaVision 3D: ML Metrics Improvement Pipeline
# Quick-start script for executing the nested repeated-CV framework
#
# Usage:
#   bash RUN_PIPELINE.sh          # Full run (10 repeats, 3 inner folds, 50 Optuna trials)
#   bash RUN_PIPELINE.sh quick    # Quick test (1 repeat, 2 folds, 0 trials)
#   bash RUN_PIPELINE.sh partial  # Partial run (2 repeats, 3 folds, 20 trials)
#   bash RUN_PIPELINE.sh debug    # Debug mode with verbose logging

set -e

cd "$(dirname "$0")" || exit 1

MODE=${1:-full}

case "$MODE" in
  quick)
    echo "=== Quick Smoke Test (1 repeat, 2 inner folds, no Optuna) ==="
    echo "Expected runtime: 2–5 minutes"
    echo "Purpose: Validate all components without heavy computation"
    python model/notebooks/improve_metrics.py --repeats 1 --inner-folds 2 --trials 0
    ;;
  partial)
    echo "=== Partial Run (2 repeats, 3 inner folds, 20 Optuna trials) ==="
    echo "Expected runtime: 30–60 minutes"
    echo "Purpose: Quick assessment of method ranking"
    python model/notebooks/improve_metrics.py --repeats 2 --inner-folds 3 --trials 20
    ;;
  debug)
    echo "=== Debug Mode (1 repeat, 2 inner folds, verbose logging) ==="
    echo "Expected runtime: 2–5 minutes"
    echo "Purpose: Troubleshoot issues with detailed logging"
    python model/notebooks/improve_metrics.py --repeats 1 --inner-folds 2 --trials 0 -v
    ;;
  full)
    echo "=== Full Nested Repeated-CV Pipeline ==="
    echo "Parameters: 10 repeats, 5 outer folds, 3 inner folds, 50 Optuna trials"
    echo "Expected runtime: 6–12 hours (depends on CPU cores)"
    echo "Outputs: metrics/improvement_comparison.csv, metrics/improvement_report.md"
    echo ""
    read -p "Proceed? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      python model/notebooks/improve_metrics.py --repeats 10 --inner-folds 3 --trials 50
    else
      echo "Cancelled."
      exit 1
    fi
    ;;
  *)
    echo "Usage: bash RUN_PIPELINE.sh [quick|partial|debug|full]"
    echo ""
    echo "  quick   - Smoke test all components (2–5 min)"
    echo "  partial - 2 repeats, 20 trials (30–60 min)"
    echo "  debug   - Verbose logging for troubleshooting"
    echo "  full    - Full pipeline (default, 6–12 hours)"
    exit 1
    ;;
esac

echo ""
echo "=== Pipeline Complete ==="
echo "Review results:"
echo "  - metrics/improvement_report.md (summary & decisions)"
echo "  - metrics/improvement_comparison.csv (all-methods detail)"
