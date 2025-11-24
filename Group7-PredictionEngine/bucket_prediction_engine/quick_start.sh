#!/bin/bash
# Quick start script for bucket prediction engine

echo "=============================================="
echo "Bucket Prediction Engine - Quick Start"
echo "=============================================="
echo ""

# Check if XES file provided
if [ -z "$1" ]; then
    echo "Usage: ./quick_start.sh <path_to_xes_file> [max_cases]"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh ../HospitalBilling-EventLog_1_all/HospitalBilling-EventLog.xes 1000"
    echo "  ./quick_start.sh my_event_log.xes"
    exit 1
fi

XES_FILE=$1
MAX_CASES=${2:-5000}

echo "📂 XES File: $XES_FILE"
echo "📊 Max cases: $MAX_CASES"
echo ""

# Check if file exists
if [ ! -f "$XES_FILE" ]; then
    echo "❌ Error: File not found: $XES_FILE"
    exit 1
fi

# Check dependencies
echo "🔍 Checking dependencies..."
if ! python -c "import pm4py" 2>/dev/null; then
    echo "⚠️  pm4py not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Running quick test..."
echo ""

python quick_test.py "$XES_FILE" "$MAX_CASES"

echo ""
echo "=============================================="
echo "✅ Quick test complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Train on full dataset:"
echo "   python train_models.py $XES_FILE"
echo ""
echo "2. Train with custom parameters:"
echo "   python train_models.py $XES_FILE --max-bucket 15 --epochs 20"
echo ""
echo "3. Train only Phase 1 (outcome prediction):"
echo "   python train_models.py $XES_FILE --phase1-only"
echo ""
echo "4. Train only Phase 2 (bucket models):"
echo "   python train_models.py $XES_FILE --phase2-only --epochs 15"
echo ""
