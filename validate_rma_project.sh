#!/bin/bash
# Validation script: checks that RMA project structure is complete

set -e

echo "=========================================="
echo "F1Tenth RMA Project Structure Validation"
echo "=========================================="
echo ""

PROJECT_ROOT="/research_ws/src/f1tenth_research/f1tenth_research"

# Check directories exist
echo "[1/4] Checking directory structure..."
required_dirs=(
    "envs"
    "experts"
    "models"
    "training"
    "eval"
    "gazebo_deployment"
    "configs"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$PROJECT_ROOT/$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ (MISSING)"
        exit 1
    fi
done

# Check key files exist
echo ""
echo "[2/4] Checking key Python files..."
required_files=(
    "envs/__init__.py"
    "envs/randomization.py"
    "envs/f1tenth_env.py"
    "envs/reward.py"
    "experts/__init__.py"
    "models/__init__.py"
    "training/__init__.py"
    "training/phase1_ppo_il.py"
    "training/phase2_adaptation.py"
    "eval/__init__.py"
    "eval/generalization_sweep.py"
    "gazebo_deployment/__init__.py"
    "gazebo_deployment/rma_deployment_node.py"
    "configs/rma_config.yaml"
)

for file in "${required_files[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        exit 1
    fi
done

# Check documentation
echo ""
echo "[3/4] Checking documentation..."
doc_files=(
    "/research_ws/README_RMA.md"
    "/research_ws/QUICKSTART_RMA.md"
    "/research_ws/BUILD_SUMMARY.md"
)

for file in "${doc_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $(basename $file)"
    else
        echo "  ✗ $(basename $file) (MISSING)"
        exit 1
    fi
done

# Quick Python syntax check
echo ""
echo "[4/4] Checking Python syntax..."
python_files=(
    "$PROJECT_ROOT/envs/randomization.py"
    "$PROJECT_ROOT/envs/f1tenth_env.py"
    "$PROJECT_ROOT/envs/reward.py"
    "$PROJECT_ROOT/experts/__init__.py"
    "$PROJECT_ROOT/models/__init__.py"
    "$PROJECT_ROOT/training/phase1_ppo_il.py"
    "$PROJECT_ROOT/training/phase2_adaptation.py"
    "$PROJECT_ROOT/eval/generalization_sweep.py"
    "$PROJECT_ROOT/gazebo_deployment/rma_deployment_node.py"
)

for file in "${python_files[@]}"; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo "  ✓ $(basename $file)"
    else
        echo "  ✗ $(basename $file) (SYNTAX ERROR)"
        python3 -m py_compile "$file"
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo ""
echo "Project is ready for training."
echo ""
echo "Quick start:"
echo "  cd $PROJECT_ROOT"
echo "  python training/phase1_ppo_il.py --config configs/rma_config.yaml --device cuda --debug"
echo ""
