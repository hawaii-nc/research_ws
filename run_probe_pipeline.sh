#!/bin/bash
# run_probe_pipeline.sh — one-command, fully automated Steps 1-3.
#
# Usage (from the repo root, i.e. the directory containing f1tenth_research/):
#     bash f1tenth_research/probe_eval/run_probe_pipeline.sh
#
# Edit the variables below once; everything else is hands-off.
# Each step writes JSON + PNG into $OUTDIR and prints a VERDICT line —
# grep VERDICT at the end for the summary.

set -euo pipefail

# ----------------------------- EDIT THESE -----------------------------------
AC_CKPT="checkpoints/phase1_lidar_v29/final.pt"
PHI_CKPT="checkpoints/phase2/final_v29_window100.pt"
CONFIG="f1tenth_research/configs/rma_config.yaml"
WINDOW=100                                           # phi history window (10 or 100)
EPISODES=150                                         # rollout episodes to collect
OUTDIR="probe_results/$(date +%Y%m%d_%H%M%S)_w${WINDOW}"
# -----------------------------------------------------------------------------

DATASET="${OUTDIR}/probe_dataset_w${WINDOW}.npz"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/pipeline.log"
exec > >(tee -a "${LOG}") 2>&1

echo "=== probe pipeline: window=${WINDOW}, episodes=${EPISODES} ==="
echo "=== output dir: ${OUTDIR} ==="

echo -e "\n=== STEP 1: decode ceiling (probes on true z = mu(e), no rollouts) ==="
python3 -m f1tenth_research.probe_eval.step1_ceiling_probe \
    --actor_critic "${AC_CKPT}" \
    --config "${CONFIG}" \
    --n_samples 50000 \
    --outdir "${OUTDIR}/step1"

echo -e "\n=== STEP 0: collect rollout dataset (shared by Steps 2 & 3) ==="
python3 -m f1tenth_research.probe_eval.collect_probe_dataset \
    --actor_critic "${AC_CKPT}" \
    --phase2 "${PHI_CKPT}" \
    --config "${CONFIG}" \
    --episodes "${EPISODES}" \
    --window "${WINDOW}" \
    --policy_z oracle \
    --out "${DATASET}"

echo -e "\n=== STEP 2: observability (fresh supervised net history -> grip) ==="
python3 -m f1tenth_research.probe_eval.step2_observability \
    --dataset "${DATASET}" \
    --window "${WINDOW}" \
    --outdir "${OUTDIR}/step2"

echo -e "\n=== STEP 3: E5 probe on phi's latent (EMA + excitation-stratified) ==="
python3 -m f1tenth_research.probe_eval.step3_excitation_probe \
    --dataset "${DATASET}" \
    --outdir "${OUTDIR}/step3"

echo -e "\n=== PIPELINE COMPLETE — verdicts: ==="
grep -h "VERDICT" "${LOG}" || true
echo "All outputs in ${OUTDIR}/ (JSON + PNG per step, full log in pipeline.log)"
