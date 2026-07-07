#!/bin/bash
# Run this once v28 finishes training
CHECKPOINT="checkpoints/phase1_lidar_v28/final.pt"
TRACKS="aut_tight esp_tight gbr_tight mco_tight spielberg_tight silverstone_tight monza_tight sochi_tight ims_tight zandvoort_tight sepang_tight yasmarina_tight budapest_tight melbourne_tight"

echo "=== v28 Evaluation Across All 14 Tracks ==="
for track in $TRACKS; do
    echo ""
    echo "--- $track ---"
    python3 -m f1tenth_research.eval.quick_eval \
        --config f1tenth_research/configs/rma_config.yaml \
        --checkpoint "$CHECKPOINT" \
        --track "$track" --num_episodes 5 2>&1 | grep -E "Summary|avg_length|avg_reward|Centerline"
done
