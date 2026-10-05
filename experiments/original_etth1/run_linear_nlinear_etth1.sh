#!/usr/bin/env bash
set -euo pipefail

OFFICIAL_REPO="${OFFICIAL_REPO:-$HOME/Documents/LTSF-Linear}"
PROJECT_REPO="${PROJECT_REPO:-$HOME/Documents/COMP8240-LTSF-Linear-Reproduction}"
LOG_DIR="$PROJECT_REPO/experiments/original_etth1/logs"

EXPECTED_COMMIT="0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
EXPECTED_DATA_HASH="f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"

mkdir -p "$LOG_DIR"
cd "$OFFICIAL_REPO"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "ERROR: unexpected LTSF-Linear commit: $ACTUAL_COMMIT"
    exit 1
fi

ACTUAL_HASH="$(shasum -a 256 dataset/ETTh1.csv | awk '{print $1}')"
if [[ "$ACTUAL_HASH" != "$EXPECTED_DATA_HASH" ]]; then
    echo "ERROR: ETTh1 hash mismatch: $ACTUAL_HASH"
    exit 1
fi

for MODEL in Linear NLinear; do
    for HORIZON in 96 192 336 720; do

        MODEL_LOWER="$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')"
        LOG="$LOG_DIR/${MODEL_LOWER}_etth1_h${HORIZON}_mac_arm64.log"

        echo "============================================================"
        echo "Running model=$MODEL horizon=$HORIZON"
        echo "Log: $LOG"
        echo "============================================================"

        python -u run_longExp.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path ETTh1.csv \
          --model_id "ETTh1_336_${HORIZON}" \
          --model "$MODEL" \
          --data ETTh1 \
          --features M \
          --seq_len 336 \
          --label_len 48 \
          --pred_len "$HORIZON" \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --moving_avg 25 \
          --des Exp \
          --itr 1 \
          --train_epochs 10 \
          --batch_size 32 \
          --patience 3 \
          --learning_rate 0.005 \
          --num_workers 0 \
          2>&1 | tee "$LOG"

        echo
        grep -Ei 'mse:.*mae:|mae:.*mse:' "$LOG" || {
            echo "ERROR: final metric line missing from $LOG"
            exit 1
        }

        echo
    done
done

echo "All Linear and NLinear ETTh1 runs completed successfully."
