#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="$HOME/Documents/COMP8240-LTSF-Linear-Reproduction"
OFFICIAL="$HOME/Documents/LTSF-Linear"
RUN_ROOT="$PROJECT/experiments/new_data/synthetic_family_v1"
EXPECTED_SHA="0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"

echo "=== VERIFY ORIGINAL IMPLEMENTATION ==="
ACTUAL_SHA="$(git -C "$OFFICIAL" rev-parse HEAD)"

if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    echo "STOP: Unexpected original source revision."
    exit 1
fi

if ! git -C "$OFFICIAL" diff --quiet ||
   ! git -C "$OFFICIAL" diff --cached --quiet; then
    echo "STOP: Original source has tracked modifications."
    exit 1
fi

cd "$PROJECT"

CONDITIONS=(
    level_shift
    regime_change
    noise_medium
    noise_high
    outliers_low
    outliers_high
    nonlinear
)

MODELS=(Linear NLinear DLinear)

echo "=== START 21-RUN SYNTHETIC BENCHMARK ==="

for CONDITION in "${CONDITIONS[@]}"; do

    DATA_FILE="$PROJECT/data/synthetic/$CONDITION.csv"
    test -f "$DATA_FILE" || {
        echo "STOP: Missing dataset $DATA_FILE"
        exit 1
    }

    for MODEL in "${MODELS[@]}"; do
        RUN_DIR="$RUN_ROOT/$CONDITION/$MODEL"

        echo
        echo "======================================="
        echo "Condition: $CONDITION"
        echo "Model:     $MODEL"
        echo "======================================="

        if [[ -d "$RUN_DIR" ]]; then
            PRED_COUNT=0
            if [[ -d "$RUN_DIR/results" ]]; then
                PRED_COUNT="$(
                    find "$RUN_DIR/results" -name pred.npy \
                        -type f | wc -l | tr -d ' '
                )"
            fi

            if [[ -f "$RUN_DIR/run.log" ]] &&
               grep -q '^mse:' "$RUN_DIR/run.log" &&
               [[ "$PRED_COUNT" == "1" ]]; then
                echo "Completed run exists. Skipping safely."
                continue
            fi

            echo "STOP: Incomplete existing run:"
            echo "$RUN_DIR"
            echo "Preserved for investigation."
            exit 1
        fi

        mkdir -p "$RUN_DIR"

        if ! (
            cd "$RUN_DIR"

            conda run --no-capture-output \
              -n comp8240-ltsf \
              python -u "$OFFICIAL/run_longExp.py" \
              --is_training 1 \
              --model_id "COMP8240_${CONDITION}_336_96" \
              --model "$MODEL" \
              --data custom \
              --root_path "$PROJECT/data/synthetic" \
              --data_path "$CONDITION.csv" \
              --features M \
              --target channel_1 \
              --freq h \
              --seq_len 336 \
              --label_len 48 \
              --pred_len 96 \
              --enc_in 4 \
              --dec_in 4 \
              --c_out 4 \
              --batch_size 32 \
              --learning_rate 0.005 \
              --train_epochs 10 \
              --patience 3 \
              --num_workers 0 \
              --itr 1 \
              --des COMP8240_A3_SYN_V1 \
              --checkpoints "$RUN_DIR/checkpoints" \
              > run.log 2>&1
        ); then
            echo "ERROR: Training failed."
            tail -50 "$RUN_DIR/run.log"
            exit 1
        fi

        if ! grep -q '^mse:' "$RUN_DIR/run.log"; then
            echo "ERROR: Final metrics missing."
            exit 1
        fi

        PRED_COUNT="$(
            find "$RUN_DIR/results" -name pred.npy \
                -type f | wc -l | tr -d ' '
        )"

        if [[ "$PRED_COUNT" != "1" ]]; then
            echo "ERROR: Prediction output missing or ambiguous."
            exit 1
        fi

        echo "PASS:"
        grep '^mse:' "$RUN_DIR/run.log" | tail -1
    done
done

echo
echo "=== COMPLETE 21-RUN RESULTS SUMMARY ==="
printf "%-17s %-10s %s\n" "Condition" "Model" "Result"

for CONDITION in "${CONDITIONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        LOG="$RUN_ROOT/$CONDITION/$MODEL/run.log"
        SCORE="$(grep '^mse:' "$LOG" | tail -1)"

        printf "%-17s %-10s %s\n" \
            "$CONDITION" "$MODEL" "$SCORE"
    done
done

echo
echo "SYNTHETIC BATCH: COMPLETE"
