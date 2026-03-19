#!/bin/bash
# Simple script to run SleepEEG to Epilepsy transfer learning experiment

echo "🔬 TF-C Transfer Learning: SleepEEG to Epilepsy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if datasets exist
if [ ! -d "src/tfc/data/SleepEEG" ] || [ ! -d "src/tfc/data/Epilepsy" ]; then
    echo "⚠️  Datasets not found. Running download script..."
    bash download_datasets.sh
    echo "✅ Datasets downloaded!"
fi

# Step 1: Pretraining on SleepEEG
echo ""
echo "📚 Step 1: Pretraining on SleepEEG dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uv run python -m src.tfc.main \
    --run_description=pretrain \
    --seed=42 \
    --training_mode=pre_train \
    --pretrain_dataset=SleepEEG \
    --target_dataset=Epilepsy \
    --sourcedata_path=./src/tfc/data/SleepEEG \
    --targetdata_path=./src/tfc/data/Epilepsy \
    --logs_save_dir=./src/tfc/experiments_logs \
    --device=cpu

if [ $? -ne 0 ]; then
    echo "❌ Pretraining failed!"
    exit 1
fi

echo "✅ Pretraining completed!"

# Step 2: Fine-tuning on Epilepsy
echo ""
echo "🎯 Step 2: Fine-tuning on Epilepsy dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

uv run python -m src.tfc.main \
    --run_description=finetune \
    --seed=42 \
    --training_mode=fine_tune_test \
    --pretrain_dataset=SleepEEG \
    --target_dataset=Epilepsy \
    --sourcedata_path=./src/tfc/data/SleepEEG \
    --targetdata_path=./src/tfc/data/Epilepsy \
    --logs_save_dir=./src/tfc/experiments_logs \
    --device=cpu

if [ $? -ne 0 ]; then
    echo "❌ Fine-tuning failed!"
    exit 1
fi

echo ""
echo "✨ Experiment completed successfully!"
echo "📊 Results saved in: ./src/tfc/experiments_logs/"