exp_name='gpu_001'

python train.py \
    --dataset data/kagu  \
    --name "$exp_name"_train \
    --p_device gpu \
    --p_n_gpus 4 \
    \
    --backbone inception \
    --backbone_pretrained True \
    --backbone_frozen True \
    --predictor lstm \
    --dropout 0.4 \
    --teacher True \
    \
    --lr 1e-8 \
    \
    --dbg \
    --tb \


# Check the exit code of the first script
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit $?
fi


