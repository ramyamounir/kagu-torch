exp_name='slurm_001'

python train.py \
    --dataset data/kagu  \
    --name "$exp_name"_train \
    --p_name epic
    --p_device slurm \
    --p_partition general \
    --p_n_nodes 1 \
    --p_n_gpus 4 \
    --p_n_cpus 2 \
    --p_ram 32 \
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


