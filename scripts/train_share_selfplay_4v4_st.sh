#!/bin/sh

env="MultipleCombat"
scenario="4v4/ShootMissile/HierarchySelfplay"
algo="st"
exp="v1"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 16 --cuda --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 1000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 8192 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 1 --eval-episodes 1 \
    --user-name "jyh"  --wandb-name "thu_jsbsim" \
     --embed-dim 128 --num-spatial-heads 8  --num-temporal-heads 8 \
    --temporal-ff-dim 512 --num-temporal-layers 2  --memory-length 64 --dropout 0.1  --ego-dim 9 --relative-dim 6 \
    --num-friendly 3 --num-enemy 4 --num-missiles 1  --use-type-embedding  --data-chunk-length-st 64
