#!/bin/sh

#SBATCH --partition=cpu_compute       # 指定分区
#SBATCH --ntasks=1                    # 请求1个任务                
#SBATCH --cpus-per-task=32          # 请求每个任务36个CPU核

#SBATCH --output=job_output_%j.log    # 标准输出将保存到 job_output_<job_id>.log 文件中
echo "--- Loading Conda Environment ---"
source /public/home/test202301/miniconda3/etc/profile.d/conda.sh
conda activate lag_hyx
echo "--- Conda Environment Loaded ---"

echo "--- Starting Training Script ---"
./train_share_selfplay_2v2_st.sh
echo "--- Training Script Finished ---"
