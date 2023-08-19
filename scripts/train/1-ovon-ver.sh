#!/bin/bash
#SBATCH --job-name=ovon
#SBATCH --output=slurm_logs/ovon-ver-clip_vit-%j.out
#SBATCH --error=slurm_logs/ovon-ver-clip_vit-%j.err
#SBATCH --gpus 4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 7
#SBATCH --ntasks-per-node 4
#SBATCH --constraint=a40
#SBATCH --partition=short
#SBATCH --signal=USR1@100
#SBATCH --requeue

export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export MAGNUM_LOG=quiet

MAIN_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MAIN_ADDR

source /srv/kira-lab/share4/yali30/mamba/mamba_install/etc/profile.d/conda.sh
conda deactivate
conda activate ovon_duplicate

cd /srv/kira-lab/share4/yali30/ovon_duplicate/ovon

TENSORBOARD_DIR="runs/ovon/ver/clip_vit_b_16/tb/"
CHECKPOINT_DIR="runs/ovon/ver/clip_vit_b_16/data/new_checkpoints/"
LOG_DIR="runs/ovon/ver/clip_vit_b_16/train.log"
DATA_PATH="/srv/kira-lab/share4/yali30/cow_ovon/hm3d_data/datasets/ovon_naoki/ovon/hm3d/v2"

srun python -um ovon.run \
  --run-type train \
  --exp-config config/experiments/ver_objectnav.yaml \
  habitat_baselines.trainer_name="ver" \
  habitat_baselines.num_environments=32 \
  habitat_baselines.rl.policy.name=PointNavResNetCLIPPolicy \
  habitat_baselines.rl.ddppo.train_encoder=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.tensorboard_dir=${TENSORBOARD_DIR} \
  habitat_baselines.checkpoint_folder=${CHECKPOINT_DIR} \
  habitat_baselines.log_file=${LOG_DIR} \
  habitat.dataset.data_path=${DATA_PATH}/train/train.json.gz \
  +habitat/task/lab_sensors@habitat.task.lab_sensors.clip_objectgoal_sensor=clip_objectgoal_sensor \
  ~habitat.task.lab_sensors.objectgoal_sensor \
  habitat.task.lab_sensors.clip_objectgoal_sensor.cache=/srv/kira-lab/share4/yali30/ovon_duplicate/ovon/ovon_stretch_cache_naoki.pkl \
  habitat.task.measurements.success.success_distance=0.25 \
  habitat.dataset.type="OVON-v1" \
  habitat.task.measurements.distance_to_goal.type=OVONDistanceToGoal \
  habitat.simulator.type="OVONSim-v0" \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.ddppo.backbone=resnet50_clip_avgpool \
  habitat_baselines.rl.policy.clip_model=ViT-B/16