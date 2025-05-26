cd /data/mmh/DeepSeek-V3/inference

export NCCL_DEBUG=WARNING
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_DESYNC_DEBUG=1
# export NCCL_DESYNC_DEBUG=1


# fix bug: the nccl works with docker in interactivate shell, but has problems in non-interactivate mode, after diff the environment variables, we should do this things
# copied from  /etc/bash.bashrc in container.
source /etc/profile.d/env.sh
source /opt/dtk-24.04.3/env.sh
source /opt/dtk/env.sh
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_LAUNCH_MODE=GROUP
export NCCL_MAX_NCHANNELS=128
export NCCL_MIN_NCHANNELS=128
export NCCL_P2P_LEVEL=SYS
# end copied from /etc/bash.bashrc


torchrun --nnode=16 --nproc_per_node=4 --master-addr="10.18.17.159" --master-port=1234 --node_rank=$1 multi_node_test.py \
	--ckpt-path /data/mmh/ds-v3-pp --config-path /data/mmh/DeepSeek-V3/inference/configs/config_671B.json
