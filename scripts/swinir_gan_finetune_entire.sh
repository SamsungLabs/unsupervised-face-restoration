#!/bin/bash

# NOTE: This will cause shell to exit if any command fails (equivalent to using &&)
# It's generally good practice to put this at the start of your shell scripts.
set -e 
# set -x

# sourcing bashrc
. /group-volume/Super-Resolution/Tianshu-Contents/configs/.bashrc
base_dir="/group-volume/Super-Resolution/Tianshu-Contents/code_cleanup/code-release"

run_commands() {
    # activate conda environment, cd to run dir, and run python script with args
    conda activate nafnet
    cd $base_dir

    local args=(
        --in_path data/celeba-raw-noise-4x-iso-1500/train_lq 
        --out_path data/celeba-raw-noise-4x-iso-1500/test_target_script
        --started_timesteps 150 
        --apply_filter 
        --reg_end_timesteps 90 
        --down_N 16 
        --config_path configs/targets/iddpm_ffhq512_swinir_gan.yaml 
        --swinir 
    )
    # python -m debugpy --listen localhost:5678 --wait-for-client generate_pseudo_targets.py "${args[@]}"
    python generate_pseudo_targets.py "${args[@]}"



    local args=(
        --cfg_path configs/main/swinir_gan.yaml 
        --save_dir logs/swinir_gan_finetune 
    )

    # python -m debugpy --listen localhost:5678 --wait-for-client train.py "${args[@]}"
    python train.py "${args[@]}"



    # specifcy paths
    gt_path="data/CelebA-Test-split/test_gt"
    out_path="results/swinir_finetune"
    pred_path=$out_path/restored_faces

    # evaluation
    local args=(
        --in_path data/celeba-raw-noise-4x-iso-1500/test_lq 
        --out_path $out_path
        --ckpt_path logs/swinir_gan_finetune/ckpts/model_latest.pth 
    )
    python run_swinir.py "${args[@]}"


    # reference based metrics arguments
    local args_eval=(
        -restored_folder $pred_path 
        -gt_folder $gt_path 
        --out_path $out_path
    )

    # fid arguments
    local args_eval_fid=(
        -restored_folder $pred_path 
        --out_path $out_path
        --fid_stats weights/metrics/inception_CelebA-Test.pth
    )

    # run evaluations
    python calculate_psnr_ssim.py "${args_eval[@]}"
    python calculate_lpips.py "${args_eval[@]}"
    python calculate_fid.py "${args_eval_fid[@]}"


    # Non-reference based metrics arguments
    local args_eval=(
        --pred_dir $pred_path 
        --out_path $out_path 
    )

    # run evaluations
    python eval_real.py "${args_eval[@]}"



}

run_commands

