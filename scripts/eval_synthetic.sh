results_path=$1
gt_path=$2

# $pred_path is the folder that contains results
pred_path=$results_path/restored_faces

evaluation() {
    # reference based metrics arguments
    local args_eval=(
        -restored_folder $pred_path 
        -gt_folder $gt_path 
        --out_path $results_path
    )

    # fid arguments
    local args_eval_fid=(
        -restored_folder $pred_path 
        --out_path $results_path
        --fid_stats weights/metrics/inception_CelebA-Test.pth
    )

    # Non-reference based metrics arguments
    local args_eval_nr=(
        --pred_dir $pred_path 
        --out_path $results_path 
    )

    # run evaluations
    python calculate_psnr_ssim.py "${args_eval[@]}"
    python calculate_lpips.py "${args_eval[@]}"
    python calculate_fid.py "${args_eval_fid[@]}"
    python eval_real.py "${args_eval_nr[@]}"
}

evaluation