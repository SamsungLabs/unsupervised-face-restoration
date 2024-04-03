input_dir=$1
output_dir=$2
filter_down_factor=$3

python generate_pseudo_targets.py --in_path $input_dir \
                                  --out_path $output_dir \
                                  --down_N $filter_down_factor \
                                  --started_timesteps 150 \
                                  --reg_end_timesteps 90 \
                                  --apply_filter \
                                  --config_path configs/targets/iddpm_ffhq512_swinir_gan.yaml \
                                  --swinir 