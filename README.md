# Towards Unsupervised Blind Face Restoration using Diffusion Prior

Tianshu Kuai, Sina Honari, Igor Gilitschenski, and Alex Levinshtein


[**Project**](https://tianshukuai.github.io/) | [**Paper**](https://tianshukuai.github.io/)


![teaser](misc/imgs/teaser.png)


---

## Getting Started

Create a new conda environment using the provided `environment.yml` file:
```
# Install conda environment
conda env create -f environment.yml
conda activate Diff_tuned_BFR
```

Download pre-trained models:
```
bash misc/download_weights.sh
```

## Dataset Preparation

## Generating Pseudo Targets

#### SwinIR
Generate pseudo targets for pre-trained SwinIR:
```
# define paths
input_dir=data/celeba-raw-noise-4x-iso-1500/train_lq
output_dir=data/celeba-raw-noise-4x-iso-1500/train_targets

# generate pseudo targets for SwinIR
bash scripts/generate_targets_general.sh input_dir output_dir 16
# argv[1]: low-quality inputs
# args[2]: output directory for pseudo targets
# argv[3]: downsampling factor for low pass filter
```

---

#### CodeFormer or any pre-trained restoration model
Generate pseudo targets for any pre-trained restoration model:
```
# run pre-trained CodeFormer on inputs
input_dir=data/celeba-raw-noise-4x-iso-1500/train_lq

# run CodeFormer (to be added)

# define paths
pretrained_results_dir=data/celeba-raw-noise-4x-iso-1500/codeformer
output_dir=data/celeba-raw-noise-4x-iso-1500/train_targets

# generate pseudo targets
bash scripts/generate_targets_general.sh pretrained_results_dir output_dir 16
# argv[1]: pre-trained restoration model outputs
# args[2]: output directory for pseudo targets
# argv[3]: downsampling factor for low pass filter
```

## Fine-tuning using Pseudo Targets

### SwinIR
```
config=configs/main/swinir_gan.yaml
log_dir=logs/swinir_gan_finetune

python train.py --cfg_path $config --save_dir $log_dir
```

---

### CodeFormer
```

# to be added

```

## Evaluation

### SwinIR
```
# define paths
test_inputs=data/celeba-raw-noise-4x-iso-1500/test_lq
results_dir=results/swinir_finetune
gt_dir=data/CelebA-Test-split/test_gt

# run fine-tuned model on testing set
python run_swinir.py --in_path $test_inputs --out_path $results_dir --ckpt_path $log_dir/ckpts/model_latest.pth

# run evaluation
bash scripts/eval_synthetic.sh $results_dir $gt_dir
# argv[1]: results
# args[2]: gt images
```

---

### CodeFormer
```

# to be added

```


## Pre-training

To pre-train your own SwinIR on synthetic dataset:
```
to be added
```

For prefer to pre-training the Codeformer, please refer to their official implementation [here](https://github.com/sczhou/CodeFormer). We use the released checkpoint from CodeFormer authors as pre-trained model.


## Acknowledgement

## Citation
If you find this project useful in your work, please consider citing it:
```
@inproceedings{kuai2024towards,
    author={Kuai, Tianshu and Honari, Sina and Gilitschenski, Igor and Levinshtein, Alex},
    title={Towards Unsupervised Blind Face Restoration using Diffusion Prior},
    booktitle={arXiv},
    year={2024},
}
```