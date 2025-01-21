# [WACV 2025 (Oral)] Towards Unsupervised Blind Face Restoration using Diffusion Prior

Tianshu Kuai, Sina Honari, Igor Gilitschenski, and Alex Levinshtein


[**Project Page**](https://dt-bfr.github.io/) | [**arXiv**](https://arxiv.org/abs/2410.04618) | [**Dataset**](https://drive.google.com/drive/folders/1aIJnDIIlHsaLWKZvOjSxsg3Q_pA8gMDt?usp=sharing)


![teaser](misc/imgs/teaser.png)


---

## Installation

Create a new conda environment using the provided `environment.yml` file:
```
# install conda environment
conda env create -f environment.yml
conda activate Diff_tuned_BFR

# install pytorch
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install python dependencies for CodeFormer
cd CodeFormer
python basicsr/setup.py develop
cd ..
```

Download the essential pre-trained model weights:
```
bash misc/download_weights.sh
```

## Dataset Preparation

Download the entire [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) for the adversarial loss during fine-tuning. Preprocess the `1024x1024` images to save them into size of `512x512` under the same parent directory:
```
python datapipe/prepare/face/big2small_face.py --face_dir [Face folder(1024x1024)] --pch_size 512
```


### Synthetic Dataset
To reproduce our results on synthetic dataset, download the [CelebA-Test dataset](https://xinntao.github.io/projects/gfpgan) (Both HQ and LQ). We provide script to generate our synthetic dataset described in the paper. Place the high-quality `512x512` face images in a directory `--hq_dir` and run:
```
cd data_prep
python generate_LQ_from_HQ.py --hq_dir ../data/celeba_512_validation --results_dir 4x-downsampling-moderate-noise --iso_min 1500 --iso_max 1500 --scale 4
cd ..
```
The synthesized low-quality images will be saved to `./data_prep/4x-downsampling-moderate-noise/div2k_scale_x4_IMX754_bayer_ISO1500-1500/demosaic_bayer_rendered`.

Then split it into training set (2500 images) and testing set (500 images). Move the first 2500 images to `data/celeba-raw-noise-4x-iso-1500/train_lq` and the rest of the 500 images to `data/celeba-raw-noise-4x-iso-1500/test_lq`. Also create directories that contain the corresponding ground-truth images: `data/train_gt` and `data/test_gt` by splitting the `data/celeba_512_validation`. 

### Real-world Dataset

We use the entire [Wider-Test](https://shangchenzhou.com/projects/CodeFormer/) for fine-tuning the pre-trained model. To evaluate the fine-tuned model, download our [Wider-Test-200](https://drive.google.com/drive/folders/1aIJnDIIlHsaLWKZvOjSxsg3Q_pA8gMDt?usp=sharing). Note that our `Wider-Test-200` does not contain overlapping images in the `Wider-Test` dataset.


## Generating Pseudo Targets

### SwinIR
Generate pseudo targets for pre-trained SwinIR:
```
# define paths
input_dir=data/celeba-raw-noise-4x-iso-1500/train_lq
output_dir=data/celeba-raw-noise-4x-iso-1500/train_targets

# generate pseudo targets for SwinIR
bash scripts/generate_targets_swinir.sh $input_dir $output_dir 16
# argv[1]: low-quality inputs
# args[2]: output directory for pseudo targets
# argv[3]: downsampling factor for low pass filter
```

---

### CodeFormer or any pre-trained restoration model
Generate pseudo targets for any pre-trained restoration model (you can replace the CodeFormer inference with another pre-trained restoration model). Here we use pre-trained CodeFormer as an example:
```
# define paths
input_dir=../data/celeba-raw-noise-4x-iso-3000/train_lq
pretrained_results_dir=../data/celeba-raw-noise-4x-iso-3000/codeformer

# run CodeFormer
cd CodeFormer
python inference_codeformer.py -w 0.5 --has_aligned --input_path $input_dir --output_path $pretrained_results_dir
cd ..

# define paths
pretrained_results_dir=data/celeba-raw-noise-4x-iso-3000/codeformer
output_dir=data/celeba-raw-noise-4x-iso-3000/train_targets_codeformer

# generate pseudo targets
bash scripts/generate_targets_general.sh $pretrained_results_dir $output_dir 8
# argv[1]: pre-trained restoration model outputs
# args[2]: output directory for pseudo targets
# argv[3]: downsampling factor for low pass filter
```

## Fine-tuning using Pseudo Targets

### SwinIR
Prepare a config file for fine-tuning by following the example config [here](./configs/main/swinir_gan.yaml). Specifically, make sure the paths specified under the `data` section are correct. Then run the following commands for fine-tuning:
```
config=configs/main/swinir_gan.yaml
log_dir=logs/swinir_gan_finetune

python train.py --cfg_path $config --save_dir $log_dir
```

The results are saved in the corresponding folder under `logs`.

---

### CodeFormer
Prepare a config file for fine-tuning by following the example config [here](./CodeFormer/options/codeformer_finetune.yml). Specifically, make sure the paths specified under the `datasets` section are correct. Then run the following commands for fine-tuning:
```
cd CodeFormer
python finetune.py -opt options/codeformer_finetune.yml
cd ..
```

The results are saved in the corresponding folder under `CodeFormer/experiments`.

## Evaluation

The results are saved as `.txt` files inside the corresponding `$results_dir`.

### SwinIR

#### Synthetic Datasets
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

#### Real-world Datasets
```
# define paths
test_inputs=data/Wider-Test-200
results_dir=results/swinir_finetune_wider-test-200

# run fine-tuned model on testing set
python run_swinir.py --in_path $test_inputs --out_path $results_dir --ckpt_path $log_dir/ckpts/model_latest.pth

# run evaluation
bash scripts/eval_real.sh $results_dir
# argv[1]: results
```

---

### CodeFormer
#### Synthetic Datasets
```
# define paths
test_inputs=../data/celeba-raw-noise-4x-iso-3000/test_lq
results_dir=../results/codeformer_finetune/restored_faces
results_dir_eval=../results/codeformer_finetune
gt_dir=data/CelebA-Test-split/test_gt

# run fine-tuned model on testing set
cd CodeFormer
python inference_codeformer.py -w 0.5 --has_aligned --input_path $test_inputs --output_path $results_dir --ckpt $Finetuned_CodeFormer_ckpt
cd ..

# run evaluation
bash scripts/eval_synthetic.sh $results_dir_eval $gt_dir
# argv[1]: results
# args[2]: gt images
```

#### Real-world Datasets
```
# define paths
test_inputs=../data/Wider-Test-200
results_dir=../results/codeformer_finetune_wider-test-200/restored_faces
results_dir_eval=../results/codeformer_finetune_wider-test-200

# run fine-tuned model on testing set
cd CodeFormer
python inference_codeformer.py -w 0.5 --has_aligned --input_path $test_inputs --output_path $results_dir --ckpt $Finetuned_CodeFormer_ckpt
cd ..

# run evaluation
bash scripts/eval_real.sh $results_dir_eval
# argv[1]: results
```


## Pre-training

Please refer to our paper for details on the SwinIR pre-training. To pre-train your own SwinIR on synthetic dataset generated by commonly used degradation pipeline in blind face restoration works:
```
# define paths
config=configs/others/swinir_gan_pretrain.yaml
log_dir=logs/swinir_gan_pretrain

python train.py --cfg_path $config --save_dir $log_dir
```

For pre-training the Codeformer, please refer to their official implementation [here](https://github.com/sczhou/CodeFormer). We use the released checkpoint from CodeFormer authors as the pre-trained model.


## Acknowledgement

Our project is mainly based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [DifFace](https://github.com/zsyOAOA/DifFace), and [CodeFormer](https://github.com/sczhou/CodeFormer). We also used the evaluation scripts from [VQFR](https://github.com/TencentARC/VQFR), and the low-pass filter implementation from [Resizer](https://github.com/assafshocher/resizer). A big thanks to their works and efforts in releasing the code.

## Citation
If you find this project useful in your work, please consider citing it:
```
@inproceedings{kuai2025towards,
    author={Kuai, Tianshu and Honari, Sina and Gilitschenski, Igor and Levinshtein, Alex},
    title={Towards Unsupervised Blind Face Restoration using Diffusion Prior},
    booktitle={WACV},
    year={2025},
}
```

## License
Please see the [LICENSE](LICENSE) file. 
