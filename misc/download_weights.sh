# download pre-trained weights

cd weights

# download the pre-trained diffusion model
mkdir diffusion && cd diffusion
wget https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth && cd ..

# download our pre-trained SwinIR model

# mkdir SwinIR && cd SwinIR
# wget $Our_SwinIR_link && cd .. && cd ..

# download pre-trained CodeFormer models

cd CodeFormer/weights/CodeFormer
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/vqgan_discriminator.pth && cd ../../..


# download stats files for computing FID

# mkdir metrics && cd metrics
# wget $Our_synthetic_FID_stats_link && wget $FFHQ_FID_stats_link && cd ..