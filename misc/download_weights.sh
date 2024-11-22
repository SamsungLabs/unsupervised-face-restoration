# download pre-trained weights

# download the pre-trained diffusion model
cd weights && mkdir diffusion && cd diffusion
wget https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth && cd ..

# download our pre-trained SwinIR model

mkdir SwinIR && cd SwinIR
wget https://github.com/SamsungLabs/DT-BFR/releases/download/v1.0.0/swinir_gan_v1.pth && cd .. && cd ..

# download pre-trained CodeFormer models

cd CodeFormer/weights/CodeFormer
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/vqgan_discriminator.pth && cd ../../..


# download stats files for computing FID

cd weights && mkdir metrics && cd metrics
wget https://github.com/SamsungLabs/DT-BFR/releases/download/v1.0.0/inception_CelebA-Test-500.pth
wget https://github.com/SamsungLabs/DT-BFR/releases/download/v1.0.0/inception_FFHQ_512.pth && cd ../..