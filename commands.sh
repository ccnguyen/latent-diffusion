CUDA_VISIBLE_DEVICES=0 python main.py --base configs/autoencoder/autoencoder_kl_8x8x64.yaml -t --gpus 0,
CUDA_VISIBLE_DEVICES=7 python main.py --base configs/autoencoder/ae_kl_64x64x3_zeus.yaml -t --gpus 0,
CUDA_VISIBLE_DEVICES=4 python main.py --base configs/autoencoder/ae_kl_64x64x3_zeus.yaml -t --gpus 0,

CUDA_VISIBLE_DEVICES=6 python main.py --base configs/latent-diffusion/lol-ldm-kl-4-zeus.yaml -t --gpus 0,


CUDA_VISIBLE_DEVICES=0 python main.py --base configs/autoencoder/ae_kl_64x64x3_local.yaml -t --gpus 0,
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/lol-ldm-kl-4-local.yaml -t --gpus 0,
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/lol-ldm-kl-4-local-attn.yaml -t --gpus 0,


########## test
CUDA_VISIBLE_DEVICES=0 python test.py --base configs/latent-diffusion/lol-ldm-kl-4-local.yaml --gpus 0,
CUDA_VISIBLE_DEVICES=0 python test.py --base configs/autoencoder/ae_kl_64x64x3_local.yaml --gpus 0,

CUDA_VISIBLE_DEVICES=0 python test_ocr.py --base configs/latent-diffusion/lol-ldm-kl-4-local.yaml --gpus 0, --dataset SVTP
