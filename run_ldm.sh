CUDA_VISIBLE_DEVICES=1 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset SVT --bright 0.4 --noise 0.25 --ares --batch_size 3
CUDA_VISIBLE_DEVICES=1 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset SVT --bright 0.4 --noise 0.3 --ares --batch_size 3
CUDA_VISIBLE_DEVICES=5 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.5 --noise 0.4 --ares --batch_size 3
CUDA_VISIBLE_DEVICES=1 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.4 --noise 0.4 --ares --batch_size 3
