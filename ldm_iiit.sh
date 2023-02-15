CUDA_VISIBLE_DEVICES=0 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.5 --noise 0.05 --ares --batch_size 3
CUDA_VISIBLE_DEVICES=0 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.5 --noise 0.15 --ares --batch_size 3
CUDA_VISIBLE_DEVICES=0 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.5 --noise 0.25 --ares --batch_size 3
CUDA_VISIBLE_DEVICES=0 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.5 --noise 0.35 --ares --batch_size 3
#CUDA_VISIBLE_DEVICES=1 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.4 --noise 0.05 --ares --batch_size 2
#CUDA_VISIBLE_DEVICES=1 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.4 --noise 0.15 --ares --batch_size 2
#CUDA_VISIBLE_DEVICES=1 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.4 --noise 0.25 --ares --batch_size 2
#CUDA_VISIBLE_DEVICES=1 python test_ocr_new.py --base configs/latent-diffusion/custom_ares.yaml --gpus 0, --dataset IIIT5k --bright 0.4 --noise 0.35 --ares --batch_size 2
