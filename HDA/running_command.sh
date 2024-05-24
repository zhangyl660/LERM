CUDA_VISIBLE_DEVICES=2 python main.py --method NO 
CUDA_VISIBLE_DEVICES=2 python main.py --alpha 1 --method LERM_L1
CUDA_VISIBLE_DEVICES=2 python main.py --alpha 0.01 --method BNM
CUDA_VISIBLE_DEVICES=2 python main.py --alpha 0.0005 --method ENT