# For ERM
CUDA_VISIBLE_DEVICES=0 python main.py --method NO
# For LERM_L1
CUDA_VISIBLE_DEVICES=0 python main.py --alpha 1 --method LERM_L1
# For BNM
CUDA_VISIBLE_DEVICES=0 python main.py --alpha 0.01 --method BNM
# For EET
CUDA_VISIBLE_DEVICES=0 python main.py --alpha 0.0005 --method ENT