# Office-31
python cdan_mcc_sdat_lerm.py /data/dataset/office31 -d Office31 -s A -t W -a resnet50 --epochs 15 --seed 0 -b 32 --log logs/office31/LERM/a2w --gpu 0 --rho 0.02 --lr 0.002 --temperature 2.0 --bottleneck-dim 256 --lambda_method 1 --eps=0

# Office-home
python cdan_mcc_sdat_lerm.py /data/dataset/office_home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --seed 0 -b 32 --log logs/office_home/LERM/ArtoCl --gpu 0 --rho 0.02 --lr 0.01 --temperature 2.0 --bottleneck-dim 256 --lambda_method 10 --eps=0

# VisDA
python cdan_mcc_sdat_lerm.py /data/dataset/visda -d VisDA2017 -s Synthetic -t Real -a resnet101 --epochs 40 --seed 1024 --lr 0.002 --per-class-eval --train-resizing cen.crop --log logs/VisDA2017/LERM --gpu 0 --rho 0.02 --eps=0 --lambda_method 0.01