# Office-31
python cdan_mcc_sdat_entropyMin.py /data/dataset/office31 -d Office31 -s A -t W -a resnet50 --epochs 15 -b 32 --log logs/office31/entropy/a2w --gpu 0 --rho 0.02 --lr 0.002 --temperature 2.0 --bottleneck-dim 256 --lambda_method 0.1 --eps=0 

# Office-home
python cdan_mcc_sdat_entropyMin.py /data/dataset/office_home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 -b 32 --log logs/office_home/entropy/ArtoCl --gpu 0 --rho 0.02 --lr 0.01 --temperature 2.0 --bottleneck-dim 256 --lambda_method 0.1 --eps=0
