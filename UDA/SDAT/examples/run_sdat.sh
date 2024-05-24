# Office-31
python cdan_mcc_sdat.py /data/dataset/office31 -d Office31 -s A -t W -a resnet50 --epochs 15 --log logs/sdat_resnet50/Office31/a2w --gpu 0 --rho 0.02 --lr 0.002 --temperature 2.0 --bottleneck-dim 256 --eps 0

# Office-home
python cdan_mcc_sdat.py /data/dataset/office_home -d OfficeHome -s Ar -t Cl -a resnet50 --epochs 30 --log logs/sdat_resnet50/OfficeHome/ArtoCl --gpu 0 --rho 0.02 --lr 0.01 --temperature 2.0 --bottleneck-dim 256 --eps 0
