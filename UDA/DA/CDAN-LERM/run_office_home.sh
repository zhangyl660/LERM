# CDAN
python train_image.py --gpu_id 0 --method NO --num_iterations 8000  --dset office-home --s_dset_path image_list/officehome/Art.txt --t_dset_path image_list/officehome/Clipart.txt --test_interval 400 --output_dir new/CDAN/A2C

# BNM+CDAN
python train_image.py --gpu_id 0 --method BNM --num_iterations 8000  --dset office-home --s_dset_path image_list/officehome/Art.txt --t_dset_path image_list/officehome/Clipart.txt --test_interval 400 --output_dir new/BNM/A2C --trade_off 0.1

# EntMin+CDAN
python train_image.py --gpu_id 0 --method ENT --num_iterations 8000  --dset office-home --s_dset_path image_list/officehome/Art.txt --t_dset_path image_list/officehome/Clipart.txt --test_interval 400 --output_dir new/ENT/A2C --trade_off 0.1

# LERM+CDAN
python train_image.py --gpu_id 0 --method LERM --num_iterations 8000  --dset office-home --s_dset_path image_list/officehome/Art.txt --t_dset_path image_list/officehome/Clipart.txt --test_interval 400 --output_dir new/LERM/A2C --trade_off 10
