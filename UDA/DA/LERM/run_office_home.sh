# ENT
python BNM/train_image.py --gpu_id 0 --method ENT --num_iterations 8000  --dset office-home --s_dset_path data/office-home/Art.txt --t_dset_path data/office-home/Clipart.txt --test_interval 400 --output_dir ENT/A2C --trade_off 0.1

# BNM
python BNM/train_image.py --gpu_id 0 --method BNM --num_iterations 8000  --dset office-home --s_dset_path data/office-home/Art.txt --t_dset_path data/office-home/Clipart.txt --test_interval 400 --output_dir BNM/A2C --trade_off 0.1

# LERM
python BNM/train_image.py --gpu_id 0 --method LERM --num_iterations 8000  --dset office-home --s_dset_path data/office-home/Art.txt --t_dset_path data/office-home/Clipart.txt --test_interval 400 --output_dir LERM/A2C --trade_off 10