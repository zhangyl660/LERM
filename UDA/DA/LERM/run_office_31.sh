# ENT 
python BNM/train_image.py --gpu_id 0 --method ENT --num_iterations 8000  --dset office --s_dset_path /data/dataset/image_list/amazon.txt --t_dset_path /data/dataset/image_list/dslr.txt --test_interval 400 --output_dir ENT/a2d --trade_off 0.1

# BNM
python BNM/train_image.py --gpu_id 0 --method BNM --num_iterations 8000  --dset office --s_dset_path /data/dataset/image_list/amazon.txt --t_dset_path /data/dataset/image_list/dslr.txt --test_interval 400 --output_dir BNM/a2d --trade_off 0.1

# LERM
python BNM/train_image.py --gpu_id 0 --method LERM --num_iterations 8000  --dset office --s_dset_path /data/dataset/image_list/amazon.txt --t_dset_path /data/dataset/image_list/dslr.txt --test_interval 400 --output_dir LERM/a2d  --trade_off 10