# CDAN
python train_image.py --gpu_id 0 --method NO --num_iterations 8000 --dset office --s_dset_path /data/office31/image_list/amazon.txt --t_dset_path /data/office31/image_list/dslr.txt --test_interval 400 --output_dir CDAN/a2d


# CDAN+BNM
python train_image.py --gpu_id 0 --method BNM --num_iterations 8000  --dset office --s_dset_path /data/office31/image_list/amazon.txt --t_dset_path /data/office31/image_list/dslr.txt --test_interval 400 --output_dir CDAN/BNM/a2d --trade_off 0.1


# CDAN+EntMin
python train_image.py --gpu_id 0 --method ENT --num_iterations 8000  --dset office --s_dset_path /data/office31/image_list/amazon.txt --t_dset_path /data/office31/image_list/dslr.txt --test_interval 400 --output_dir CDAN/ENT/a2d --trade_off 0.1


# CDAN+LERM
python train_image.py --gpu_id 0 --method LERM --num_iterations 8000  --dset office --s_dset_path /data/office31/image_list/amazon.txt --t_dset_path /data/office31/image_list/dslr.txt --test_interval 400 --output_dir CDAN/LERM/a2d --trade_off 0.1