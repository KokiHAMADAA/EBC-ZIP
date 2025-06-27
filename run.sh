python trainer.py \
    --model_name vgg19_ae --block_size 16 \
    --dataset shb --input_size 448 --num_crops 1 \
    --reg_loss zipnll --aux_loss none --weight_cls 1.0 --weight_reg 1.0 --weight_aux 0.0 \
    --amp --num_workers 8 --total_epochs 1300 --save_best_k 5  --ckpt_dir_name vgg19_ae_32_448x1
