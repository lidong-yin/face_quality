export OMP_NUM_THREADS=4
torchrun --nproc_per_node=4 train_arcface_header.py \
     --train_features_path dataset/xxx_blur_sampler_10_feats.npy \
     --train_labels_path dataset/xxx_blur_sampler_10_gt.npy \
     --output_dir output/arcface_xxx \
     --num_classes 202936 \
     --num_epoch 30 \
     --batch_size 1024