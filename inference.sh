# 推理feats
# python inference_feats.py \
#     --model_path weights/fuq_regressor_v1.pth \
#     --feature_file dataset/lyg_v0_feats.npy \
#     --output dataset/lyg_v0_data_q1.npy


# 推理imgs
python inference_imgs.py \
    --model_path weights/model-imgs-mobilenet_v3_small.pth \
    --input_dir benchmark_face_quality.pkl