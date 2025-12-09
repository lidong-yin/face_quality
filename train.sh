# 只训练 header, 输入为feats特征
python train.py data.input_type=feat model.model_name=QualityRegressorHeder trainer.batchsize=16384

# 训练backbone + header, 输入为img
python train.py data.input_type=img model.model_name=FaceQualityModel trainer.batchsize=1024
