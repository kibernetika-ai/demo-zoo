#!/usr/bin/env bash

cd /opt/styles

STYLES_DIR=/opt/styles/models/21styles
CARTOON_DIR=/opt/styles/models/cartoon
FACEDETECT_DIR=/opt/styles/models/faces
YOUNG_DIR=/opt/styles/models/young

kuberlab-serving --driver null \
  --model-path null \
  --hooks all_styles.py \
  --http_enable \
  -o max_size=512 \
  -o style_model_path=$STYLES_DIR/21styles.model \
  -o styles_samples_path=$STYLES_DIR/21styles \
  -o cartoon_model_path=$CARTOON_DIR \
  -o  tf_opencv_model_path=$FACEDETECT_DIR \
  -o beauty_model_path=$YOUNG_DIR \
  -o face_detection_tensor_rt=False \
  -o face_detection_type=tf-opencv \
  -o color_correction=True \
  -o style_size=512 \
  -o output_view=s \
  -o transfer_mode=direct