#!/usr/bin/env bash

cd /opt/styles

STYLES_DIR=/opt/styles/models/21styles
CARTOON_DIR=/opt/styles/models/cartoon
FACEDETECT_DIR=/opt/styles/models/faces
YOUNG_DIR=/opt/styles/models/young
SERVING_MODELS=""
if [ "$DISABLE_STYLES" != "true" ]; then
  SERVING_MODELS="-o style_model_path=$STYLES_DIR/21styles.model $SERVING_MODELS"
fi
if [ "$DISABLE_CARTOONS" != "true" ]; then
  SERVING_MODELS="-o cartoon_model_path=$CARTOON_DIR $SERVING_MODELS"
fi
if [ "$DISABLE_BEAUTY" != "true" ]; then
  SERVING_MODELS="-o beauty_model_path=$YOUNG_DIR $SERVING_MODELS"
fi
echo "Sering opts: $SERVING_MODELS"
kuberlab-serving --driver null \
  --model-path null \
  --hooks all_styles.py \
  --http-enable \
  -o max_size=512 \
  -o styles_samples_path=$STYLES_DIR/21styles \
  -o tf_opencv_model_path=$FACEDETECT_DIR \
  -o face_detection_tensor_rt=False \
  -o face_detection_type=tf-opencv \
  -o color_correction=True \
  -o style_size=512 \
  -o output_view=s \
  -o transfer_mode=direct $SERVING_MODELS \
  --status-name=output \
  --cloud_config="36cbba05-b29c-43c2-96fe-54f804713b9f@https://dev.kibernetika.io/api/v0.2/workspace/demo-set/serving/picture-status-1-0-0/config/live"
