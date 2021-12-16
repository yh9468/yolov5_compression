python /home/ubuntu/compress_detection/yolov5/train.py --img 640 --batch 64 --epochs 50 --workers 8 \
--data /home/ubuntu/compress_detection/yolov5/data/SKU-110K.yaml --cfg /home/ubuntu/compress_detection/yolov5/models/maskyolov5s.yaml \
--name exp --weights /home/ubuntu/compress_detection/yolov5/yolov5s.pt --prune --prune-rate 0.7 --prune-freq 16