pip install tensorboard
pip install -r requirements.txt
apt update
apt install -y libgl1-mesa-glx
apt install -y libglib2.0-0
python -m torch.distributed.launch --nproc_per_node 4 train.py --batch-size 48 --data data/train_test.yaml \
        --device 0,1,2,3 --project runs/s_baseline