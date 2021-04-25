cd src
# train
#python main.py multi_pose --exp_id hg4_1x_gray --dataset thermal_pose --arch hourglass4 --batch_size 1 --lr 2e-4 --input_res 256  --resume \
#--gpus 0 --num_epochs 2 --val_intervals 1
python test.py multi_pose --exp_id hg4_1x_gray --dataset thermal_pose --arch hourglass4 --resume
