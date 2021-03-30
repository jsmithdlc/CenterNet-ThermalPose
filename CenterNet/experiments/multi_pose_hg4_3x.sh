cd src
# train
# python main.py multi_pose --exp_id test_hg4 --dataset thermal_pose --arch hourglass4 --batch_size 4 --lr 2.5e-4 --input_res 256 --load_model ../models/poseae/checkpoint.pth --gpus 0 --num_epochs 2 --val_intervals 1
python test.py multi_pose --exp_id hg4_1x_gray --dataset thermal_pose --arch hourglass4 --input_res 256 --load_model ../exp/multi_pose/hg4_1x_gray/model_best.pth --debug 2
cd ..
