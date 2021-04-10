cd src
# train
python main.py multi_pose --exp_id test_hg4 --dataset thermal_pose --arch hourglass4 --batch_size 2 --lr 2.5e-4 --input_res 256 --load_model ../models/poseae_checkpoint.pth --gpus 0 --num_epochs 2 --val_intervals 1
#python test.py multi_pose --exp_id hg4_1x_gray --dataset thermal_pose --arch hourglass4 --input_res 512 --load_model ../models/model_last.pth 
cd ..
