cd src
# train
python main.py multi_pose --exp_id test_hg4 --dataset thermal_pose --arch hourglass3 --batch_size 2 --lr 2.5e-4 --input_res 256  --load_model ../models/CornerNet_100.pkl --gpus 0 --num_epochs 2 --val_intervals 1
#python test.py multi_pose --exp_id test_hg4 --dataset thermal_pose --arch hourglass3 --resume --debug 2
cd ..
