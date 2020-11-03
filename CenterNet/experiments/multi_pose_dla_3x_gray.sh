cd src
# train
# python main.py multi_pose --exp_id dla_1x --dataset thermal_pose --batch_size 3 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0 --num_workers 16
# test
python test.py multi_pose --exp_id dla_3x_gray --dataset thermal_pose --keep_res --load_model ../exp/multi_pose/dla_3x_gray/model_best.pth
# flip test
#python test.py multi_pose --exp_id dla_1x --dataset coco_hp --keep_res --resume --flip_test
cd ..
