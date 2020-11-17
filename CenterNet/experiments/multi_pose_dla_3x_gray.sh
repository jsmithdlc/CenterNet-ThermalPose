cd src
# train
#python main.py multi_pose --exp_id train_dla_chenwang --dataset chen_wang --batch_size 2 --lr 5e-4 --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0 --num_workers 16
# test
python test.py multi_pose --exp_id test_dla_chenwang --dataset chen_wang --keep_res --load_model ../models/multi_pose_dla_3x_gray_384_0frz.pth
# flip test
#python test.py multi_pose --exp_id dla_1x --dataset coco_hp --keep_res --resume --flip_test
cd ..
