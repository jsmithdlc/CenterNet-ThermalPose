cd src
# train
#python main.py multi_pose --exp_id hrnet_3x_gray_finetune --dataset thermal_pose --arch hrnet32 --batch_size 4 --lr 1e-3 --load_model ../models/multi_pose_hrnet_3x_gray.pth --gpus 0 --num_epochs 50 
# test
python test.py multi_pose --exp_id hrnet_3x_gray --dataset thermal_pose --arch hrnet32 --keep_res --load_model ../exp/multi_pose/hrnet_32_color_step40-48/model_50.pth
# flip test
#python test.py multi_pose --exp_id hg_3x --dataset coco_hp --arch hourglass --keep_res --resume --flip_test
cd ..
