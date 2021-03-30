cd src

# test
# python test.py multi_pose --exp_id dla_3x_gray_0frz --keep_res --dataset thermal_pose --load_model ../models/multi_pose_dla_3x_gray_0frz.pth
python test.py multi_pose --exp_id hrnet_best_timetest --arch hrnet32 --input_res 512 --dataset thermal_pose --load_model ../models/multi_pose_hrnet_3x_gray_finetune.pth --save_times_per_people
cd ..
