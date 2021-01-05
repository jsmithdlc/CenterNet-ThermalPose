cd src

# test
# python test.py multi_pose --exp_id dla_3x_gray_0frz --keep_res --dataset thermal_pose --load_model ../models/multi_pose_dla_3x_gray_0frz.pth
python test.py multi_pose --arch hourglass --exp_id hg_3x_gray_0frz --input_res 512 --dataset thermal_pose --load_model ../models/multi_pose_hg_3x_gray_0frz.pth
cd ..
