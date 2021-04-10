cd src 
python test.py multi_pose --dataset thermal_pose  --arch hrnet32 --test --keep_res --load_model ../models/multi_pose_hrnet_3x_gray_finetune.pth 
cd ..
