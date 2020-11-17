cd src
# train
#python main.py multi_pose --exp_id hg_3x --dataset coco_hp --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 -load_model ../models/ctdet_coco_hg.pth --gpus 0,1,2,3,4 --num_epochs 150 --lr_step 130
# or use the following command if your have dla_1x trained
# python main.py multi_pose --exp_id hg_3x --dataset coco_hp  --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 --gpus 0,1,2,3,4 --num_epochs 150 --lr_step 130 --load_model ../exp/multi_pose/hg_1x/model_40.pth --resume
# test
python test.py multi_pose --exp_id test_hg_3x --dataset chen_wang --arch hourglass --keep_res --load_model ../models/multi_pose_hg_3x_gray_0frz.pth
# flip test
#python test.py multi_pose --exp_id hg_3x --dataset coco_hp --arch hourglass --keep_res --resume --flip_test
cd ..
