cd src
# train
python main.py multi_pose --exp_id hg_1x --dataset thermal_pose --arch hourglass --batch_size 1 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpus 0 --num_epochs 50 --lr_step 40 --freeze_backbone
# test
#python test.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --keep_res --resume
# flip test
#python test.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --keep_res --resume --flip_test
cd ..
