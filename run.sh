!/bin/bash


echo "正类: 0"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 0 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_0/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 1"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 1 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_1/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 2"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 2 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_2/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 3"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 3 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_3/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 4"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 4 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_4/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 5"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 5 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_5/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 6"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 6 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_6/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 7"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 7 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_7/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6&&
echo "正类: 8"
python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 8 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_8/last.model --svdd_epochs 500 --batch_size 128 --svdd_lr 0.01 --dweight_decay 1e-6
#echo "正类: 9"
#python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 9 --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_9/last.model --svdd_epochs 500 --batch_size 128



#for i in $(seq 3 9)
#do
#  echo "正类: ${i}"
#  python eval.py --mode ood_pre --dataset cifar10 --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx ${i} --load_path /home/xinghongjie/ping/CSI2/logs/cifar10_resnet18_unsup_simclr_CSI_shift_rotation_one_class_${i}/last.model --svdd_epochs 500 --batch_size 128
#  done
#exit 0


