
python gen.py --load_json 32/nor_cifar10.json

python gen.py --load_json 32/adv_ddpm_cifar10.json
python gen.py --load_json 32/adv_ddim_cifar10.json
python gen.py --load_json 32/adv_pndm_cifar10.json

python gen.py --load_json 32/adv_ddpm_cifar10_ema.json

python extract.py --load_json 32/rn18_cifar10_ema_multiLID.json
python extract.py --load_json 32/rn18_cifar10_multiLID.json
python extract.py --load_json 32/rn18_cifar10_multiLID_ddim.json
python extract.py --load_json 32/rn18_cifar10_multiLID_pndm.json


python detect.py --load_json 32/rn18_rf_cifar10_multiLID.json
python detect.py --load_json 32/rn18_rf_cifar10_multiLID_ddim.json
python detect.py --load_json 32/rn18_rf_cifar10_multiLID_pndm.json

python detect.py --load_json 32/rn18_lr_cifar10_multiLID.json
python detect.py --load_json 32/rn18_lr_cifar10_multiLID_ddim.json
python detect.py --load_json 32/rn18_lr_cifar10_multiLID_pndm.json