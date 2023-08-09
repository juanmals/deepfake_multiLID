
python gen.py --load_json imagenet/nor_imagenet.json


python gen.py --load_json imagenet/adv_imagenet_dit.json
python gen.py --load_json imagenet/adv_imagenet_mdt.json

python gen.py --load_json imagenet/adv_imagenet_mdt.json


python extract.py --load_json imagenet/rn18_imagenet_dit_multiLID.json
python extract.py --load_json imagenet/pretr_rn18_imagenet_dit_multiLID.json



python detect.py --load_json imagenet/pretrimagenet_rn18_lr_imagenet_multiLID.json
python detect.py --load_json imagenet/pretrimagenet_rn18_rf_imagenet_multiLID.json

python detect.py --load_json imagenet/rn18_lr_imagenet_multiLID.json
python detect.py --load_json imagenet/rn18_rf_imagenet_multiLID.json