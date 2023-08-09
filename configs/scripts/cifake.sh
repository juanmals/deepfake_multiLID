
python gen.py --load_json cifake/nor_cifake.json
python gen.py --load_json cifake/adv_cifake.json

python extract.py --load_json cifake/rn18_cifake_multiLID.json

python detect.py --load_json cifake/rn18_rf_cifake_multiLID.json
python detect.py --load_json cifake/rn18_lr_cifake_multiLID.json

python extract.py --load_json cifake/perturbation/rn18_cifake_blur1_multiLID.json

