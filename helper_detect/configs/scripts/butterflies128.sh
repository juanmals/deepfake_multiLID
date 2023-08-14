
python gen.py --load_json 128/nor_butterflies128.json 
python gen.py --load_json 128/adv_ddpm_butterflies128.json 

python extract.py --load_json 128/rn18_butterflies128_multiLID.json
python extract.py --load_json 128/rn18_butterflies128_ema_multiLID.json

python detect.py --load_json 128/rn18_rf_butterflies128_multiLID.json
python detect.py --load_json 128/rn18_lr_butterflies128_multiLID.json

python detect.py --load_json 128/rn18_rf_butterflies128_ema_multiLID.json
python detect.py --load_json 128/rn18_lr_butterflies128_ema_multiLID.json

