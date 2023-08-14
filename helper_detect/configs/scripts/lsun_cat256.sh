


python gen.py --load_json 256/nor_lsun-cat256.json
python gen.py --load_json 256/adv_ddpm_lsun-cat256.json
python gen.py --load_json 256/adv_ddim_lsun-cat256.json
python gen.py --load_json 256/adv_pndm_lsun-cat256.json


python gen.py --load_json 256/adv_ddpm_ema_lsun-cat256.json
python gen.py --load_json 256/adv_ddim_ema_lsun-cat256.json
python gen.py --load_json 256/adv_pndm_ema_lsun-cat256.json


python extract.py --load_json 256/rn18_lsun-cat256_multiLID.json
python extract.py --load_json 256/rn18_lsun-cat256_multiLID_ddim.json
python extract.py --load_json 256/rn18_lsun-cat256_multiLID_pndm.json


python extract.py --load_json 256/rn18_lsun-cat256_ema_multiLID.json
python extract.py --load_json 256/rn18_lsun-cat256_ema_multiLID_ddim.json
python extract.py --load_json 256/rn18_lsun-cat256_ema_multiLID_pndm.json


python detect.py --load_json 256/rn18_rf_lsun-cat256_multiLID.json
python detect.py --load_json 256/rn18_rf_lsun-cat256_multiLID_ddim.json
python detect.py --load_json 256/rn18_rf_lsun-cat256_multiLID_pndm.json

python detect.py --load_json 256/rn18_lr_lsun-cat256_multiLID.json
python detect.py --load_json 256/rn18_lr_lsun-cat256_multiLID_ddim.json
python detect.py --load_json 256/rn18_lr_lsun-cat256_multiLID_pndm.json