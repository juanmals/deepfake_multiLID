python gen.py --load_json 256/nor_lsun-church256.json
python gen.py --load_json 256/adv_ddpm_lsun-church256.json
python gen.py --load_json 256/adv_ddim_lsun-church256.json
python gen.py --load_json 256/adv_pndm_lsun-church256.json

python gen.py --load_json 256/adv_ddpm_ema_lsun-church256.json
python gen.py --load_json 256/adv_ddim_ema_lsun-church256.json
python gen.py --load_json 256/adv_pndm_ema_lsun-church256.json


python extract.py --load_json 256/rn18_lsun-church256_multiLID.json
python extract.py --load_json 256/rn18_lsun-church256_multiLID_ddim.json
python extract.py --load_json 256/rn18_lsun-church256_multiLID_pndm.json


python extract.py --load_json 256/rn18_lsun-church256_multiLID_ddpm_ema.json
python extract.py --load_json 256/rn18_lsun-church256_multiLID_ddim_ema.json
python extract.py --load_json 256/rn18_lsun-church256_multiLID_pndm_ema.json


python detect.py --load_json 256/rn18_rf_lsun-church256_multiLID.json
python detect.py --load_json 256/rn18_rf_lsun-church256_multiLID_ddim.json
python detect.py --load_json 256/rn18_rf_lsun-church256_multiLID_pndm.json

python detect.py --load_json 256/rn18_lr_lsun-church256_multiLID.json
python detect.py --load_json 256/rn18_lr_lsun-church256_multiLID_ddim.json
python detect.py --load_json 256/rn18_lr_lsun-church256_multiLID_pndm.json