
python gen.py --load_json 256/nor_lsun-bedroom256.json
python gen.py --load_json 256/adv_ddpm_lsun-bedroom256.json

python gen.py --load_json 256/adv_ddim_ema_lsun-bedroom256.json
python gen.py --load_json 256/adv_ddim_ema_lsun-bedroom256_3.json
python gen.py --load_json 256/adv_pndm_lsun-bedroom256.json
python gen.py --load_json 256/adv_ncsnpp_lsun-bedroom256.json


python gen.py --load_json 256/adv_ddpm_ema_lsun-bedroom256.json
python gen.py --load_json 256/adv_ddim_ema_lsun-bedroom256.json
python gen.py --load_json 256/adv_pndm_ema_lsun-bedroom256.json

python gen.py --load_json 256/adv_sdv21_lsun-bedroom256.json
python gen.py --load_json 256/adv_vqd_lsun-bedroom256.json
python gen.py --load_json 256/adv_ldm_lsun-bedroom256.json


python extract.py --load_json 256/rn18_lsun-bedroom256_multiLID_sdv21.json
python extract.py --load_json 256/rn18_lsun-bedroom256_multiLID_vqd.json
python extract.py --load_json 256/rn18_lsun-bedroom256_multiLID_ldm.json
python extract.py --load_json 256/rn18_lsun-bedroom256_multiLID_adm.json


python extract.py --load_json 256/rn18_lsun-bedroom256_multiLID_ddpm.json
python extract.py --load_json 256/rn18_lsun-bedroom256_multiLID_ddim.json
python extract.py --load_json 256/rn18_lsun-bedroom256_multiLID_pndm.json


python extract.py --load_json 256/rn18_lsun-bedroom256_ema_multiLID.json
python extract.py --load_json 256/rn18_lsun-bedroom256_ema_multiLID_ddim.json
python extract.py --load_json 256/rn18_lsun-bedroom256_ema_multiLID_pndm.json


python detect.py --load_json 256/rn18_rf_lsun-bedroom256_multiLID.json
python detect.py --load_json 256/rn18_rf_lsun-bedroom256_multiLID_ddim.json
python detect.py --load_json 256/rn18_rf_lsun-bedroom256_multiLID_pndm.json


python detect.py --load_json 256/rn18_lr_lsun-bedroom256_multiLID.json
python detect.py --load_json 256/rn18_lr_lsun-bedroom256_multiLID_ddim.json
python detect.py --load_json 256/rn18_lr_lsun-bedroom256_multiLID_pndm.json


python detect.py --load_json 256/multilabelclassifier/rn18_rf_lsun-bedroom256_multiLID.json