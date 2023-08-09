
python gen.py --load_json 256/nor_haircolor_celebaHQ256.json
python gen.py --load_json 256/adv_ddpm_celebaHQ256.json
python gen.py --load_json 256/adv_ldm_celebaHQ256.json
python gen.py --load_json 256/adv_ncsnpp_celebaHQ256.json

python gen.py --load_json 256/adv_ddpm_ema_celebaHQ256.json
python gen.py --load_json 256/adv_ddim_ema_celebaHQ256.json
python gen.py --load_json 256/adv_pndm_ema_celebaHQ256.json

python extract.py --load_json 256/rn18_celebaHQ256_multiLID.json
python extract.py --load_json 256/rn18_celebaHQ256_multiLID_ldm.json
python extract.py --load_json 256/rn18_celebaHQ256_multiLID_ncsnpp.json

python extract.py --load_json 256/rn18_celebaHQ256_ema_multiLID.json
python extract.py --load_json 256/rn18_celebaHQ256_ema_multiLID_ddim.json
python extract.py --load_json 256/rn18_celebaHQ256_ema_multiLID_pndm.json


python detect.py --load_json 256/rn18_rf_celebaHQ256_multiLID.json
python detect.py --load_json 256/rn18_lr_celebaHQ256_multiLID.json

python detect.py --load_json 256/rn18_rf_celebaHQ256_multiLID_ldm.json
python detect.py --load_json 256/rn18_lr_celebaHQ256_multiLID_ldm.json


python extract.py --load_json 256/perturbation/gaussianblur0.0/lsun-cat256/rn18_lsun-cat256_gaussianblur0.0_0.15_multiLID_ddpm_ema.json
python extract.py --load_json 256/perturbation/gaussianblur0.0/lsun-cat256/rn18_lsun-cat256_gaussianblur0.0_0.5_multiLID_ddpm_ema.json
python extract.py --load_json 256/perturbation/gaussianblur0.0/lsun-cat256/rn18_lsun-cat256_gaussianblur0.0_1_multiLID_ddpm_ema.json
python extract.py --load_json 256/perturbation/gaussianblur0.0/lsun-cat256/rn18_lsun-cat256_gaussianblur0.0_3_multiLID_ddpm_ema.json
