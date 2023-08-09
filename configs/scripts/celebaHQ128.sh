
python gen.py --load_json 128/nor_haircolor_celebaHQ128.json
python gen.py --load_json 128/adv_ddpm_celebaHQ128.json
python gen.py --load_json 128/adv_ldm_celebaHQ128.json
python gen.py --load_json 128/adv_ncsnpp_celebaHQ128.json


python extract.py --load_json 128/rn18_celebaHQ128_multiLID.json


python detect.py --load_json 128/rn18_rf_celebaHQ128_multiLID.json
python detect.py --load_json 128/rn18_lr_celebaHQ128_multiLID.json