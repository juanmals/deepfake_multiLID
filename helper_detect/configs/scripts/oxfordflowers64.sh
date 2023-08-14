
python gen.py     --load_json 64/nor_oxfordflowers64.json
python gen.py     --load_json 64/adv_ddpm_oxfordflowers64.json

python extract.py --load_json 64/rn18_oxfordflowers64_ema_multiLID.json


python detect.py  --load_json 64/rn18_rf_oxfordflowers64_ema_multiLID.json
python detect.py  --load_json 64/rn18_lr_oxfordflowers64_ema_multiLID.json