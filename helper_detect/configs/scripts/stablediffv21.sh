

python gen.py --load_json stablediffv2/nor_stablediffv21_laion768.json
python gen.py --load_json stablediffv2/adv_stablediffv21_laion768.json


python extract.py --load_json   stablediffv2/rn18_stablediffv21_laion768_1000_multiLID.json
python extract.py --load_json   stablediffv2/rn18_stablediffv21_laion768_2000_multiLID.json
python extract.py --load_json   stablediffv2/rn18_stablediffv21_laion768_5000_multiLID.json
python extract.py --load_json   stablediffv2/rn18_stablediffv21_laion768_8000_multiLID.json

python extract.py --load_json   stablediffv2/rn18_stablediffv21_laion768_border_2000_multiLID.json




python detect.py --load_json stablediffv2/rn18_lr_stablediffv2_multiLID.json
python detect.py --load_json stablediffv2/rn18_rf_stablediffv21_laion768_1000_multiLID.json


python detect.py --load_json stablediffv2/rn18_rf_stablediffv21_laion768_2000_multiLID.json



