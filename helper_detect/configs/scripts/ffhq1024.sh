
python gen.py --load_json 1024/adv_ncsnpp_ffhq1024.json

python gen.py --load_json 1024/nor_ffhq1024.json


python extract.py --load_json 1024/rn18_ffhq1024_multiLID_ncsnpp.json
python detect.py --load_json 1024/rn18_rf_ffhq1024_multiLID_ncsnpp.json