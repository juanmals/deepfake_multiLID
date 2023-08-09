
python gen.py --load_json artifact/nor_af-church.json

python gen.py --load_json artifact/adv_ddpm_stablediffv2.json


python gen.py --load_json artifact_csv/nor_af-all_balanced_10000.json
python gen.py --load_json artifact_csv/adv_af-all_gan_balanced_10000.json
python gen.py --load_json artifact_csv/adv_af-all_diff_balanced_10000.json

python gen.py --load_json artifact_csv/nor_af-all_balanced_10500.json
python gen.py --load_json artifact_csv/adv_af-all_fake_balanced_10500.json

python extract.py --load_json artifact_csv/perturbation/rn18_af-all_blur0.5_2000_multiLID.json


python extract.py --load_json artifact_csv/rn18_af-all_10000_gan_balanced_multiLID.json
python extract.py --load_json artifact_csv/rn18_af-all_10000_diff_balanced_multiLID.json
python extract.py --load_json artifact_csv/rn18_af-all_10500_balanced_multiLID.json