
python gen.py --load_json 28/nor_mnist.json
python gen.py --load_json 28/adv_ddpm_mnist.json

python extract.py --load_json 28/rn18_mnist_multiLID.json


python detect.py --load_json 28/rn18_rf_mnist_multiLID.json

python detect.py --load_json 28/rn18_lr_mnist_multiLID.json
