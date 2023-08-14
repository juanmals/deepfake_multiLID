###### detect ######

# not random initialized resnet18
## Logistic Regression
python detect.py --load_json "rn18_lr_cifar10-butterfly32.json"
python detect.py --load_json "rn18_lr_cifar10-butterfly32_multiLID.json"
python detect.py --load_json "rn18_lr_cifar10-butterfly32_bb-multiLID.json"

## Random Forest
python detect.py --load_json "rn18_rf_cifar10-butterfly32.json"
python detect.py --load_json "rn18_rf_cifar10-butterfly32_multiLID.json"
python detect.py --load_json "rn18_rf_cifar10-butterfly32_bb-multiLID.json"

# pretrained on cifar10 
## Logistic Regression
python detect.py --load_json "pretrcifar10_rn18_lr_cifar10-butterfly32.json"
python detect.py --load_json "pretrcifar10_rn18_lr_cifar10-butterfly32_multiLID.json"
python detect.py --load_json "pretrcifar10_rn18_lr_cifar10-butterfly32_bb-multiLID.json"

## Random Forest
python detect.py --load_json "pretrcifar10_rn18_rf_cifar10-butterfly32.json"
python detect.py --load_json "pretrcifar10_rn18_rf_cifar10-butterfly32_multiLID.json"
python detect.py --load_json "pretrcifar10_rn18_rf_cifar10-butterfly32_bb-multiLID.json"


###### detect - distinguish different ######
python detect.py --load_json "pretrcifar10_rn18_rf_cifar10_multiLID_distinguish.json"