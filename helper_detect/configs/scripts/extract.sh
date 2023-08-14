
# not pretrained random initialized resnet18
# python extract.py --load_json "32/rn18_cifar10.json"
# python extract.py --load_json "32/rn18_butterfly32.json" 

# python extract.py --load_json "32/rn18_cifar10_multiLID.json"
# python extract.py --load_json "32/rn18_butterfly32_multiLID.json"

# python extract.py --load_json "32/rn18_cifar10_bb-multiLID.json"
# python extract.py --load_json "32/rn18_butterfly32_bb-multiLID.json"


# pretrained on cifar10 
python extract.py --load_json "32/pretrcifar10_rn18_cifar10.json"
python extract.py --load_json "32/pretrcifar10_rn18_butterfly32.json"

python extract.py --load_json "32/pretrcifar10_rn18_cifar10_multiLID.json"
python extract.py --load_json "32/pretrcifar10_rn18_butterfly32_multiLID.json"

python extract.py --load_json "32/pretrcifar10_rn18_cifar10_bb-multiLID.json"
python extract.py --load_json "32/pretrcifar10_rn18_butterfly32_bb-multiLID.json"

