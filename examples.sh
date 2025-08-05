# FLIR
python train.py --cfg ./models/crossMamba/yolov5l_crossMamba_FLIR.yaml --data ./data/multispectral/FLIR-align-3class.yaml --hyp ./data/hyp.scratch_FLIR.yaml --project saves/FLIR_CrossMamba --name CrossMamba --epochs 15

# CVC-14
python train.py --cfg ./models/crossMamba/yolov5l_crossMamba_CVC.yaml --data ./data/multispectral/CVC-14.yaml --hyp ./data/hyp.scratch_CVC.yaml --project saves/CVC_CrossMamba --name CrossMamba --epochs 30

# KAIST
python train.py --cfg ./models/crossMamba/yolov5l_crossMamba_KAIST.yaml --data ./data/multispectral/KAIST.yaml --hyp ./data/hyp.scratch_KAIST.yaml --project saves/KAIST_CrossMamba --name CrossMamba --epochs 60

# LLVIP
python train.py --cfg ./models/crossMamba/yolov5l_crossMamba_LLVIP.yaml --data ./data/multispectral/LLVIP.yaml --hyp ./data/hyp.scratch_LLVIP.yaml --project saves/LLVIP_CrossMamba --name CrossMamba --epochs 30



# test
python test.py --weights ./saves/xxxxx/weights/best.pt --data ./data/multispectral/FLIR-align-3class.yaml

python test.py --weights ./saves/xxxxx/weights/best.pt --data ./data/multispectral/CVC-14.yaml

python test.py --weights ./saves/xxxxx/weights/best.pt --data ./data/multispectral/KAIST.yaml

python test.py --weights ./saves/xxxxx/weights/best.pt --data ./data/multispectral/LLVIP.yaml