import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import xywh2xyxy
from utils.datasets import letterbox
import torch.nn as nn
import torchvision

def set_module_inplace(model, inplace=False):
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = inplace

class YOLOV5TorchObjectDetector(nn.Module):
    def __init__(self, model_weight, device, img_size, names=None, confidence=0.45, iou_thresh=0.45, agnostic_nms=False):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.confidence = confidence
        self.iou_thresh = iou_thresh
        self.agnostic = agnostic_nms
        self.model = attempt_load(model_weight, map_location=device)
        set_module_inplace(self.model, inplace=False)
        self.model.requires_grad_(True)
        self.model.to(device)
        if names is None:
            self.names = ['class0', 'class1', 'class2']
        else:
            self.names = [n.strip() for n in names]
        img = torch.zeros((1, 3, *self.img_size), device=device)
        self.model(img, img)  # dummy warmup

    @staticmethod
    def non_max_suppression(prediction, logits, conf_thres=0.3, iou_thres=0.45):
        nc = prediction.shape[2] - 5
        xc = prediction[..., 4] > conf_thres
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        logits_output = [torch.zeros((0, nc), device=logits.device)] * logits.shape[0]
        for xi, (x, log_) in enumerate(zip(prediction, logits)):
            x = x[xc[xi]]
            log_ = log_[xc[xi]]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            box = xywh2xyxy(x[:, :4])
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            log_ = log_[conf.view(-1) > conf_thres]
            n = x.shape[0]
            if not n:
                continue
            boxes, scores = x[:, :4], x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            output[xi] = x[i]
            logits_output[xi] = log_[i]
        return output, logits_output

    @staticmethod
    def yolo_resize(img, new_shape=(640, 640)):
        return letterbox(img, new_shape=new_shape)[0]

    def forward(self, img1, img2):
        prediction, logits, _ = self.model(img1, img2, augment=False)
        prediction, logits = self.non_max_suppression(prediction, logits, self.confidence, self.iou_thresh)
        self.boxes, self.class_names, self.classes, self.confidences = [[[] for _ in range(img1.shape[0])] for _ in range(4)]
        for i, det in enumerate(prediction):
            if len(det):
                for *xyxy, conf, cls in det:
                    bbox = [int(b) for b in xyxy]
                    self.boxes[i].append(bbox)
                    self.confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    self.classes[i].append(cls)
                    if self.names is not None:
                        self.class_names[i].append(self.names[cls])
                    else:
                        self.class_names[i].append(cls)
        return [self.boxes, self.classes, self.class_names, self.confidences], logits

    def preprocessing(self, img, img2=None):
        # 支持单张或成对输入
        def _one(imgx):
            if len(imgx.shape) != 4:
                imgx = np.expand_dims(imgx, axis=0)
            im0 = imgx.astype(np.uint8)
            imgx = np.array([self.yolo_resize(im, new_shape=self.img_size) for im in im0])
            imgx = imgx.transpose((0, 3, 1, 2))
            imgx = np.ascontiguousarray(imgx)
            imgx = torch.from_numpy(imgx).to(self.device)
            imgx = imgx / 255.0
            return imgx
        if img2 is None:
            return _one(img)
        else:
            return _one(img), _one(img2)

