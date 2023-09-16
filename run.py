import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont


# GroundingDINO
from DINO.API import DINO_API

# SAM
from Segment_Anything.API import SAM_API

import warnings
warnings.simplefilter("ignore", UserWarning)

class run:
    def __init__(self,dino_ckpt='models/groundingdino_swint_ogc.pth', sam_ckpt='models/sam_vit_h_4b8939.pth',device='cuda') -> None:
        self.dino = DINO_API(dino_ckpt,device=device)
        self.sam = SAM_API(sam_ckpt,device=device)
    
    
    def process_boxes(self, H, W, boxes_filt, scores,pred_phrases, iou_threshold=0.8,use_nms=True):
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W,H,W,H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] /2
            boxes_filt[i][2:] += boxes_filt[i][:2]
            
        boxes_filt = boxes_filt.cpu()
        if use_nms:
            boxes_filt,pred_phrases = self.nms(boxes_filt, scores,pred_phrases, iou_threshold)
        return boxes_filt, pred_phrases
     
    def nms(self, boxes_filt,scores,pred_phrases,iou_threshold):
        # print(f"Before NMS: {boxes_filt.shape[0]} boxes")
        nms_idx = torchvision.ops.nms(
            boxes_filt, scores, iou_threshold).numpy().tolist()
        boxes_filt = boxes_filt[nms_idx]
        pred_phrases = [pred_phrases[idx] for idx in nms_idx]
        # print(f"After NMS: {boxes_filt.shape[0]} boxes")
        return boxes_filt, pred_phrases
        
    def predict(self, im_path, text_prompt, output_dir="./output",name=""):
        
        os.makedirs(output_dir,exist_ok=True)
        os.makedirs(os.path.join(output_dir,"mask"),exist_ok=True)
        os.makedirs(os.path.join(output_dir,'img'),exist_ok=True)
        image_pil = Image.open(im_path).convert('RGB')
        
        boxes_filt, scores, pred_phrases = self.dino.get_grounding_output(
            image_pil.copy(), text_prompt, box_threshold=0.3, text_threshold=0.25
        )
        size = image_pil.size
        
        H,W = size[1],size[0]
        boxes_filt, pred_phrases = self.process_boxes(H, W, boxes_filt,scores,pred_phrases)
        
        # 所有的mask，把mask合成一张图，所有的mask汇成一个([0-1])
        masks, masks_pil, mask = self.sam.predict(image_pil,boxes_filt) 
        
        np_image = np.array(image_pil)
        mask = mask.squeeze().cpu().numpy()[:,:,None]
        mask_mul_image = np.ones_like(np_image)*255
        mask_mul_image = mask_mul_image*(1-mask) + np_image * mask
        mask_mul_image = Image.fromarray(mask_mul_image.astype(np.uint8))
        if name =="":
            name = "mask_mul_image"
        mask_mul_image.save(f"{output_dir}/img/{name}.png")
        masks_pil.save(f"{output_dir}/mask/{name}_mask.png")
        
        
if __name__ == '__main__':
    app = run()
    data_dir = "video_x512/car_w_bg"
    for im in os.listdir(data_dir):
        if im.endswith(".jpg"):
            print(im)
            im_path = os.path.join(data_dir,im)
            im_name = im.split('.')[0]
            app.predict(im_path=im_path,text_prompt='a jeep car',name=im_name)
        
        