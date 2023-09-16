import torch
import numpy as np
from PIL import Image
from .segment_anything import build_sam, SamPredictor

class SAM_API:
    def __init__(self, sam_checkpoint:str, device) -> None:
        self.sam = build_sam(checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)
        self.device = device
               
    def predict(self,image,boxes_filt):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        self.sam_predictor.set_image(image)
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image.shape[:2]
        ).to(self.device)
    
        masks,_,_ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False
        )
        
        masks_pil = torch.sum(masks,dim=0).unsqueeze(0)
        mask = torch.where(masks_pil>0, 1,0)
        masks_pil = torch.where(masks_pil>0, True, False)
        masks_pil = masks_pil[0][0].cpu().numpy()
        
        masks_pil = Image.fromarray(masks_pil)
        return masks, masks_pil, mask
        