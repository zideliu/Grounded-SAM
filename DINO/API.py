import torch
import random
import numpy as np

from PIL import ImageFont
import DINO.groundingdino.datasets.transforms as T
from DINO.groundingdino.models import build_model
from DINO.groundingdino.util.slconfig import SLConfig
from DINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def transform_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

class DINO_API:
    def __init__(self, ckpt,device,config_file="DINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"):
        self.model = self.load_model(config_file,ckpt,device=device)


    
    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(
            clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model
    
    def get_grounding_output(self, image, caption, box_threshold=0.3, text_threshold=0.25, with_logits=True):
        image = transform_image(image)
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        with torch.no_grad():
            outputs = self.model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(
                    pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases


    def draw_mask(mask, draw, random_color=False):
        if random_color:
            color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255), 153)
        else:
            color = (30, 144, 255, 153)

        nonzero_coords = np.transpose(np.nonzero(mask))

        for coord in nonzero_coords:
            draw.point(coord[::-1], fill=color)
    
    def draw_box(box, draw, label):
    # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        draw.rectangle(((box[0], box[1]), (box[2], box[3])),
                    outline=color,  width=2)

        if label:
            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((box[0], box[1]), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (box[0], box[1], w + box[0], box[1] + h)
            draw.rectangle(bbox, fill=color)
            draw.text((box[0], box[1]), str(label), fill="white")

            draw.text((box[0], box[1]), label)