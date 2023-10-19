import torch
import nnet
from dataloader.data.imaug import transform, create_operators
from .util import *


def infer(args):
    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH
    model = nnet.get_models(args)
    ckpt_path = ""
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.eval()
    postprocess = nnet.get_postprocess(args)

    image = cv2.imread(args.image_test_path)
    h, w = image.shape[0:2]
    wh_ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, wh_ratio)

    norm_img = resize_norm_img(image, max_wh_ratio)

    output = model(norm_img)[0]
    postprocessed_output = postprocess(output.cpu().detach().numpy())
    print(postprocessed_output)

    return postprocessed_output





