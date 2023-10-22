import torch
import nnet
from dataloader.data.imaug import transform, create_operators
from utils.util import *
import time

def infer(args):
    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH
    model = nnet.get_models(args)
    ckpt_path = "/home/sangdt/research/voice/svtr-pytorch/ckpt/SVTR/checkpoints/SVTR.ckpt"
    # ckpt_path = "./ckpt/SVTR_best_200epochs.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    postprocess = nnet.get_postprocess(args)

    start = time.time()
    image = cv2.imread(args.image_test_path)
    h, w = image.shape[0:2]
    wh_ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, wh_ratio)

    norm_img = resize_norm_img(image, max_wh_ratio)

    norm_img = torch.tensor(np.expand_dims(norm_img, axis=0)).to(args.device)

    print(f"normshape: {norm_img.shape}")

    output = model(norm_img)
    output = output[0]
    postprocessed_output = postprocess(output.cpu().detach().numpy())
    print(postprocessed_output)
    end = time.time()-start
    print(f"time: {end}")
    return postprocessed_output





