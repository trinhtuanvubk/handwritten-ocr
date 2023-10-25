import torch
import nnet
from dataloader.data.imaug import transform, create_operators
from utils.util import *
import time

beam_decoder = nnet.BeamCTCDecoder(nnet.vi_dict, lm_path="./nnet/ngram/vi_lm_4grams.bin",
                                 alpha=0, beta=2,
                                 cutoff_top_n=40, cutoff_prob=1.0,
                                 beam_width=100, num_processes=16,
                                 blank_index=0)
        
def infer(args):
    # vi_dict = nnet.vi_dict

    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH
    model = nnet.get_models(args)
    ckpt_path = "/home/sangdt/research/voice/svtr-pytorch/ckpt/SVTR_kalapa2110/checkpoints/SVTR.ckpt"
    # ckpt_path = "./ckpt/SVTR_best_200epochs.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    postprocess = nnet.get_postprocess(args)

    start = time.time()
    image = cv2.imread(args.image_test_path)
    h, w = image.shape[0:2]
    print(image.shape)
    wh_ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, wh_ratio)

    norm_img = resize_norm_img(image, max_wh_ratio)
    # norm_img, _ = resize_norm_img(image, image_shape=(image.shape[2], image.shape[0], image.shape[1]))

    norm_img = torch.tensor(np.expand_dims(norm_img, axis=0)).to(args.device)

    print(f"normshape: {norm_img.shape}")

    output = model(norm_img)
    output = output[0]
    print(output.shape)
    # postprocessed_output = postprocess(output.cpu().detach().numpy())
    # print(postprocessed_output.shape)
    beam_output, _ = beam_decoder.decode(output)
    postprocessed_output = [output[0].strip() for output in beam_output]
    print(postprocessed_output)
    end = time.time()-start
    print(f"time: {end}")
    return postprocessed_output





