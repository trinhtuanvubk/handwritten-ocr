import torch
import nnet
from dataloader.data.imaug import transform, create_operators
from utils.util import *
import time
from pyctcdecode import build_ctcdecoder

vi_dict = ['', 'a', 'A', 'à', 'À', 'ả', 'Ả', 'ã', 'Ã', 'á', 'Á', 'ạ', 'Ạ', 'ă', 'Ă', 'ằ', 'Ằ', 'ẳ', 'Ẳ', 'ẵ', 'Ẵ', 'ắ', 'Ắ', 'ặ', 'Ặ', 'â', 'Â', 'ầ', 'Ầ', 'ẩ', 'Ẩ', 'ẫ', 'Ẫ', 'ấ', 'Ấ', 'ậ', 'Ậ', 'b', 'B', 'c', 'C', 'd', 'D', 'đ', 'Đ', 'e', 'E', 'è', 'È', 'ẻ', 'Ẻ', 'ẽ', 'Ẽ', 'é', 'É', 'ẹ', 'Ẹ', 'ê', 'Ê', 'ề', 'Ề', 'ể', 'Ể', 'ễ', 'Ễ', 'ế', 'Ế', 'ệ', 'Ệ', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'ì', 'Ì', 'ỉ', 'Ỉ', 'ĩ', 'Ĩ', 'í', 'Í', 'ị', 'Ị', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ò', 'Ò', 'ỏ', 'Ỏ', 'õ', 'Õ', 'ó', 'Ó', 'ọ', 'Ọ', 'ô', 'Ô', 'ồ', 'Ồ', 'ổ', 'Ổ', 'ỗ', 'Ỗ', 'ố', 'Ố', 'ộ', 'Ộ', 'ơ', 'Ơ', 'ờ', 'Ờ', 'ở', 'Ở', 'ỡ', 'Ỡ', 'ớ', 'Ớ', 'ợ', 'Ợ', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'ù', 'Ù', 'ủ', 'Ủ', 'ũ', 'Ũ', 'ú', 'Ú', 'ụ', 'Ụ', 'ư', 'Ư', 'ừ', 'Ừ', 'ử', 'Ử', 'ữ', 'Ữ', 'ứ', 'Ứ', 'ự', 'Ự', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'ỳ', 'Ỳ', 'ỷ', 'Ỷ', 'ỹ', 'Ỹ', 'ý', 'Ý', 'ỵ', 'Ỵ', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', "'", "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', ' ']
# prepare decoder and decode logits via shallow fusion
decoder = build_ctcdecoder(
    vi_dict,
    kenlm_model_path='nnet/ngram/address_fix.arpa',  # either .arpa or .bin file
    alpha=0.0,  # tuned on a val set
    beta=2.0,  # tuned on a val set
)

# beam_decoder = nnet.BeamCTCDecoder(nnet.vi_dict, lm_path='nnet/ngram/address.arpa',
#                                  alpha=0.4, beta=4,
#                                  cutoff_top_n=40, cutoff_prob=1.0,
#                                  beam_width=100, num_processes=16,
#                                  blank_index=0)
        
def infer(args):
    # vi_dict = nnet.vi_dict

    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH
    model = nnet.get_models(args)
    ckpt_path = "/home/sangdt/research/voice/svtr-pytorch/ckpt/best_2510/kala_lmdb_fix_aug_2410/checkpoints/SVTR.ckpt"
    # ckpt_path = "./ckpt/SVTR_best_200epochs.ckpt"
    ckpt_path = "/home/ai22/Documents/VUTT/kalapa/handwritten-ocr/ckpt/SVTR_pretrained_large_2610/checkpoints/SVTR.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    postprocess = nnet.get_postprocess(args)

    start = time.time()
    image = cv2.imread(args.image_test_path)
    h, w = image.shape[0:2]
    print(image.shape)
    # wh_ratio = w * 1.0 / h
    # max_wh_ratio = max(max_wh_ratio, wh_ratio)

    norm_img = resize_norm_img(image, max_wh_ratio)
    # norm_img, _ = resize_norm_img(image, image_shape=(image.shape[2], image.shape[0], image.shape[1]))

    norm_img = torch.tensor(np.expand_dims(norm_img, axis=0)).to(args.device)

    print(f"normshape: {norm_img.shape}")

    output = model(norm_img)
    output = output[0]
    print(output.shape)
    # postprocessed_output = postprocess(output.cpu().detach().numpy())
    # print(postprocessed_output.shape)

    # beam_output, _ = beam_decoder.decode(output)
    # postprocessed_output = [output[0].strip() for output in beam_output]
    postprocessed_output = decoder.decode(output[0].cpu().detach().numpy())
    postprocessed_output = postprocessed_output.replace('blank',"")
    print(postprocessed_output)
    end = time.time()-start
    print(f"time: {end}")
    return postprocessed_output





