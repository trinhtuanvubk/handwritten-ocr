import torch

import cv2
from utils.util import *
from pyctcdecode import build_ctcdecoder
import time
import glob
import csv
from utils import get_args
import nnet
from utils.preprocess import detect_text_lines

vi_dict = ['', 'a', 'A', 'à', 'À', 'ả', 'Ả', 'ã', 'Ã', 'á', 'Á', 'ạ', 'Ạ', 'ă', 'Ă', 'ằ', 'Ằ', 'ẳ', 'Ẳ', 'ẵ', 'Ẵ', 'ắ', 'Ắ', 'ặ', 'Ặ', 'â', 'Â', 'ầ', 'Ầ', 'ẩ', 'Ẩ', 'ẫ', 'Ẫ', 'ấ', 'Ấ', 'ậ', 'Ậ', 'b', 'B', 'c', 'C', 'd', 'D', 'đ', 'Đ', 'e', 'E', 'è', 'È', 'ẻ', 'Ẻ', 'ẽ', 'Ẽ', 'é', 'É', 'ẹ', 'Ẹ', 'ê', 'Ê', 'ề', 'Ề', 'ể', 'Ể', 'ễ', 'Ễ', 'ế', 'Ế', 'ệ', 'Ệ', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'ì', 'Ì', 'ỉ', 'Ỉ', 'ĩ', 'Ĩ', 'í', 'Í', 'ị', 'Ị', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ò', 'Ò', 'ỏ', 'Ỏ', 'õ', 'Õ', 'ó', 'Ó', 'ọ', 'Ọ', 'ô', 'Ô', 'ồ', 'Ồ', 'ổ', 'Ổ', 'ỗ', 'Ỗ', 'ố', 'Ố', 'ộ', 'Ộ', 'ơ', 'Ơ', 'ờ', 'Ờ', 'ở', 'Ở', 'ỡ', 'Ỡ', 'ớ', 'Ớ', 'ợ', 'Ợ', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'ù', 'Ù', 'ủ', 'Ủ', 'ũ', 'Ũ', 'ú', 'Ú', 'ụ', 'Ụ', 'ư', 'Ư', 'ừ', 'Ừ', 'ử', 'Ử', 'ữ', 'Ữ', 'ứ', 'Ứ', 'ự', 'Ự', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'ỳ', 'Ỳ', 'ỷ', 'Ỷ', 'ỹ', 'Ỹ', 'ý', 'Ý', 'ỵ', 'Ỵ', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', "'", "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', ' ']
# prepare decoder and decode logits via shallow fusion
decoder = build_ctcdecoder(
    vi_dict,
    kenlm_model_path='ckpt/ngram/address_fix_811.bin',  # either .arpa or .bin file
    alpha=0.3,  # tuned on a val set
    beta=2.0,  # tuned on a val set
)

def check_num(word:str):
    for i in word:
        if i.isdigit():
            return True
    return False



def submission(args, use_lm=True):
    model = nnet.get_models(args)
    model = model.to(args.device)
    ckpt_path = "./ckpt/SVTR_711_Son/SVTR_0811_0.92.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open('./onnx.csv', 'a+') as f:
        writer = csv.writer(f,  delimiter=',')
        writer.writerow(["id", "answer"])

        imgC, imgH, imgW = (3,48,720)
        max_wh_ratio = imgW / imgH
        
        postprocess = nnet.get_postprocess(args)

        start = time.time()

        norm_img_batch = []

        # Get a list of all subfolders
        subfolders = glob.glob("./data/public_test/images/*")
        # subfolders = glob.glob("./data/kalapa_fixed_raw/train/images_note")

        # Get a list of all images in all subfolders
        image_path = []
        images = []
        for subfolder in subfolders:
            image_path += glob.glob(subfolder + "/*.*")
        for i in image_path:
            print(i)
            start = time.time()
            image = cv2.imread(i)
            image_name =  "/".join(i.rsplit("/", 2)[-2:])
            image = detect_text_lines(image)
            if image is None:
                writer.writerow([image_name, ""])
                continue

            norm_img = resize_norm_img(image, max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_torch = torch.tensor(norm_img).to(args.device)
            print(norm_img_torch.shape)
            logits = model(norm_img_torch)[0]
            logits = logits.cpu().detach().numpy()
            print(logits.shape)
            # break
            if args.decode_type == use_lm:
                try:
                    text = decoder.decode(logits, beam_prune_logp=-15, token_min_logp=-7)
                    text = text.replace("  "," ").replace("uỵ", "ụy")
                    writer.writerow([image_name, text])
                except:
                    print("hihi")

            elif args.decode_type=="both":
                text_last_output = ""
                # do lm first
                
                text_lm_output = decoder.decode(logits[0], beam_prune_logp=-15, token_min_logp=-7)
                    # postprocessed_output = decoder.decode(output[0].cpu().detach().numpy())
                text_lm_output = text_lm_output.replace("  "," ").replace("uỵ", "ụy")

                #  do normal
                normal_output = postprocess(logits)
                text_normal_output = normal_output[0][0]
                print(text_normal_output)

                # print(batch_nolm_output)
                if len(text_lm_output.split(" ")) > len(text_normal_output.split(" ")):
                    text_last_output = text_lm_output
                elif len(text_lm_output.split(" ")) < len(text_normal_output.split(" ")):
                    text_last_output = text_normal_output
                else:
                    text_last_output = text_lm_output
                    for lm_word, nolm_word in zip(text_lm_output.split(" "), text_normal_output.split(" ")):
                        if check_num(lm_word):
                            text_last_output.replace(lm_word, nolm_word)
                print(f"process time: {time.time()-start}")
                writer.writerow([image_name, text_last_output])
           

if __name__=="__main__":
    args = get_args()
    # args.device = torch.device("cpu")
    args.decode_type = "both"
    submission(args, use_lm=False)


