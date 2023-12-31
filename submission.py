import torch
import nnet
from dataloader.data.imaug import transform, create_operators
from utils.util import *
from utils import get_args
import time
import glob
import csv
from pyctcdecode import build_ctcdecoder
import multiprocessing

from utils.preprocess import detect_text_lines

# prepare decoder and decode logits via shallow fusion
hotwords = ["Đg", "Ng", "ng", "Tdp", "TDP"]
vi_dict = ['', 'a', 'A', 'à', 'À', 'ả', 'Ả', 'ã', 'Ã', 'á', 'Á', 'ạ', 'Ạ', 'ă', 'Ă', 'ằ', 'Ằ', 'ẳ', 'Ẳ', 'ẵ', 'Ẵ', 'ắ', 'Ắ', 'ặ', 'Ặ', 'â', 'Â', 'ầ', 'Ầ', 'ẩ', 'Ẩ', 'ẫ', 'Ẫ', 'ấ', 'Ấ', 'ậ', 'Ậ', 'b', 'B', 'c', 'C', 'd', 'D', 'đ', 'Đ', 'e', 'E', 'è', 'È', 'ẻ', 'Ẻ', 'ẽ', 'Ẽ', 'é', 'É', 'ẹ', 'Ẹ', 'ê', 'Ê', 'ề', 'Ề', 'ể', 'Ể', 'ễ', 'Ễ', 'ế', 'Ế', 'ệ', 'Ệ', 'f', 'F', 'g', 'G', 'h', 'H', 'i', 'I', 'ì', 'Ì', 'ỉ', 'Ỉ', 'ĩ', 'Ĩ', 'í', 'Í', 'ị', 'Ị', 'j', 'J', 'k', 'K', 'l', 'L', 'm', 'M', 'n', 'N', 'o', 'O', 'ò', 'Ò', 'ỏ', 'Ỏ', 'õ', 'Õ', 'ó', 'Ó', 'ọ', 'Ọ', 'ô', 'Ô', 'ồ', 'Ồ', 'ổ', 'Ổ', 'ỗ', 'Ỗ', 'ố', 'Ố', 'ộ', 'Ộ', 'ơ', 'Ơ', 'ờ', 'Ờ', 'ở', 'Ở', 'ỡ', 'Ỡ', 'ớ', 'Ớ', 'ợ', 'Ợ', 'p', 'P', 'q', 'Q', 'r', 'R', 's', 'S', 't', 'T', 'u', 'U', 'ù', 'Ù', 'ủ', 'Ủ', 'ũ', 'Ũ', 'ú', 'Ú', 'ụ', 'Ụ', 'ư', 'Ư', 'ừ', 'Ừ', 'ử', 'Ử', 'ữ', 'Ữ', 'ứ', 'Ứ', 'ự', 'Ự', 'v', 'V', 'w', 'W', 'x', 'X', 'y', 'Y', 'ỳ', 'Ỳ', 'ỷ', 'Ỷ', 'ỹ', 'Ỹ', 'ý', 'Ý', 'ỵ', 'Ỵ', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '!', '"', '#', '$', '%', '&', "'", "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ', ' ']

decoder = build_ctcdecoder(
    vi_dict,
    kenlm_model_path='ckpt/ngram/address_fix_811.bin',  # either .arpa or .bin file
    # kenlm_model_path=None,
    alpha=0.3,  # tuned on a val set
    beta=2.0,  # tuned on a val set
    # hotwords = hotwords,
    # hotword_weight=5.0,
)

def check_num(word:str):
    for i in word:
        if i.isdigit():
            return True
    return False

def submission(args, use_lm=True):
    with open('./1611_0811.csv', 'a+') as f:
        writer = csv.writer(f,  delimiter=',')
        writer.writerow(["id", "answer"])

        imgC, imgH, imgW = (3,48,720)
        max_wh_ratio = imgW / imgH
        model = nnet.get_models(args)
        model = model.to(args.device)
        # ckpt_path = "./ckpt/SVTR_kalapa2110/checkpoints/SVTR.ckpt"
        # ckpt_path = "./ckpt/SVTR_kalapa_611/checkpoints/SVTR.ckpt"
        # ckpt_path = "./ckpt/SVTR_111123/SVTR_111123.ckpt"
        ckpt_path = "./ckpt/SVTR_711_Son/SVTR_0811_0.92.ckpt"
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
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
            image = cv2.imread(i)
            image_name =  "/".join(i.rsplit("/", 2)[-2:])
            image = detect_text_lines(image)
            if image is None:
                writer.writerow([image_name, ""])
                continue
            os.makedirs("./test_detect_line", exist_ok=True)
            cv2.imwrite(os.path.join("./test_detect_line", i.split("/", 4)[-1].replace("/","_")), image)
            images.append({'image_name': image_name, 'image': image})

        

        # print(images)
        img_num = len(images)
        idx = 0
        batch_num = 1
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            name_batch = []
            imgC, imgH, imgW = (3,48,720)
            max_wh_ratio = imgW / imgH

            # for ino in range(beg_img_no, end_img_no):
            #     h, w = images[ino].shape[0:2]
            #     wh_ratio = w * 1.0 / h
            #     max_wh_ratio = max(max_wh_ratio, wh_ratio)
            print(max_wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = resize_norm_img(images[ino]['image'], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
                name_batch.append(images[ino]['image_name'])
            
            norm_img_batch = np.concatenate(norm_img_batch)
            print(norm_img_batch.shape)
            norm_img_torch = torch.tensor(norm_img_batch).to(args.device)

            print(f"normshape: {norm_img_torch.shape}")

            output = model(norm_img_torch)[0]
            print(output.shape)
            print(output)
            if args.decode_type == use_lm:
                try:
                    with multiprocessing.get_context("fork").Pool() as pool:
                        postprocessed_output = decoder.decode_batch(pool, output.cpu().detach().numpy())
                                                                    # beam_prune_logp=-15,
                                                                    # token_min_logp=-7)
                    # postprocessed_output = decoder.decode(output[0].cpu().detach().numpy())
                    postprocessed_output = [i.replace("  "," ").replace("uỵ", "ụy") for i in postprocessed_output]
                    print(postprocessed_output)

                    for i, j in zip(name_batch, postprocessed_output):
                        print(i, j)
                        # if not isinstance(j, list):
                        #     j = [""]
                        writer.writerow([i, j])
                except:
                    print("hihi")

            # output = [i[0] for i in output]
            elif args.decode_type=="nolm":
                postprocessed_output = postprocess(output.cpu().detach().numpy())
            
                print(postprocessed_output)

                for i, j in zip(name_batch, postprocessed_output):
                    print(i)
                    writer.writerow([i, j[0]])
            elif args.decode_type=="both":
                batch_last_output = []
                # do lm first
                with multiprocessing.get_context("fork").Pool() as pool:
                    batch_lm_output = decoder.decode_batch(pool, output.cpu().detach().numpy(),
                                                                    beam_prune_logp=-15,
                                                                    token_min_logp=-7)
                    # postprocessed_output = decoder.decode(output[0].cpu().detach().numpy())
                    batch_lm_output = [i.replace("  "," ").replace("uỵ", "ụy") for i in batch_lm_output]

                #  do normal
                batch_nolm_output = postprocess(output.cpu().detach().numpy())
                batch_nolm_output = [i[0] for i in batch_nolm_output]

                # print(batch_nolm_output)
                for lm_output, nolm_output in zip(batch_lm_output, batch_nolm_output):

                    if len(lm_output.split(" ")) > len(nolm_output.split(" ")):
                        last_output = lm_output
                    elif len(lm_output.split(" ")) < len(nolm_output.split(" ")):
                        last_output = nolm_output
                    else:
                        last_output = lm_output
                        for lm_word, nolm_word in zip(lm_output.split(" "), nolm_output.split(" ")):
                            if check_num(lm_word):
                                last_output.replace(lm_word, nolm_word)
                    batch_last_output.append(last_output)

                for i, j in zip(name_batch, batch_last_output):
                    print(i)
                    writer.writerow([i, j])

            

    # return postprocessed_output

    
    

if __name__=="__main__":
    args = get_args()
    args.device = torch.device("cpu")
    args.decode_type = "both"
    submission(args, use_lm=False)


