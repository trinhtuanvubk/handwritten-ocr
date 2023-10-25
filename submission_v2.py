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

# prepare decoder and decode logits via shallow fusion
hotwords = ["Đg", "Ng", "ng", "Tdp", "TDP"]
decoder = build_ctcdecoder(
    nnet.vi_dict,
    kenlm_model_path='nnet/ngram/address_fix.arpa',  # either .arpa or .bin file
    alpha=0.3,  # tuned on a val set
    beta=2.0,  # tuned on a val set
    hotwords = hotwords,
    hotword_weight=5.0,
)

def submission(args, use_lm=True):
    with open('/home/sangdt/research/voice/svtr-pytorch/data/OCR/2510_ngram_2.csv', 'a+') as f:
        writer = csv.writer(f,  delimiter=',')
        writer.writerow(["id", "answer"])

        imgC, imgH, imgW = (3,48,720)
        max_wh_ratio = imgW / imgH
        model = nnet.get_models(args)
        model = model.to(args.device)
        # ckpt_path = "./ckpt/SVTR_kalapa2110/checkpoints/SVTR.ckpt"
        ckpt_path = "./ckpt/best_2510/kala_lmdb_fix_aug_2410/checkpoints/SVTR.ckpt"
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        postprocess = nnet.get_postprocess(args)

        start = time.time()

        norm_img_batch = []
        all_folder = os.listdir("./data/OCR/public_test/images/")
        

        # Get a list of all subfolders
        subfolders = glob.glob("./data/OCR/public_test/images/*")

        # Get a list of all images in all subfolders
        image_path = []
        images = []
        for subfolder in subfolders:
            image_path += glob.glob(subfolder + "/*.*")
        for i in image_path:
            image = cv2.imread(i)
            images.append({'image_name': "/".join(i.rsplit("/", 2)[-2:]), 'image': image})

        # print(images)
        img_num = len(images)
        idx = 0
        batch_num = 6
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
            if use_lm:
                try:
                    with multiprocessing.get_context("fork").Pool() as pool:
                        postprocessed_output = decoder.decode_batch(pool, output.cpu().detach().numpy())
                    # postprocessed_output = decoder.decode(output[0].cpu().detach().numpy())
                    postprocessed_output = [i.replace('blank',"") for i in postprocessed_output]
                    print(postprocessed_output)

                    for i, j in zip(name_batch, postprocessed_output):
                        print(i, j)
                        # if not isinstance(j, list):
                        #     j = [""]
                        writer.writerow([i, j])
                except:
                    print("hihi")

            # output = [i[0] for i in output]
            else:
                postprocessed_output = postprocess(output.cpu().detach().numpy())
            
                print(postprocessed_output)

                for i, j in zip(name_batch, postprocessed_output):
                    print(i)
                    writer.writerow([i, j[0]])


    # return postprocessed_output



if __name__=="__main__":
    args = get_args()
    submission(args, use_lm=True)