import torch
import nnet
from dataloader.data.imaug import transform, create_operators
from utils.util import *
from utils import get_args
import time
import glob
import csv



def submission_one_sample(args):
    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH
    model = nnet.get_models(args)
    ckpt_path = "/home/sangdt/research/voice/svtr-pytorch/ckpt/SVTR_kalapa2110/checkpoints/SVTR.ckpt"
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


def submission(args):
    with open('/home/sangdt/research/voice/svtr-pytorch/data/OCR/2110_0.csv', 'a+') as f:
        writer = csv.writer(f,  delimiter=',')
        writer.writerow(["id", "answer"])

        imgC, imgH, imgW = (3,48,720)
        max_wh_ratio = imgW / imgH
        model = nnet.get_models(args)
        model = model.to(args.device)
        ckpt_path = "./ckpt/SVTR_kalapa2110/checkpoints/SVTR.ckpt"
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
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
            # print(output)
            # output = [i[0] for i in output]
            postprocessed_output = postprocess(output.cpu().detach().numpy())
            print(postprocessed_output)

            for i, j in zip(name_batch, postprocessed_output):
                writer.writerow([i, j[0]])


    # return postprocessed_output



if __name__=="__main__":
    args = get_args()
    submission(args)