import torch
import nnet
from dataloader.data.imaug import transform, create_operators
from utils.util import *
from utils import get_args
import time
import glob



def submission_one_sample(args):
    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH
    model = nnet.get_models(args)
    ckpt_path = "/home/sangdt/research/voice/svtr-pytorch/ckpt/SVTR/checkpoints/SVTR.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
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
    imgC, imgH, imgW = (3,48,720)
    max_wh_ratio = imgW / imgH
    model = nnet.get_models(args)
    ckpt_path = "./ckpt/SVTR/checkpoints/SVTR.ckpt"
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
    for subfolder in subfolders:
        image_path += glob.glob(subfolder + "/*.*")
    for i in image_path:
        image = cv2.imread()

    print(images)
    img_num = len(images)
    idx = 0
    batch_num = 6
    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        imgC, imgH, imgW = (3,48,320)
        max_wh_ratio = imgW / imgH

        for ino in range(beg_img_no, end_img_no):
            h, w = images[ino].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img(images[ino], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        
        norm_img_batch = np.concatenate(norm_img_batch)

        norm_img = torch.tensor(norm_img_batch).to(args.device)

        print(f"normshape: {norm_img.shape}")

        output = model(norm_img)
        output = [i[0] for i in output]
        postprocessed_output = postprocess(output.cpu().detach().numpy())
        print(postprocessed_output)
        end = time.time()-start
        print(f"time: {end}")
    return postprocessed_output



if __name__=="__main__":
    args = get_args()
    submission(args)