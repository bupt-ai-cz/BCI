import os
import cv2 as cv
from PIL import Image
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from tqdm import tqdm
import argparse

def parse_opt():
#Set train options
    parser = argparse.ArgumentParser(description='Evaluate options')
    parser.add_argument('--result_path', type=str, default='./results/pyramidpix2pix', help='results saved path')
    opt = parser.parse_args()
    return opt
opt = parse_opt()


def psnr_and_ssim(result_path):
    psnr = []
    ssim = []
    for i in tqdm(os.listdir(os.path.join(result_path,'test_latest/images'))):
        if 'fake_B' in i:
            try:
                fake = cv.imread(os.path.join(result_path,'test_latest/images',i))
                real = cv.imread(os.path.join(result_path,'test_latest/images',i.replace('fake_B','real_B')))
                PSNR = peak_signal_noise_ratio(fake, real)
                psnr.append(PSNR)
                SSIM = structural_similarity(fake, real, multichannel=True)
                ssim.append(SSIM)
            except:
                print("there is something wrong with " + i)
        else:
            continue
    average_psnr=sum(psnr)/len(psnr)
    average_ssim=sum(ssim)/len(ssim)
    print("The average psnr is " + str(average_psnr))
    print("The average ssim is " + str(average_ssim))

psnr_and_ssim(opt.result_path)