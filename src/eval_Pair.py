from model import PairNet
import tensorflow as tf
import utils
from glob import glob
import numpy as np
import skimage.io as io
from skimage.measure import compare_psnr,compare_ssim
import argparse

# Arguments
parser =argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='pair_relu_completed')
parser.add_argument('--load',type=str, default='F://university/course/4th_1/acquisition/prj/model/Pair/pair_relu_1600.ckpt')
parser.add_argument('--overfit',type=bool, default=False)
args = parser.parse_args()

# Model
img_size=16
imgs_gt = tf.placeholder(dtype=tf.float32, shape=(None, img_size*img_size))
imgs_blur = tf.placeholder(dtype=tf.float32, shape=(None, img_size*img_size))
imgs_noise = tf.placeholder(dtype=tf.float32, shape=(None, img_size*img_size))
model = PairNet(img_size=img_size)
pred = model.forward(imgs_blur, imgs_noise)

# Session
sess = tf.Session(config=tf.ConfigProto())
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10)
if args.load:
    saver.restore(sess,args.load)

# Data
np.random.seed(0)

base_path = 'F://university/course/4th_1/acquisition/prj'
if args.overfit:
    eval_names = glob(base_path + '/overfit/*.jpg')
else:
    eval_names = glob(base_path+'/eval/Pair/*.jpg')

SSIM_list = []
PSNR_list = []
for name in eval_names:
    img = io.imread(name,as_grey=True)
    if img.dtype == np.uint8:
        img = np.float32(img/255.0)
    #io.imsave(name[:-4]+'_grey.png',img)
    patches_gt,new_shape = utils.Img2patch(img,patch_size=img_size)

    # Blur
    kernel = utils.kernel_generator(kernel_size=36)
    img_blur = utils.blur(img,kernel)
    io.imsave(name[:-4]+'_blur.png',img_blur)
    patches_blur, _ = utils.Img2patch(img_blur, patch_size=img_size)

    # Noise
    sigma = np.random.randint(20,30)
    img_noise = utils.noise(img, sigma)
    io.imsave(name[:-4]+'_noise.png',img_noise)
    patches_noise, _ = utils.Img2patch(img_noise, patch_size=img_size)

    _pred = sess.run(pred, feed_dict={imgs_blur:patches_blur,imgs_noise:patches_noise,imgs_gt:patches_gt})
    img_pred = utils.Patch2img(_pred[0],img_size,new_shape)
    img_pred = np.clip(img_pred,0,1)
    # fill the black points
    for y in range(img_pred.shape[0]):
        for x in range(img_pred.shape[1]):
            if img_pred[y][x] < 1e-8:
                if y==0:
                    img_pred[y][x] = img_pred[y+1][x]
                elif y==img_pred.shape[0]-1:
                    img_pred[y][x] = img_pred[y-1][x]
                elif x==0:
                    img_pred[y][x] = img_pred[y][x+1]
                elif x==img_pred.shape[1]-1:
                    img_pred[y][x] = img_pred[y][x-1]
                else:
                    img_pred[y][x] = img_pred[y + 1][x] / 4 + img_pred[y][x + 1] / 4 + img_pred[y - 1][x] / 4 + \
                                     img_pred[y][x - 1] / 4
    io.imsave(name[:-4]+'_'+args.name+'_pred.png',img_pred)

    img_new = img[:new_shape[0],:new_shape[1]]
    psnr = compare_psnr(img_new,img_pred)
    ssim = compare_ssim(img_new,img_pred)
    metrics = "PSNR:%.4f, SSIM:%.4f" %(psnr, ssim)
    print(metrics)
    PSNR_list.append(psnr)
    SSIM_list.append(ssim)

metrics_mean = "PSNR:%.4f, SSIM:%.4f\n" % (np.mean(PSNR_list), np.mean(SSIM_list))
log_eval = open(base_path+'/eval/log/'+args.name+'_eval.txt','w+')
log_eval.write(metrics_mean)
log_eval.close()




