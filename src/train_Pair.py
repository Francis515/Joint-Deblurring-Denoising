import tensorflow as tf
from model import PairNet
import utils
from glob import glob
from skimage import io
import numpy  as np
import argparse

# Arguments
parser =argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='pair_overfit_alpha0')
parser.add_argument('--load',type=str)
parser.add_argument('--overfit', type=bool, default=True)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs_begin', type=int, default=1)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()


# Data
img_size = 16
np.random.seed(0)
train_patches_gt = []
train_patches_blur = []
train_patches_noise = []
base_path = 'F://university/course/4th_1/acquisition/prj'
if args.overfit:
    train_names = glob(base_path+'/overfit/*.jpg')
else:
    train_names = glob(base_path+'/train/*.jpg')
for i in range(len(train_names)):
    name = train_names[i]
    img = io.imread(name,as_grey=True)
    if img.dtype == np.uint8:
        img = np.float32(img/255.0)
    patches_gt,_ = utils.Img2patch(img,patch_size=img_size)
    train_patches_gt.append(patches_gt)

    # blur
    kernel = utils.kernel_generator(kernel_size=36)
    img_blur = utils.blur(img,kernel)
    patches_blur, _ = utils.Img2patch(img_blur, patch_size=img_size)
    train_patches_blur.append(patches_blur)

    # noise
    sigma = np.random.randint(20,30)
    img_noise = utils.noise(img, sigma)
    patches_noise, _ = utils.Img2patch(img_noise, patch_size=img_size)
    train_patches_noise.append(patches_noise)
    if i%100==0:
        print(i,' loaded')

# (2000, x, 256) -> (sigma, 256)
# Do not use cancatenate, too slow
train_patches_gt = np.vstack(train_patches_gt)
train_patches_blur = np.vstack(train_patches_blur)
train_patches_noise = np.vstack(train_patches_noise)


# Model
imgs_gt = tf.placeholder(dtype=tf.float32, shape=(None, img_size*img_size))
imgs_blur = tf.placeholder(dtype=tf.float32, shape=(None, img_size*img_size))
imgs_noise = tf.placeholder(dtype=tf.float32, shape=(None, img_size*img_size))
model = PairNet(img_size=img_size)
pred = model.forward(imgs_blur, imgs_noise)
loss = model.loss(imgs_gt,pred[0],pred[1],pred[2])

optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
step = optimizer.minimize(loss[0])

# Session
sess = tf.Session(config=tf.ConfigProto())
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=10)
if args.load:
    saver.restore(sess,args.load)

# Train
samples = train_patches_gt.shape[0]
indx_list = np.arange(samples)
np.random.shuffle(indx_list)
for ep in range(args.epochs_begin, args.epochs_begin+args.epochs):
    for iter in range(samples//args.batch_size):
        indx = indx_list[iter*args.batch_size:(iter+1)*args.batch_size]
        batch_blur = train_patches_blur[indx]
        batch_noise = train_patches_noise[indx]
        batch_gt = train_patches_gt[indx]

        _ = sess.run(step, feed_dict={imgs_blur:batch_blur,imgs_noise:batch_noise,imgs_gt:batch_gt})
        if iter%10 == 0:
            _loss = sess.run(loss, feed_dict={imgs_blur:batch_blur,imgs_noise:batch_noise,imgs_gt:batch_gt})
            print("epoch %d, iteration %d" %(ep, iter))
            log = "Total loss:%.4f, Pred_loss:%.4f, Blur_loss:%.4f, Noise_loss:%.4f\n" % \
                  (_loss[0],_loss[1],_loss[2],_loss[3])
            print(log)
            log_file = open(base_path+'/model/log/'+ args.name+'_loss.txt','a')
            log_file.write(log)
            log_file.close()
    if ep%500 == 0:
        saver.save(sess, base_path+'/model/Pair/'+args.name+'_%d.ckpt'%ep)

