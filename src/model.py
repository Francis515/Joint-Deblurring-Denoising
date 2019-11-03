import tensorflow as tf


# End-to-end Dense Net
class PairNet:
    def __init__(self, img_size=16, leaky_alpha=0):
        self.img_size = img_size # 16
        self.units = img_size*img_size
        self.alpha = leaky_alpha
        pass

    def forward(self, imgs_blur, imgs_noise):

        # imgs_X (None, 256)
        blur1 = tf.layers.dense(imgs_blur, units=self.units)
        blur1 = tf.nn.leaky_relu(blur1, alpha=self.alpha)
        blur2 = tf.layers.dense(blur1, units=self.units)
        blur2 = tf.nn.leaky_relu(blur2, alpha=self.alpha)

        # blur2 (None, 256)
        noise1 = tf.layers.dense(imgs_noise, units=self.units)
        noise1 = tf.nn.leaky_relu(noise1, alpha=self.alpha)
        noise2 =  tf.layers.dense(noise1, units=self.units)
        noise2 = tf.nn.leaky_relu(noise2, alpha=self.alpha)
        # noise2 (None, 256)

        x1 = tf.concat([blur2,noise2], axis=1)
        # x1 (None, 512)
        x2 = tf.layers.dense(x1, units=self.units*2)
        x2 = tf.nn.leaky_relu(x2, alpha=self.alpha)
        # x2 (None, 512)
        x3 = tf.layers.dense(x2, units=self.units*4)
        x3 = tf.nn.leaky_relu(x3, alpha=self.alpha)
        # x3 (None, 1024)
        x3_ = tf.layers.dense(x3, units=self.units*4)
        x3_ = tf.nn.leaky_relu(x3_, alpha=self.alpha)

        x2_ = tf.layers.dense(x3_, units=self.units*2)
        x2_ = tf.nn.leaky_relu(x2_, alpha=self.alpha)
        # x2_ (None, 512)
        x1_ = tf.layers.dense(x2_, units=self.units)
        x1_ = tf.nn.leaky_relu(x1_, alpha=self.alpha)
        # x1_ (none, 256)
        return [x1_, blur2, noise2]

    def loss(self, gt, pred, blur, noise):
        l1 = 1e-2
        l2 = 1e-3
        loss_blur = tf.reduce_mean(tf.square(blur-gt))
        loss_noise = tf.reduce_mean(tf.square(noise-gt))
        loss_pred = tf.reduce_mean(tf.square(pred-gt))
        loss = loss_pred+loss_blur*l1+loss_noise*l2
        return [loss,loss_pred,loss_noise,loss_blur]




