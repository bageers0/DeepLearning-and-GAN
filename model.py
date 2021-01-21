import os
import tensorflow as tf
import numpy as np
from data_load_save import loadfile,savefile
class Gannetworks():
    def __init__(self,epoch,lr_rate,baech_size):
      self.epoch=epoch
      self.lr_rate=lr_rate
      self.baech_size=baech_size
    def model_placeholder():
      norm_z=tf.placeholder(dtype=tf.float32,shape=(None,100))
      input_real=tf.placeholder(dtype=tf.float32,shape=(None,64,64,3))
      return norm_z,input_real
    def generative(self,norm_z,is_train=True):
      with tf.variable_scope("generative",reuse=not is_train):
        #img=dence(norm_z,100,4*4*512)
        img = tf.layers.dense(norm_z, 8 * 8* 256)
        #img=tf.nn.relu(img)
        img=tf.reshape(img,(-1,8,8,256))#-1<=>self.beach_size
        img=tf.layers.batch_normalization(img,training=is_train)
        img=tf.nn.relu(img)
        #img=upscale(img,256,512)
        img = tf.layers.conv2d_transpose(img, 128, 5, strides=2, padding='same')
        #img=conv2d(img,256,128,1)
        img=tf.layers.batch_normalization(img,training=is_train)
        img = tf.nn.relu(img)
        #img = upscale(img,128,256)
        img = tf.layers.conv2d_transpose(img, 64, 5, strides=2, padding='same')
        #img = conv2d(img,128,64,1)
        img = tf.layers.batch_normalization(img,training=is_train)
        img = tf.nn.relu(img)
        #img = upscale(img,64,128)
        img = tf.layers.conv2d_transpose(img,3, 5, strides=2, padding='same')
        img = tf.nn.tanh(img)
        return img
    def discriminator(self,imgs,dropout=0.8,reuse=False):
      with tf.variable_scope("discriminator",reuse=reuse):
        #out_Labal=conv2d(imgs,3,128,2)
        out_Labal = tf.layers.conv2d(imgs, 64, 5, strides=2, padding='same',kernel_initializer= tf.contrib.layers.xavier_initializer(seed=2))
        out_Labal=tf.maximum(0.2*out_Labal,out_Labal)
        out_Labal=tf.nn.dropout(out_Labal,keep_prob=dropout)
        out_Labal = tf.layers.conv2d(out_Labal, 128, 5, strides=2, padding='same', kernel_initializer= tf.contrib.layers.xavier_initializer(seed=2))
        out_Labal = tf.layers.batch_normalization(out_Labal,training=True)
        out_Labal = tf.maximum( 0.2 * out_Labal,out_Labal)
        out_Labal = tf.nn.dropout(out_Labal, keep_prob=dropout)
        out_Labal = tf.layers.conv2d(out_Labal, 256, 5, strides=2, padding='same',kernel_initializer= tf.contrib.layers.xavier_initializer(seed=2))
        out_Labal = tf.layers.batch_normalization(out_Labal,training=True)
        out_Labal = tf.maximum( 0.2 * out_Labal,out_Labal)
        out_Labal = tf.nn.dropout(out_Labal, keep_prob=dropout)
        out_Labal=tf.reduce_mean(out_Labal,[1,2])     
        out_Labal = tf.layers.dense(out_Labal,1)#globalaveragepooling2D
        out_Labals=tf.nn.sigmoid(out_Labal)
        return out_Labals,out_Labal
    def model_loss(self,norms_z,Img):
      g_img=self.generative(norms_z,True)
      d_real0,d_real=self.discriminator(Img,0.8,False)
      d_fack0,d_fack=self.discriminator(g_img,0.8,True)
      print(d_real.shape,d_real0.shape,d_fack0.shape,d_fack.shape,g_img.shape)
      g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fack,labels=tf.ones_like(d_fack0)))
      d_loss=0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real,labels=tf.ones_like(d_real0)))+0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fack,labels=tf.zeros_like(d_fack0)))
      print(g_loss.shape,d_loss.shape)
      return g_loss,d_loss,g_img,d_real0,d_fack0
    def model_train(self):
      norm_zs,input_reals=self.model_placeholder()
      G_loss,D_loss,G_img,D_real,D_fack=self.model_loss(norm_zs,input_reals)
      print(G_loss.shape,D_loss.shape)
      V_vars=tf.trainable_variables()
      d_vars = [var for var in V_vars if var.name.startswith('discriminator')]
      g_vars = [var for var in V_vars if var.name.startswith('generative')]
      print(d_vars,g_vars)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        D_train = tf.train.AdamOptimizer(self.lr_rate,beta1=0.5).minimize(D_loss,var_list=d_vars)
        G_train = tf.train.AdamOptimizer(self.lr_rate,beta1=0.5).minimize(G_loss,var_list=g_vars)
      img_path="D:\\BaiduNetdiskDownload\\data\\data\\"#your image path
      train_real = loadfile(img_path)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_local_variables() )
        for i in range(self.epoch):
            Norm_zs=np.random.uniform(-1,1,(self.baech_size,100))
            Input_reals=train_real[np.random.randint(0,train_real.shape[0],self.baech_size)]
            _=sess.run(D_train,feed_dict={norm_zs:Norm_zs,input_reals:Input_reals})
            _=sess.run(G_train,feed_dict={norm_zs:Norm_zs,input_reals:Input_reals})
            if i%5==0:
                train_loss_d = D_loss.eval({norm_zs:Norm_zs,input_reals:Input_reals})
                train_loss_g = G_loss.eval({norm_zs:Norm_zs,input_reals:Input_reals})
                savefile(sess.run(G_img, feed_dict={norm_zs: Norm_zs})[0], i)
                print("Epoch {}/{}...".format(i, self.epoch),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
