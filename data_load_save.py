import os
import PIL
import  numpy as np
def loadfile(filename):
    imgs=os.listdir(filename)
    x_train = np.empty((imgs.__len__(), 64, 64, 3), dtype='float32')
    for i in range(len(imgs)):
        img=PIL.Image.open(filename+imgs[i])
        img=np.asarray(img,dtype="float32")
        x_train[i, :, :, :] = img / 127.5 - 1
    return x_train
def savefile(gen_imgs,epoch):
    gen_imgs = 0.5 * gen_imgs + 0.5
    #print(np.uint8(255 * gen_imgs))
    img = PIL.Image.fromarray(np.uint8(255 * gen_imgs))
    
    img.save("./output//%d.png" % epoch)
