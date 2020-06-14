#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import os, keras
from keras import backend as K
import keras, datetime, os, functools
from PIL import Image as PIL_Image
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

def load_image(path, mode=None):
    raw = PIL_Image.open(path)
    if mode is not None:
        raw = raw.convert(mode)
        img = np.expand_dims(img, axis=2)
    img = np.array(raw)
    img = np.array(img).astype('float32')/255
    return img

def load_images(directory, mode=None):
    data = []
    for path in sorted(os.listdir(directory)):
        if path.endswith(".png"):
            data.append(load_image(os.path.join(directory,path), mode))
    return data

raw_images_rgb = load_images("example/case2/small_set/")
print("[+] Images:", len(raw_images_rgb), raw_images_rgb[0].shape)

target_model_path = os.path.join("example", "case2", "capcha_solver_100pic.h5")
with tf.variable_scope("cn1", reuse=tf.AUTO_REUSE):
    target_model = keras.models.load_model(target_model_path)
    target_model_weights = target_model.get_weights()
    print(target_model.summary())
    
    
def apply_mask_trigger(tnsr_input, tnsr_mask, tnsr_delta, gvar):
    con_mask = tf.tanh(tnsr_mask)/2.0 + 0.5
    nc_mask = np.zeros((gvar["h"], gvar["w"]), dtype=np.float32) + 1
    con_mask = con_mask * nc_mask
    use_mask = tf.tile(tf.reshape(con_mask, (1,gvar["h"],gvar["w"],1)), [1,1,1,gvar["rgb"]])
    return input_tensor * (1 - use_mask) + tnsr_delta * use_mask

def inject_trojan_cifar(raw_input_tensor, mask, delta):
    image_shape = list(raw_input_tensor.shape)[1:]
    print("image_shape:", image_shape)
    h, w, rgb = tuple(image_shape)
    con_mask = tf.tanh(mask)/2.0 + 0.5
    nc_mask = np.zeros((h, w), dtype=np.float32) + 1
    con_mask = con_mask * nc_mask
    use_mask = tf.tile(tf.reshape(con_mask, (1,h,w,1)), [1,1,1,rgb])
    trojan_image = raw_input_tensor * (1 - use_mask) + delta * use_mask
    #print(trojan_image,raw_input_tensor,delta,use_mask, image_shape)
    #trojan_image = tf.clip_by_value(trojan_image, self.l_bounds, self.h_bounds)
    return trojan_image

input_tensor_shape = (None, 36, 150, 3)
(batch, h,w,rgb) = input_tensor_shape
# delta_init = np.random.rand(functools.reduce(lambda x,y:x * y,image_shape)).reshape(image_shape)
delta_init = np.ones((h,w,rgb))
mask_init = np.zeros((h,w))
r=1.5
for i in range(h):
    for j in range(w):
        if not( j >= w/r and j < w*(r-1)/r  and i >= h/r and i < h*(r-1)/r):
            mask_init[i,j] = 0.8

lr = 2e-3
target_output = np.zeros([64,88])
for i in range(64):
    target_output[i][0] = 0.25
    target_output[i][23] = 0.25
    target_output[i][46] = 0.25
    target_output[i][69] = 0.25
    
#target_model.pop()


from IPython.display import Image, display
def displayInline(res_img_dir, e, rtrojan_image):
    if not os.path.exists(res_img_dir):
        os.makedirs(res_img_dir)
    RE_img = os.path.join(res_img_dir, 'adv_step%d_all.png' % e)
    imgs = [array_to_img(rtrojan_image[i]) for i in range(0,len(rtrojan_image),5)]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = PIL_Image.fromarray(imgs_comb)
    imgs_comb.save(RE_img) 
    display(Image(filename=RE_img))
    
def sess_hook(e, arg):
    if e % 1000 == 0:
        (routput, rloss,rloss1,rloss2,rloss3, rtrojan_image, rdelta, rmask) = arg
        #res_img_dir = os.path.join(self.config["root_dir"], self.config["res_dir"], 'imgs/')
        print(e, rloss,rloss1,rloss2,rloss3, routput[0], routput[0,[0,23,46,69]])#list(np.argmax(np.reshape(routput,[-1,4,22]), axis=2)))
        displayInline(os.path.join("static", 'result', 'imgs'), e, rtrojan_image)
                 
def solve_trigger(args):
    (model, raw_input_data, target_output, plc_raw_input, tnsr_delta, delta_init, tnsr_mask, mask_init, prob_output, lr, loss,loss1,loss2,loss3) = args
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(tnsr_delta.assign(delta_init))
        sess.run(tnsr_mask.assign(mask_init))
        model.set_weights(target_model_weights) 
        for e in range(200000):
            _, routput, rloss,rloss1,rloss2,rloss3, rtrojan_image, rdelta, rmask = \
                sess.run([train_op, prob_output, loss,loss1,loss2,loss3, trojan_image, tnsr_delta, tnsr_mask], \
                    feed_dict={plc_raw_input: raw_input_data})
            sess_hook(e, (routput, rloss,rloss1,rloss2,rloss3, rtrojan_image, rdelta, rmask))
            

with tf.variable_scope("cn1", reuse=tf.AUTO_REUSE):
    plc_raw_input = tf.placeholder(tf.float32, shape=input_tensor_shape)
    print("plc_raw_input:", plc_raw_input.shape)
    tnsr_mask = tf.get_variable("mask", (h, w), dtype=tf.float32)
    tnsr_delta= tf.get_variable("delta",(h, w, rgb))
    trojan_image = inject_trojan_cifar(plc_raw_input, tnsr_mask, tnsr_delta)
    tnsr_convert_trojan_image = tf.image.rgb_to_grayscale(trojan_image)

    print("trojan_image:", tnsr_convert_trojan_image.shape)
    prob_output = target_model(tnsr_convert_trojan_image)
    print("prob_output:", prob_output.shape)
    #loss = tf.reduce_sum(keras.losses.categorical_crossentropy(y_true=target, y_pred=target_output))
    loss1 = tf.reduce_sum(keras.losses.categorical_crossentropy(y_true=target_output, y_pred=prob_output))
    loss2 = tf.reduce_mean(tf.tanh(tnsr_mask)/2.0 + 0.5) * 350
    loss3 = tf.reduce_mean(tf.image.total_variation(tnsr_delta)) * 0.1
    loss = loss1 + loss2 + loss3
    train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[tnsr_delta, tnsr_mask])
    grads = tf.gradients(loss, tnsr_delta)

    args = (target_model, raw_images_rgb, target_output, plc_raw_input, tnsr_delta, delta_init, tnsr_mask, mask_init, prob_output, lr, loss,loss1,loss2,loss3)
    solve_trigger(args)