import numpy as np
import tensorflow as tf
from keras import backend as K
import keras, datetime, os, functools
from PIL import Image as PIL_Image
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

class model_inspector():
    def __init__(self, config={}):
        """
        self.variable_scope_name = config["variable_scope_name"] if "variable_scope_name" in config else "trojan_detect"
        self.verbose = config["verbose"] if "verbose" in config else 0
        self.input_tensors = config["input_tensors"]
        self.target_model = config["target_model"]
        
        if isinstance(self.target_model, keras.engine.sequential.Sequential):
            self.model_inspect_helper = model_inspect_sequential()
        elif isinstance(self.target_model, keras.engine.training.Model):
            self.model_inspect_helper = model_inspect_functional()
        else:
            raise("Do Not Supported Model Type:", self.target_model)
        """
        
    def inspector1(self, target_model, input_tensor):
        last_output = target_model(input_tensor[0])
        t_model = keras.models.clone_model(target_model)
        #t_model = self.model_inspect_helper.pop(t_model)
        t_model.pop()
        logits = t_model(input_tensor)
        
        return [target_model, t_model], [last_output, logits], []
    
class image_trojan_setup_helper():
    
    def __init__(self):
        self.mean = np.array([125.307, 122.95, 113.865 ])
        self.std  = np.array([62.9932, 62.0887, 66.7048])
        self.l_bounds = np.asarray( [(0-self.mean[0])/self.std[0], (0-self.mean[1])/self.std[1], (0-self.mean[2])/self.std[2]])
        self.h_bounds = np.asarray( [(255-self.mean[0])/self.std[0], (255-self.mean[1])/self.std[1], (255-self.mean[2])/self.std[2]])
    
    def inject_trojan_cifar(self, input_tensor_shape, trigger=None):
        image_shape = list(input_tensor_shape)[1:]
        h, w, rgb = tuple(image_shape)
        if not trigger:
            mask = tf.get_variable("mask", (h, w), dtype=tf.float32)
            delta= tf.get_variable("delta",(h, w, rgb))
        else:
            mask, delta = trigger
            
        raw_input_tensor = tf.placeholder(tf.float32, shape=input_tensor_shape)
        con_mask = tf.tanh(mask)/2.0 + 0.5
        nc_mask = np.zeros((h, w), dtype=np.float32) + 1
        con_mask = con_mask * nc_mask
        use_mask = tf.tile(tf.reshape(con_mask, (1,h,w,1)), [1,1,1,rgb])
        trojan_image = raw_input_tensor * (1 - use_mask) + delta * use_mask
        trojan_image = tf.clip_by_value(trojan_image, self.l_bounds, self.h_bounds)
        
        delta_init = np.ones((h,w,3)).reshape(image_shape)
        #delta_init = np.random.rand(functools.reduce(lambda x,y:x * y,image_shape)).reshape(image_shape)
        mask_init = np.zeros((h,w))
        r=1.5
        for i in range(h):
            for j in range(w):
                if not( j >= w/r and j < w*(r-1)/r  and i >= h/r and i < h*(r-1)/r):
                    mask_init[i,j] = 0.8
        
        return [raw_input_tensor], [mask, delta], [trojan_image], [con_mask], {"delta_init":delta_init, "mask_init":mask_init}
        
    def inject_trojan1(self, input_tensor, trigger=None):
        trojan_image_roll = K.permute_dimensions(trojan_image, (0, 3, 1, 2))

    def img_preprocess(self, x_in):
        if len(x_in.shape) != 3 and len(x_in.shape) != 4:
            print('error shape', x_in.shape)
            sys.exit()
        x_in = x_in.astype('float32')
        if len(x_in.shape) == 3:
            for i in range(3):
                x_in[:,:,i] = (x_in[:,:,i] - self.mean[i]) / self.std[i]
        elif len(x_in.shape) == 4:
            for i in range(3):
                x_in[:,:,:,i] = (x_in[:,:,:,i] - self.mean[i]) / self.std[i]
        return x_in
                        
    def img_deprocess(self, x_in):
        if len(x_in.shape) != 3 and len(x_in.shape) != 4:
            print('error shape', x_in.shape)
            sys.exit()
        x_in = x_in.astype('float32')
        if len(x_in.shape) == 3:
            for i in range(3):
                x_in[:,:,i] = x_in[:,:,i] * self.std[i] + self.mean[i]
        elif len(x_in.shape) == 4:
            for i in range(3):
                x_in[:,:,:,i] = x_in[:,:,:,i] * self.std[i] + self.mean[i]
        return x_in
        
#with tf.variable_scope("cn1", reuse=tf.AUTO_REUSE):
#    raw_input_tensor = tf.placeholder(tf.float32, shape=[None,32,32,3])
#    h = image_trojan_setup_helper()
#    h.inject_trojan_cifar(raw_input_tensor)

h = image_trojan_setup_helper()

def setup_seed(directory):
    print("[+]", datetime.datetime.now(), "Prepare to setup seed samples ...")
    raw = []
    pictures = []
    for pic in sorted(os.listdir(directory)):
        if not pic.endswith("png"):
            continue
        img = PIL_Image.open(os.path.join(directory,pic)) # all convert to Height = 60, width=250, channel = 3
        raw.append(img)
        pictures.append(h.img_preprocess(np.array(img)))
        #display(Image(filename = os.path.join(directory,pic), width=200))
    #display_horizental([raw[i] for i in range(0,len(raw),5)], "/home/tongbo.luo/tluo/AISec/TrojanDetection/CNN/Detection/result/all.png")
    return pictures

#def setup_feed(directory = "/work/tmp/AISec/tluo/CNNTrojan/Dataset/cifar10/trojan/trojaned_dataset/seeds"):
def setup_feed(directory = "/home/tongbo.luo/tluo/AISec/TrojanDetection/CNN/Detection/seed_gen/"):
    start_images = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("."):
            continue
        start_image = os.path.join(directory, fname)
        start_images.append(start_image)
    images = []
    for start_image in start_images:
        image = imageio.imread(start_image)
        images.append(image)
    images = np.asarray(images).astype(np.float32)
    images = h.img_preprocess(images)
    return images

def train_callback(res_img_dir, e, rtrojan_image):
    if not os.path.exists(res_img_dir):
        os.makedirs(res_img_dir)
    RE_img = os.path.join(res_img_dir, 'adv_step%d_all.png' % e)
    imgs = [array_to_img(h.img_deprocess(rtrojan_image[i])) for i in range(0,len(rtrojan_image),5)]
    display_horizental(imgs, RE_img)
    
def display_horizental(imgs, RE_img):
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    imgs_comb = PIL_Image.fromarray(imgs_comb)
    imgs_comb.save(RE_img)    
    #display(imgs_comb)
    #display(Image(filename = RE_img, width=200*len(imgs)))
    