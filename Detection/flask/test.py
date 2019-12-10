from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 
from scipy import misc  
import numpy
import numpy.random
import scipy.ndimage
import scipy.misc

def random_crop_image(image):
      height, width = image.shape[:2]
      random_array = numpy.random.random(size=4);
      w = int((width*0.5)*(1+random_array[0]*0.5))
      h = int((height*0.5)*(1+random_array[1]*0.5))
      x = int(random_array[2]*(width-w))
      y = int(random_array[3]*(height-h))

      image_crop = image[y:h+y,x:w+x,0:3]
      image_crop = misc.imresize(image_crop,image.shape)
      return image_crop

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        preprocessing_function=random_crop_image,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect')

img = load_img('lena.jpg') 
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',save_prefix='/home/yan/Desktop/test/test/', save_format='jpg'):
    i += 1
    if i > 20:
        break  # 否则生成器会退出循环
  