import keras, os, imageio, datetime, functools
import numpy as np
import tensorflow as tf
from keras import backend as K
#from IPython.display import Image, display
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

full_RE = 0

class TrojanDetector2():
    def __init__(self, target_model_path, metadata = {}, config = {}):
        # target_model: path to the file of target_model
        # metadata: information related to the model
        # config: configuration for the detection algorithm
        
        self.verbose = 0
        self.work_dir = os.getcwd()
        self.variable_scope_name = "trojan_detect"
        self.target_model_path = target_model_path
        self.target_model = None
        
        self.config = {
            'root_dir' : '/work/tmp/AISec/tluo/CNNTrojan/',
            'seed_gen' : os.path.join(self.work_dir, 'seed_gen'),
            'res_dir'  : os.path.join(self.work_dir, 'static','result'),
        }
        if config:
            for k, v in config.items():
                self.config[k] = v
                
        self.metadata = {
            'target_label' : 0,
            'Troj_size' : 64,
        }
        if metadata:
            for k, v in metadata.items():
                self.metadata[k] = v
        print([(k,v) for k,v in self.metadata.items()])
        
        if not self.verbose:
            print('[+] Config', self.config)
        
        #self.target_model.summary()
        
        self.placeholders, self.model_input, self.triggers, self.debug1 = None, None, None, None
        
        self.monitor_neurons = None
        self.num_of_classes = -1
                            
    # Detect Trojan
    def detect(self):
        # Basic Analysis of target model
        self.__preprossing()
        # Load/Gen Seed Samples
        self.setup_input_feeds()
        self.setup_output_feeds()
        # Prepare Trojan Input
        self.prepare_graph()
        self.solve_trigger()
        
    # Extrace Critical Information from the model
    def __preprossing(self):
        if not self.target_model:
            self.__load_model()
        #print(self.target_model.summary())
        self.model_inputs = self.target_model.inputs
        self.model_outputs = self.target_model.outputs
        self.input_shape = self.target_model.get_layer(index=0).input_shape
        self.output_shape = self.target_model.get_layer(index=-1).output_shape
        self.logtis_shape = self.target_model.get_layer(index=-2).output_shape
        
    def __load_model(self):
        self.target_model = keras.models.load_model(self.target_model_path)
        self.target_model_weights = self.target_model.get_weights()
        
    def setup_input_feeds(self):
        if 'setup_input_feeds_callback' in self.metadata \
          and self.metadata['setup_input_feeds_callback_args1']:
            self.input_feeds = self.metadata['setup_input_feeds_callback'] \
                (self.metadata['setup_input_feeds_callback_args1'])
            return True
        if not os.path.exists(self.config['seed_gen']):
            os.makedirs(self.config['seed_gen'])
        num_of_classes = self.target_model.get_layer(index=-2).output_shape[1]
        for i in range(num_of_classes):
            self.__reverse_engineer_inputs1(i)
        self.input_feeds = __load_input_seed()
        
    def setup_output_feeds(self):
        if 'setup_output_feeds_callback' in self.metadata:
            self.input_feeds = self.metadata['setup_output_feeds_callback']()
            return True
        res = []
        num_of_classes = self.target_model.get_layer(index=-2).output_shape[1]
        template = np.zeros(num_of_classes)
        template[self.metadata['target_label']] = 1
        for i in range(len(self.input_feeds)):
            res.append(template)
        self.output_feeds = np.array(res)
            
    # Reverse the input samples for specific output
    def __reverse_engineer_inputs1(self, target_label, number_of_samples=5, lr = 1e-2):
        # with target logits max, and single input and output tensor
        input_shape = self.input_shape
        output_shape = self.output_shape
        print("input_shape:", input_shape)
        
        input_sample = tf.get_variable("input_sample", [number_of_samples, 32, 32, 3])
        t_model = keras.models.clone_model(self.target_model)  
        t_model.pop()
        logits_output = t_model(input_sample)
        #loss = - tf.reduce_sum(logits_output[:,target_label])
        loss = tf.reduce_sum(logits_output) - 5 * tf.reduce_sum(logits_output[:,target_label])
        
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[input_sample])
        
        input_sample_init = np.random.rand(number_of_samples * functools.reduce(lambda x,y:x * y,input_shape[1:])).reshape([number_of_samples]+list(input_shape)[1:])
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(input_sample.assign(input_sample_init))
            t_model.set_weights(self.target_model_weights)
            K.set_learning_phase(0)
            for e in range(25000):
                #_, rgrad, rloss, rtrojan_image, routput = sess.run([train_op, grads, loss, input_sample, last_output]) #, feed_dict={target_output: output_feed}
                _, rloss, rtrojan_image, routput = sess.run([train_op, loss, input_sample, logits_output])
                if e % 20000 == 0:
                    print("e:",e, "loss:",rloss, "routput:",routput[:,target_label])
            for i in range(len(rtrojan_image)):
                #imageio.imsave("/home/tongbo.luo/tluo/AISec/TrojanDetection/CNN/Detection/tmp"+str(e)+"_"+str(i)+"_.png", rtrojan_image[i])
                save_img(os.path.join(self.config['seed_gen'], str(target_label)+"_"+str(i)+"_.png"), array_to_img(rtrojan_image[i]))
            for i in range(len(rtrojan_image)):
                #display(Image(filename = os.path.join(self.config['seed_gen'], str(target_label)+"_"+str(i)+"_.png"), width=200))
                print("Generated Seed Image", os.path.join(self.config['seed_gen'], str(target_label)+"_"+str(i)+"_.png"))
                
    def prepare_graph(self):
        print("[+]", datetime.datetime.now(), "Prepare to inject trojan to seed samples ...")
        self.placeholders, self.triggers, self.model_input, self.debug1, self.init_input_val = \
            self.metadata['customized_setup_input_tensors'](self.input_shape)
        
        print("[+]", datetime.datetime.now(), "Prepare to inspect model ...")
        self.models, self.output, self.inner_sample = \
            self.metadata['customized_setup_inspect_model'](self.target_model, self.model_input)
        
        print("[+]", datetime.datetime.now(), "Prepare to calculate loss ...")
        self.loss, self.target_output = \
            self.config["customized_setup_loss"](self.output, self.output_feeds, {"trigger":self.triggers, "target_label":0})
            
    def solve_trigger(self):
        lr = 2e-2
        image_placehodler = self.placeholders[0]
        trojan_image = self.model_input[0]
        mask, delta, last_output, logits, loss = self.triggers[0], self.triggers[1], self.output[0], self.output[1], self.loss
        
        self.train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[delta, mask])
        #self.train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=[delta])
        self.grads = tf.gradients(loss, delta)
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(delta.assign(self.init_input_val["delta_init"]))
            sess.run(mask.assign(self.init_input_val["mask_init"]))
                
            self.models[0].set_weights(self.target_model_weights)
            self.models[1].set_weights(self.target_model_weights)       
            K.set_learning_phase(0)
                
            for e in range(10002):
                if self.verbose:
                    _, routput, rloss, rtrojan_image, rdelta, rmask = \
                        sess.run([self.train_op, last_output, self.loss, trojan_image, delta, self.debug1["use_mask"]], \
                                    feed_dict={image_placehodler: self.input_feeds, self.target_output: target_output})
                else:
                    _, rloss, rtrojan_image, rdelta, routput = sess.run([self.train_op, loss, trojan_image, delta, last_output], feed_dict={image_placehodler: self.input_feeds})
                if e % 2000 == 0:
                    print(datetime.datetime.now(), " == Step:", e, "=="),
                    print("[+] rloss:", rloss, "Output Labels:", np.argmax(routput, axis=1), "#correct_labels:", np.sum(np.argmax(routput, axis=1) == 0))
                    #"Output Labels:", np.max(routput, axis=1), 
                if e % 500 == 0:
                        res_img_dir = os.path.join(self.config["root_dir"], self.config["res_dir"], 'imgs/')
                        self.config["train_callback"](res_img_dir, e, rtrojan_image)
                        # if 'customized_save_res' in self.metadata:
                        #    self.metadata['customized_save_res'](self, rtrojan_image, RE_img)
                        #else:
                        #    self.save_res(rtrojan_image, RE_img)
        
                        