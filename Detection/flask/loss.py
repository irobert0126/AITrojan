import numpy as np
import tensorflow as tf
import keras
    
def setup_loss1(monitored, target_output, meta={}):
    last_output, logit = monitored[0], monitored[1]
    #loss = (tf.reduce_sum(last_output) - 15 * tf.reduce_sum(last_output[:, meta["target_label"]]))/100
    loss1 = tf.losses.softmax_cross_entropy(onehot_labels=target_output, logits=last_output)
    
    if "trigger" in meta:
        triggers = meta["trigger"]
        mask, delta = triggers[0], triggers[1]
        #loss = 0.0001 * tf.reduce_mean(tf.image.total_variation(delta))
        con_mask = tf.tanh(mask)/2.0 + 0.5
        loss2 = tf.reduce_mean(con_mask)
        loss = loss1 + loss2 * 0.02
    return loss, []
    
def setup_loss2_conmakeweight1(output, inner_nn, triggers, variable_scope_name, output_shape, config = {"con_mask_weight":1}):
    with tf.variable_scope(variable_scope_name, reuse=tf.AUTO_REUSE):
        last_output, logit = output
        mask, delta, con_mask = triggers["mask"], triggers["delta"], triggers["con_mask"]
        num_of_classes = logit.shape[1]
        
        target_output = tf.placeholder(tf.float32, shape = output_shape)
        loss = tf.reduce_sum(keras.losses.categorical_crossentropy(y_true=target_output, y_pred=last_output))

        loss += 0.00001 * tf.reduce_mean(tf.image.total_variation(delta))
        loss += config["con_mask_weight"] * tf.reduce_mean(con_mask)
        return loss, target_output
    
def setup_loss2_conmakeweight100(output, inner_nn, triggers, variable_scope_name, config = {"con_mask_weight":100}):
    with tf.variable_scope(variable_scope_name, reuse=tf.AUTO_REUSE):
        last_output, logit = output
        mask, delta, con_mask = triggers["mask"], triggers["delta"], triggers["con_mask"]
        num_of_classes = logit.shape[1]
        
        target_output = tf.placeholder(tf.float32, shape=(None, num_of_classes))
        loss = tf.reduce_sum(keras.losses.categorical_crossentropy(y_true=target_output, y_pred=last_output))

        loss += 0.00001 * tf.reduce_mean(tf.image.total_variation(delta))
        loss += config["con_mask_weight"] * tf.reduce_mean(con_mask)
        return loss, target_output