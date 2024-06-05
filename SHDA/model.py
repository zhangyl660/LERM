import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import tensorflow as tf
# remove warning
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import add_dependencies as ad # add some dependencies
import utils
import pdb

class model(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.ds = config['ds']
        self.dt = config['dt']
        self.ns = config['ns']
        self.nl = config['nl']
        self.nu = config['nu']
        self.nc = config['nc']
        self.alpha = config['alpha']
        self.tau = config['tau']
        self.d = config['d']
        self.lr = config['lr']
        self.method = config['method']
        self.create_model()

    def create_model(self):
        #================================================================#
        with tf.name_scope('inputs'):
            self.Xs = tf.placeholder(tf.float32, [None, self.ds])
            self.Xl = tf.placeholder(tf.float32, [None, self.dt])
            self.Xu = tf.placeholder(tf.float32, [None, self.dt])
            #-----------------------------------------#        
            self.Xs_label = tf.placeholder(tf.int32, [None, 1])
            self.Xl_label = tf.placeholder(tf.int32, [None, 1])
            self.Xu_label = tf.placeholder(tf.int32, [None, 1])
            #-----------------------------------------#        
            self.Yl = tf.reshape(tf.one_hot(self.Xl_label, self.nc,on_value=1,off_value=0), [self.nl, self.nc]) # shape: nl, nc
            self.Yu = tf.reshape(tf.one_hot(self.Xu_label, self.nc,on_value=1,off_value=0), [self.nu, self.nc]) # shape: nu, nc
            self.Ys = tf.reshape(tf.one_hot(self.Xs_label, self.nc,on_value=1,off_value=0), [self.ns, self.nc]) # shape: ns, nc
            #-----------------------------------------#
            self.Xt = tf.concat([self.Xl, self.Xu], 0)
            self.Ya = tf.concat([self.Ys, self.Yl], 0)
        #================================================================#
        # set the parameters of each layer
        self.G_Xs = {
            'w1': tf.Variable(tf.truncated_normal([self.ds, self.d], stddev=0.01)),
            #-----------------------------------------------------------------#
            'b1': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),
        }
        self.G_Xt = {
            'w1': tf.Variable(tf.truncated_normal([self.dt, self.d], stddev=0.01)),
            #-----------------------------------------------------------------#
            'b1': tf.Variable(tf.truncated_normal([self.d], stddev=0.01)),\
        }        
        self.F = {
            'w1': tf.Variable(tf.truncated_normal([self.d, self.nc], stddev=0.01)),
            'b1': tf.Variable(tf.truncated_normal([self.nc], stddev=0.01)),
        }
        #================================================================#
        # build projection network phi_s(Xs)
        self.Pro_Xs = utils.projection(self.Xs, self.G_Xs, tf.nn.leaky_relu)
        # build projection network phi_t(Xt)
        self.Pro_Xt = utils.projection(self.Xt, self.G_Xt, tf.nn.leaky_relu)
        self.Pro_Xl = tf.slice(self.Pro_Xt, [0, 0], [self.nl, -1])
        self.Pro_Xu = tf.slice(self.Pro_Xt, [self.nl, 0], [self.nu, -1]) 
        # connecting all data
        self.Pro_X = tf.concat([self.Pro_Xs, self.Pro_Xt], 0)
        #------------------------------------------#
        self.Pro_Xa = tf.slice(self.Pro_X, [0, 0], [self.ns+self.nl, -1])
        self.Pro_Xu = tf.slice(self.Pro_X, [self.ns+self.nl, 0], [self.nu, -1])
        #------------------------------------------#        
        self.Pro_Xs = tf.slice(self.Pro_Xa, [0, 0], [self.ns, -1])
        self.Pro_Xl = tf.slice(self.Pro_Xa, [self.ns, 0], [self.nl, -1])
        #================================================================#
        self.F_X = utils.classifier(self.Pro_X, self.F)
        self.F_Xa = tf.slice(self.F_X, [0, 0], [self.ns+self.nl, -1])
        self.F_Xu = tf.slice(self.F_X, [self.ns+self.nl, 0], [self.nu, -1])
        #------------------------------------------#        
        self.F_Xs = tf.slice(self.F_Xa, [0, 0], [self.ns, -1])
        self.F_Xl = tf.slice(self.F_Xa, [self.ns, 0], [self.nl, -1])
        #================================================================#        
        # the accuracy of xs
        self.Pred_Xs = tf.nn.softmax(self.F_Xs)
        self.Correct_Pred_Xs = tf.equal(tf.argmax(self.Ys,1), tf.argmax(self.Pred_Xs,1))
        self.Acc_Xs = tf.reduce_mean(tf.cast(self.Correct_Pred_Xs, tf.float32))
        # the accuracy of xl
        self.Pred_Xl = tf.nn.softmax(self.F_Xl)
        self.Correct_Pred_Xl = tf.equal(tf.argmax(self.Yl,1), tf.argmax(self.Pred_Xl,1))
        self.Acc_Xl = tf.reduce_mean(tf.cast(self.Correct_Pred_Xl, tf.float32))
        # the accuracy of xu
        self.Pred_Xu = tf.nn.softmax(self.F_Xu)
        self.Correct_Pred_Xu = tf.equal(tf.argmax(self.Yu,1), tf.argmax(self.Pred_Xu,1))
        self.Acc_Xu = tf.reduce_mean(tf.cast(self.Correct_Pred_Xu, tf.float32))
        #================================================================#
        self.loss_Xa = utils.get_loss_Xa(self.Ya, self.F_Xa)
        self.loss_reg = utils.get_loss_reg(self.G_Xs, self.G_Xt, self.F, self.tau)
        #self.loss_mean_Xu = utils.get_mean_loss(self.class_mean_Xu)

        # label_data
        self.Pred_Xa = tf.nn.softmax(self.F_Xa)
        self.Ya_float = tf.cast(self.Ya, dtype=tf.float32)
        self.class_mean_Xa = utils.get_class_mean_Xu(self.Pred_Xa, self.Ya_float, self.nc)
        # self.label_ccl_loss = utils.get_mean_loss(self.class_mean_Xa)

        # unlabel_data
        self.class_mean_Xu = utils.get_class_mean_Xu(self.Pred_Xu, self.Pred_Xu, self.nc)

        if self.method == "LERM_KL":
            # self.class_mean_Xu = utils.get_class_mean_Xu(self.Pred_Xu, self.Pred_Xu, self.nc)
            self.transfer_loss = utils.get_mean_kl_loss(self.class_mean_Xu)
        elif self.method == "LERM_L1":
            self.transfer_loss = utils.get_mean_l1_loss(self.class_mean_Xu)
        elif self.method == "LERM_L2":
            self.transfer_loss = utils.get_mean_l2_loss(self.class_mean_Xu)
        elif self.method == "BNM":
            s_tgt, _, _ = tf.linalg.svd(self.Pred_Xu)
            # _, s_tgt, _ = torch.svd(softmax_tgt)
            self.transfer_loss = -tf.reduce_mean(s_tgt)
        elif self.method == "ENT":
            epsilon = tf.constant(1e-6, tf.float32)
            self.transfer_loss = tf.reduce_mean(-tf.reduce_sum(self.Pred_Xu * tf.math.log(self.Pred_Xu + epsilon), 1))
        else:
            self.transfer_loss = tf.constant(0,dtype=tf.float32)

        #------------------------------------------#
        self.loss = self.loss_Xa + self.alpha*self.transfer_loss + self.loss_reg
        #================================================================#
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)      
        #------------------------------------------#        
        self.sess.run(tf.global_variables_initializer())

