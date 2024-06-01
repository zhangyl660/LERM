import scipy.io
import tensorflow as tf

import pandas as pd
import numpy as np
import utils
from sklearn.manifold import TSNE
from tcn import tcn
import scipy.io as sio

def run_tcn(acc_tcn_list, config, config_data):
    tf.set_random_seed(config['seed'])
    with tf.Session() as sess:
        model = tcn(sess=sess, config=config)
        # ------------------------------------------#
        Xs = config_data['Xs']
        Xl = config_data['Xl']
        Xu = config_data['Xu']
        Xs_label = config_data['Xs_label']
        Xl_label = config_data['Xl_label']
        Xu_label = config_data['Xu_label']
        T = config_data['T']
        nl = config['nl']
        ns = config['ns']
        nu = config['nu']

        train_feed = {model.Xs: Xs, model.Xs_label: Xs_label,
                      model.Xl: Xl, model.Xl_label: Xl_label,
                      model.Xu: Xu, model.Xu_label: Xu_label}

        # loss_tcn = np.zeros((T,1))
        total_loss_list = []
        cls_loss_list = []
        ccl_loss_list = []
        label_loss_list = []
        acc_list = []

        for t in range(T):
            # ------------------------------------------#
            # training feature network
            sess.run(model.train_step, feed_dict=train_feed)
            total_loss, cls_loss, ccl_loss, label_loss = sess.run(
                [model.loss, model.loss_Xa, model.unlabel_ccl_loss, model.label_ccl_loss], feed_dict=train_feed)
            total_loss_list.append(total_loss)
            cls_loss_list.append(cls_loss)
            ccl_loss_list.append(ccl_loss)
            label_loss_list.append(label_loss)

            Acc_Xu = sess.run(model.Acc_Xu, feed_dict=train_feed)
            acc_list.append(Acc_Xu)
            if t % 50 == 0:
                # print("the total_loss is: " + str(total_loss))
                print("------------------iteration[" + str(t) + "]------------------")
                # ------------------------------------------#
                Acc_Xu = sess.run(model.Acc_Xu, feed_dict=train_feed)  # Compute final evaluation on test data
                print("the accuracy of f(Xu) is: " + str(Acc_Xu))
                print("===============================")

        Acc_Xu = sess.run(model.Acc_Xu, feed_dict=train_feed) * 100  # Get the final accuracy of Xu
        print("the accuracy of f(Xu) is: " + str(Acc_Xu))

        acc_tcn_list.append(Acc_Xu)  # record accuracy of Xu
