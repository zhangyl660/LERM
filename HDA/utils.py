import tensorflow as tf
#=========================================================================#
def projection(X, W, F):
    H_1 = F(tf.add(tf.matmul(X, W['w1']), W['b1']))
    H_1 = tf.nn.l2_normalize(H_1, dim = 1)
    #---------------------#
    return H_1
#-----------------------------------------#
# define the classifier network
def classifier(X, W):
    H = tf.add(tf.matmul(X, W['w1']), W['b1'])
    return H
#=========================================================================#
def get_loss_Xa(Ya, F_Xa):

    loss_Xa = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Ya, logits=F_Xa))
    
    return loss_Xa
#--------------------------------------------------------#
def get_loss_reg(G_Xs, G_Xt, F, tau):
    
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(tau)(G_Xs['w1']))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(tau)(G_Xt['w1']))
    tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(tau)(F['w1']))
    
    loss_reg = tf.add_n(tf.get_collection("loss"))

    return loss_reg
#--------------------------------------------------------#
def get_mean_kl_loss(class_mean_Xt):
    nc = tf.shape(class_mean_Xt)[0]
    Yc = tf.eye(nc)
    epsilon = tf.constant(1e-6, tf.float32)
    class_mean_Xt = tf.nn.softmax(class_mean_Xt, axis=1)
    loss_mean_Xt = tf.reduce_mean(-tf.reduce_sum(Yc*tf.log(class_mean_Xt + epsilon),1))
    
    return loss_mean_Xt

def get_mean_l1_loss(class_mean_Xt):
    nc = tf.shape(class_mean_Xt)[0]
    Yc = tf.eye(nc)
    epsilon = tf.constant(1e-6, tf.float32)
    # class_mean_Xt = tf.nn.softmax(class_mean_Xt, axis=1)
    loss_mean = tf.reduce_mean(tf.abs(class_mean_Xt-Yc))
    # loss_mean_Xt = tf.reduce_mean(-tf.reduce_sum(Yc * tf.log(class_mean_Xt + epsilon), 1))
    return loss_mean

def get_mean_l2_loss(class_mean_Xt):
    nc = tf.shape(class_mean_Xt)[0]
    Yc = tf.eye(nc)
    epsilon = tf.constant(1e-6, tf.float32)
    # class_mean_Xt = tf.nn.softmax(class_mean_Xt, axis=1)
    loss_mean = tf.reduce_mean(tf.square(class_mean_Xt-Yc))
    # loss_mean_Xt = tf.reduce_mean(-tf.reduce_sum(Yc * tf.log(class_mean_Xt + epsilon), 1))
    return loss_mean


#--------------------------------------------------------#
def get_class_mean_Xt(Xl, Xu, Yl, Pred_Xu, nc):
    d = tf.shape(Xl)[1]
    #-------------------------------------------#
    Xl_label = tf.argmax(Yl,1)
    #-------------------------------------------#
    class_mean_Xt_list = []
    for c in range(nc):
        idx_Xl_c = tf.cast(tf.equal(Xl_label,c), tf.int32)
        #-------------------------------------------#
        Xl_c = tf.dynamic_partition(Xl,idx_Xl_c,2)[1]
        #---------------------------------#
        sum_Xl_c = tf.reduce_sum(Xl_c, 0)
        weight = tf.reshape(Pred_Xu[:,c], [-1,1])
        weight_Xu_c = tf.multiply(Xu, tf.tile(weight, [1,d]))
        sum_Xu_c = tf.reduce_sum(weight_Xu_c, 0)
        nl_c = tf.cast(tf.shape(Xl_c)[0], tf.float32)
        nu_c = tf.reduce_sum(weight)
        class_mean_Xt_list.append((sum_Xl_c+sum_Xu_c)/(nl_c+nu_c))
    #-----------------------------------------------------------#
    class_mean_Xt = tf.convert_to_tensor(class_mean_Xt_list)
    #-----------------------------------------------------------#
    return class_mean_Xt
#--------------------------------------------------------#
def get_class_mean_Xu(Xu, Pred_Xu, nc):
    d = tf.shape(Xu)[1]
    #-------------------------------------------#
    class_mean_Xu_list = []
    for c in range(nc):
        #---------------------------------#
        weight = tf.reshape(Pred_Xu[:,c], [-1,1])
        weight_Xu_c = tf.multiply(Xu, tf.tile(weight, [1,d]))
        sum_Xu_c = tf.reduce_sum(weight_Xu_c, 0)
        nu_c = tf.reduce_sum(weight)
        class_mean_Xu_list.append((sum_Xu_c)/(nu_c))
    #-----------------------------------------------------------#
    class_mean_Xu = tf.convert_to_tensor(class_mean_Xu_list)
    #-----------------------------------------------------------#
    return class_mean_Xu
#=========================================================================#

