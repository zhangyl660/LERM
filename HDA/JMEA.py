import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import multiprocessing
import scipy.io as sio  # read .mat files
import numpy as np
import argparse
from sklearn import preprocessing  # Normalization data
import add_dependencies as ad  # add some dependencies
from run_tcn import run_tcn
from run_KPG_tcn import run_KPG_tcn

# Test
# source_exp = [ad.E, ad.F, ad.G, ad.I]
# target_exp = [ad.S5, ad.S5, ad.S5, ad.S5]
source_exp = [ad.JMEA_E2S5, ad.JMEA_I2S5, ad.JMEA_G2S5, ad.JMEA_F2S5]
target_exp = [ad.JMEA_TE2S5, ad.JMEA_TI2S5, ad.JMEA_TG2S5, ad.JMEA_TF2S5]
results_name = 'JMEAT2T'

# source_exp = [ad.JMEA_E2S5]
# target_exp = [ad.JMEA_S5]
# results_name = 'JMEAT2T'

# source_exp = [ad.JMEA_SN2TI]
# target_exp = [ad.JMEA_TI]
# results_name = 'JMEAT2I'
#
# source_exp = [ad.JMEA_E2S10, ad.JMEA_I2S10, ad.JMEA_G2S10, ad.JMEA_F2S10]
# target_exp = [ad.JMEA_S10, ad.JMEA_S10, ad.JMEA_S10, ad.JMEA_S10]
# results_name = 'JMEAT2T_10'

# ===========================================================#
# --------------------------------------------------------#


parser = argparse.ArgumentParser(
    description='Simultaneous Semantic Alignment Network for Heterogeneous Domain Adaptation')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Trade-off parameter of transfer loss')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random_seed')
parser.add_argument('--method', type=str, default='CCL', help="Options: BNM, ENT, BFM, FBNM, CCL , NO")
parser.add_argument('--log_path', type=str, default='./log/', help="Options: BNM, ENT, BFM, FBNM, CCL , NO")
parser.add_argument('--iter', type=int, default=100, help='iter amount')
args = parser.parse_args()


if __name__ == "__main__":
    # parameters
    # T = 100 # the total iter number
    # alpha = 0.05 #control mean loss
    # tau = 0.01 # control regularization term, cannot be an integer
    # lr = 0.001 # learning rate
    # d = 600 # the dimension of common subspace
    T = args.iter  # the total iter number
    alpha = args.alpha  # control mean loss
    tau = 0.01  # control regularization term, cannot be an integer
    lr = args.lr  # learning rate
    d = 600  # the dimension of common subspace
    method = args.method
    seed = args.seed
    log_path = os.path.join(args.log_path, results_name + ".txt")
    # ===========================================================#
    length = len(source_exp)
    iter = 5  # for test
    acc_tcn_list = multiprocessing.Manager().list()
    acc_tcn = np.zeros((iter, length))

    for i in range(0, length):
        print("Source domain: " + source_exp[i])
        print("Target domain: " + target_exp[i])
        for j in range(0, iter):
            print("====================iteration[" + str(j + 1) + "]====================")
            # -------------------------------------#
            # load data
            source = sio.loadmat(source_exp[i])
            target = sio.loadmat(target_exp[i])

            Xl = target['training_features'][j]  # read labeled target data
            Xl = preprocessing.normalize(Xl, norm='l2')
            Xl_label = target['training_labels'][j]  # read labeled target data labels, form 0 start
            Xl_label = Xl_label.reshape((-1, 1))

            Xu = target['testing_features'][j]  # read unlabeled target data
            Xu = preprocessing.normalize(Xu, norm='l2')
            Xu_label = target['testing_labels'][j]  # read unlabeled target data labels, form 0 start
            Xu_label = Xu_label.reshape((-1, 1))

            Xs = source['source_features'][j]  # read source data
            Xs_label = source['source_labels'][j]  # read source data labels, form 0 start
            Xs = preprocessing.normalize(Xs, norm='l2')
            Xs_label = Xs_label.reshape((-1, 1))

            ns, ds = Xs.shape
            nl, dt = Xl.shape
            nu, _ = Xu.shape
            nc = len(np.unique(Xl_label))

            config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'nc': nc,
                      'alpha': alpha, 'tau': tau, 'd': d, 'lr': lr, 'method': method, 'seed': seed}
            config_data = {'Xs': Xs, 'Xl': Xl, 'Xu': Xu, 'Xs_label': Xs_label,
                           'Xl_label': Xl_label, 'Xu_label': Xu_label, 'T': T}

            p = multiprocessing.Process(target=run_tcn, args=(acc_tcn_list, config, config_data))
            p.start()
            p.join()
            acc_tcn[j][i] = acc_tcn_list[i * iter + j]
    print(np.mean(acc_tcn, axis=0))
    with open(log_path, 'a') as fp:
        fp.write('PN_HDA: '
                 + '| avg acc_best = ' + str(np.mean(acc_tcn, axis=0))
                 + '\n'
                 + str(args)
                 + '\n')
    np.savetxt('results/' + results_name + '.csv', acc_tcn, delimiter=',')
