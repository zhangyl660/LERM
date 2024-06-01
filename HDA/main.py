import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import multiprocessing
import scipy.io as sio  # read .mat files
import numpy as np
import argparse
from sklearn import preprocessing  # Normalization data
import add_dependencies as ad  # add some dependencies
from run_tcn import run_tcn

# -----------------------------------------#
# read mat files
# -----------------------------------------#

# source_name = ["E", "F", "G", "I"]
# target_name = ["S5", "S5", "S5", "S5"]
# source_name = ["E", "F", "G", "I"]
# target_name = ["S10", "S10", "S10", "S10"]
source_name = ["SN"]
target_name = ["TI"]

# Test
# source_exp = [ad.E, ad.F, ad.G, ad.I]
# target_exp = [ad.S5, ad.S5, ad.S5, ad.S5]
# results_name = 'T2T5'

# source_exp = [ad.E, ad.F, ad.G, ad.I]
# target_exp = [ad.S10, ad.S10, ad.S10, ad.S10]
# results_name = 'T2T10'
#
source_exp = [ad.SN]
target_exp = [ad.TI]
results_name = 'N2I'

parser = argparse.ArgumentParser(
    description='Rethinking Guidance Information to Utilize Unlabeled Samples: A Label-Encoding Perspective')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Trade-off parameter of unlabeled loss')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random_seed')
parser.add_argument('--method', type=str, default='NO', help="Options: BNM, ENT, LERM_L1, LERM_KL, LERM_L2, NO")
parser.add_argument('--log_path', type=str, default='./log/', help="Options: BNM, ENT, LERM_L1, LERM_KL, LERM_L2, NO")
parser.add_argument('--iter', type=int, default=100, help='iter amount')
args = parser.parse_args()

# ===========================================================#
# --------------------------------------------------------#
if __name__ == "__main__":
    # parameters
    T = args.iter  # the total iter number
    alpha = args.alpha  # control mean loss
    tau = 0.01  # control regularization term, cannot be an integer
    lr = args.lr  # learning rate
    d = 600  # the dimension of common subspace
    method = args.method
    seed = args.seed
    log_path = os.path.join(args.log_path, results_name + "_std" + ".txt")
    # ===========================================================#
    length = len(source_exp)
    iter = 3
    acc_tcn_list = multiprocessing.Manager().list()
    entropy_tcn_list = multiprocessing.Manager().list()
    acc_tcn = np.zeros((iter, length))
    # entropy_tcn = np.zeros((iter, length))

    for i in range(0, length):
        print("Source domain: " + source_exp[i])
        print("Target domain: " + target_exp[i])
        for j in range(0, iter):
            # for j in range(0, 2):
            print("====================iteration[" + str(j + 1) + "]====================")
            # -------------------------------------#
            # load data
            source = sio.loadmat(source_exp[i])
            target = sio.loadmat(target_exp[i])

            Xl = target['training_features'][0, j]  # read labeled target data
            Xl = preprocessing.normalize(Xl, norm='l2')
            Xl_label = target['training_labels'][0, j] - 1  # read labeled target data labels, form 0 start

            Xu = target['testing_features'][0, j]  # read unlabeled target data
            Xu = preprocessing.normalize(Xu, norm='l2')
            Xu_label = target['testing_labels'][0, j] - 1  # read unlabeled target data labels, form 0 start

            Xs = source['source_features']  # read source data
            Xs_label = source['source_labels'] - 1  # read source data labels, form 0 start
            Xs = preprocessing.normalize(Xs, norm='l2')

            ns, ds = Xs.shape
            nl, dt = Xl.shape
            nu, _ = Xu.shape
            nc = len(np.unique(Xl_label))

            config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'nc': nc,
                      'alpha': alpha, 'tau': tau, 'd': d, 'lr': lr, 'method': method, 'Source': source_name[i], 'Target': target_name[i], 'seed': seed}
            config_data = {'Xs': Xs, 'Xl': Xl, 'Xu': Xu, 'Xs_label': Xs_label,
                           'Xl_label': Xl_label, 'Xu_label': Xu_label, 'T': T}

            p = multiprocessing.Process(target=run_tcn, args=(acc_tcn_list, config, config_data))
            p.start()
            p.join()
            acc_tcn[j][i] = acc_tcn_list[i * iter + j]
            # entropy_tcn[j][i] = entropy_tcn_list[i*iter+j]
    print(np.mean(acc_tcn, axis=0))
    print("acc mean:", np.mean(np.mean(acc_tcn, axis=0)))
    # print("entorpy mean:",np.mean(entropy_tcn,axis=0))
    with open(log_path, 'a') as fp:
        fp.write('PN_HDA: '
                 + '| avg acc_best = ' + str(np.mean(acc_tcn, axis=0))
                 + '\n'
                 + str(args)
                 + '\n')

    np.savetxt('results/' + results_name + '.csv', acc_tcn, delimiter=',')
