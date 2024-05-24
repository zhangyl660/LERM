import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import multiprocessing
import scipy.io as sio # read .mat files
import numpy as np
import argparse
from sklearn import preprocessing # Normalization data
import add_dependencies as ad # add some dependencies
from run_tcn import run_tcn
#-----------------------------------------#
# read mat files
#-----------------------------------------#
# For Office31
# source_exp = [ad.SDD, ad.SWD, ad.SAD, ad.SWD, ad.SAD, ad.SDD]
# target_exp = [ad.TAR, ad.TAR, ad.TDR, ad.TDR, ad.TWR, ad.TWR]
# results_name = 'ccN_Office31'

# For Office-Home
#source_exp = [ad.SClD, ad.SPrD, ad.SReD, ad.SArD, ad.SPrD, ad.SReD, ad.SArD, ad.SClD, ad.SReD, ad.SArD, ad.SClD, ad.SPrD]
#target_exp = [ad.TArR, ad.TArR, ad.TArR, ad.TClR, ad.TClR, ad.TClR, ad.TPrR, ad.TPrR, ad.TPrR, ad.TReR, ad.TReR, ad.TReR]
# results_name = 'ccN_OH'

#For T2I
# source_exp = [ad.SN]
# target_exp = [ad.TI]
# results_name = 'ccN_T2I'

#For Text Classification
#source_exp = [ad.E, ad.F, ad.G, ad.I, ad.E, ad.F, ad.G, ad.I, ad.E, ad.F, ad.G, ad.I, ad.E, ad.F, ad.G, ad.I]
#target_exp = [ad.S5, ad.S5, ad.S5, ad.S5, ad.S10, ad.S10, ad.S10, ad.S10, ad.S15, ad.S15, ad.S15, ad.S15, ad.S20, ad.S20, ad.S20, ad.S20]
# results_name = 'ccN_TC'

# Total
# source_exp = [ad.SDD, ad.SWD, ad.SAD, ad.SWD, ad.SAD, ad.SDD, 
#               ad.SN,
#               ad.SClD, ad.SPrD, ad.SReD, ad.SArD, ad.SPrD, ad.SReD, ad.SArD, ad.SClD, ad.SReD, ad.SArD, ad.SClD, ad.SPrD,
#               ad.E, ad.F, ad.G, ad.I, ad.E, ad.F, ad.G, ad.I, ad.E, ad.F, ad.G, ad.I, ad.E, ad.F, ad.G, ad.I]
# target_exp = [ad.TAR, ad.TAR, ad.TDR, ad.TDR, ad.TWR, ad.TWR,  
#               ad.TI,
#               ad.TArR, ad.TArR, ad.TArR, ad.TClR, ad.TClR, ad.TClR, ad.TPrR, ad.TPrR, ad.TPrR, ad.TReR, ad.TReR, ad.TReR, 
#               ad.S5, ad.S5, ad.S5, ad.S5, ad.S10, ad.S10, ad.S10, ad.S10, ad.S15, ad.S15, ad.S15, ad.S15, ad.S20, ad.S20, ad.S20, ad.S20]
# results_name = 'ccN-t0-1'

#For sentiment classification across languages
# source_exp = [ad.SBEN, ad.SBEN, ad.SBEN, ad.SDEN, ad.SDEN, ad.SDEN, ad.SMEN, ad.SMEN, ad.SMEN]
# target_exp = [ad.TBFR, ad.TBGE, ad.TBJP, ad.TDFR, ad.TDGE, ad.TDJP, ad.TMFR, ad.TMGE, ad.TMJP]
#-------------------------#
#For sentiment classification across languages and products
#source_exp = [ad.SBEN, ad.SBEN, ad.SBEN, ad.SBEN, ad.SDEN, ad.SDEN, ad.SDEN, ad.SDEN, ad.SMEN, ad.SMEN, ad.SMEN, ad.SMEN]
#target_exp = [ad.TDFR, ad.TMGE, ad.TDGE, ad.TMJP, ad.TBFR, ad.TMGE, ad.TBGE, ad.TMJP, ad.TBFR, ad.TDGE, ad.TBGE, ad.TDJP]
#-------------------------#
#For text classification across languages
# source_exp = [ad.SEN, ad.SEN, ad.SEN, ad.SEN, ad.SFR, ad.SFR, ad.SFR, ad.SFR, ad.SGE, ad.SGE, ad.SGE, ad.SGE, ad.SIT, ad.SIT, ad.SIT, ad.SIT, ad.SSP, ad.SSP, ad.SSP, ad.SSP]
# target_exp = [ad.TFR, ad.TGE, ad.TIT, ad.TSP, ad.TEN, ad.TGE, ad.TIT, ad.TSP, ad.TEN, ad.TFR, ad.TIT, ad.TSP, ad.TEN, ad.TFR, ad.TGE, ad.TSP, ad.TEN, ad.TFR, ad.TGE, ad.TIT]
# results_name = 'results-text'

# source_exp = [ad.SBEN, ad.SBEN, ad.SBEN, ad.SDEN, ad.SDEN, ad.SDEN, ad.SMEN, ad.SMEN, ad.SMEN,
#               ad.SBEN, ad.SBEN, ad.SBEN, ad.SBEN, ad.SDEN, ad.SDEN, ad.SDEN, ad.SDEN, ad.SMEN, ad.SMEN, ad.SMEN, ad.SMEN,
#               ad.SEN, ad.SEN, ad.SEN, ad.SEN, ad.SFR, ad.SFR, ad.SFR, ad.SFR, ad.SGE, ad.SGE, ad.SGE, ad.SGE, ad.SIT, ad.SIT, ad.SIT, ad.SIT, ad.SSP, ad.SSP, ad.SSP, ad.SSP]
# target_exp = [ad.TBFR, ad.TBGE, ad.TBJP, ad.TDFR, ad.TDGE, ad.TDJP, ad.TMFR, ad.TMGE, ad.TMJP,
#               ad.TDFR, ad.TMGE, ad.TDGE, ad.TMJP, ad.TBFR, ad.TMGE, ad.TBGE, ad.TMJP, ad.TBFR, ad.TDGE, ad.TBGE, ad.TDJP,
#               ad.TFR, ad.TGE, ad.TIT, ad.TSP, ad.TEN, ad.TGE, ad.TIT, ad.TSP, ad.TEN, ad.TFR, ad.TIT, ad.TSP, ad.TEN, ad.TFR, ad.TGE, ad.TSP, ad.TEN, ad.TFR, ad.TGE, ad.TIT]
# results_name = 'results-total'


source_name = ["E", "F", "G", "I"]
target_name = ["S5", "S5", "S5", "S5"]
# source_name = ["SN"]
# target_name = ["TI"]


# Test
source_exp = [ad.E, ad.F, ad.G, ad.I]
target_exp = [ad.S5, ad.S5, ad.S5, ad.S5]
results_name = 'T2T5'

# source_exp = [ad.SN]
# target_exp = [ad.TI]
# results_name = 'T2I'

# source_exp = [ad.E, ad.F, ad.G, ad.I]
# target_exp = [ad.S10, ad.S10, ad.S10, ad.S10]
# results_name = 'T2T10'
#

parser = argparse.ArgumentParser(
    description='Simultaneous Semantic Alignment Network for Heterogeneous Domain Adaptation')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Trade-off parameter of transfer loss')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random_seed')
parser.add_argument('--method', type=str, default='NO', help="Options: BNM, ENT, BFM, FBNM, CCL , NO")
parser.add_argument('--log_path', type=str, default='./log/', help="Options: BNM, ENT, BFM, FBNM, CCL , NO")
parser.add_argument('--iter', type=int, default=100, help='iter amount')
args = parser.parse_args()


#===========================================================#
#--------------------------------------------------------#
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
    log_path = os.path.join(args.log_path, results_name + "_std" + ".txt")
#===========================================================#
    length = len(source_exp)
    iter = 5 # for test
    acc_tcn_list = multiprocessing.Manager().list()
    acc_tcn = np.zeros((iter, length))

    for i in range(0, length):
        print("Source domain: " + source_exp[i])
        print("Target domain: " + target_exp[i])
        for j in range(0, iter):
        #for j in range(0, 2):
            print("====================iteration[" + str(j+1) + "]====================")
            #-------------------------------------#
            # load data
            source = sio.loadmat(source_exp[i])
            target = sio.loadmat(target_exp[i])

            Xl = target['training_features'][0,j] # read labeled target data
            Xl = preprocessing.normalize(Xl, norm='l2')
            Xl_label = target['training_labels'][0,j] - 1 # read labeled target data labels, form 0 start

            Xu = target['testing_features'][0,j]  # read unlabeled target data
            Xu = preprocessing.normalize(Xu, norm='l2')
            Xu_label = target['testing_labels'][0,j] - 1  # read unlabeled target data labels, form 0 start

            Xs = source['source_features'] # read source data
            Xs_label = source['source_labels'] - 1 # read source data labels, form 0 start
            Xs = preprocessing.normalize(Xs, norm='l2')

            ns, ds = Xs.shape
            nl, dt = Xl.shape
            nu, _ = Xu.shape
            nc = len(np.unique(Xl_label))
                
            config = {'ds': ds, 'dt': dt, 'ns': ns, 'nl': nl, 'nu': nu, 'nc': nc, 
                      'alpha': alpha, 'tau': tau, 'd': d, 'lr': lr , 'method': method, 'Source': source_name[i], 'Target':target_name[i], 'seed': seed}
            config_data = {'Xs': Xs, 'Xl': Xl, 'Xu': Xu, 'Xs_label': Xs_label, 
                           'Xl_label': Xl_label, 'Xu_label': Xu_label, 'T': T}
            
            p = multiprocessing.Process(target=run_tcn, args=(acc_tcn_list,config,config_data))
            p.start()
            p.join()
            acc_tcn[j][i] = acc_tcn_list[i*iter+j]
    print(np.mean(acc_tcn, axis=0))
    with open(log_path, 'a') as fp:
        fp.write('PN_HDA: '
                 + '| avg acc_best = ' + str(np.mean(acc_tcn, axis=0))
                 + '\n'
                 + str(args)
                 + '\n')

    np.savetxt('results/'+results_name+'.csv', acc_tcn, delimiter = ',')
