OI = '../../../Datasets/Datasets_Features/Object-ImageNet'
OT = '../../../Datasets/Datasets_Features/Object-Text'
TT = '../../../Datasets/Datasets_Features/Tag-Text'
# addname
#----------------------------------------------------------------
# Object-ImageNet Datasets
SAD_8 = OI + '/Source_Amazon_Decaf_8.mat'
SAS_8 = OI + '/Source_Amazon_Surf_8.mat'
SCD_8 = OI + '/Source_Caltech_Decaf_8.mat'
SCS_8 = OI + '/Source_Caltech_Surf_8.mat'
SDD_8 = OI + '/Source_Dslr_Decaf_8.mat'
SDS_8 = OI + '/Source_Dslr_Surf_8.mat'
SWD_8 = OI + '/Source_Webcam_Decaf_8.mat'
SWS_8 = OI + '/Source_Webcam_Surf_8.mat'
#------------------------------
TID = OI + '/Target_Imgnet_Decaf.mat'
#----------------------------------------------------------------
# Object-Text Datasets
#------------------------------
SAD_6 = OT + '/Source_Amazon_Decaf_6.mat'
SAS_6 = OT + '/Source_Amazon_Surf_6.mat'
SCD_6 = OT + '/Source_Caltech_Decaf_6.mat'
SCS_6 = OT + '/Source_Caltech_Surf_6.mat'
SDD_6 = OT + '/Source_Dslr_Decaf_6.mat'
SDS_6 = OT + '/Source_Dslr_Surf_6.mat'
SWD_6 = OT + '/Source_Webcam_Decaf_6.mat'
SWS_6 = OT + '/Source_Webcam_Surf_6.mat'
TS5 = OT + '/Target_SP_5.mat'
#----------------------------------------------------------------
# Tag-Text
SNN_6 = TT + '/Source_Nustag_Neural_6.mat'
#----------------------------------------------------------------


OH = '/data/dataset/hda/Datasets/Office-Home'
OF = '/data/dataset/hda/Datasets/Office31'
TC = '/data/dataset/hda/Datasets/TextCategorization'
TI = '/data/dataset/hda/Datasets/T2IClassification'
KPG = '/data/dataset/hda/Datasets/KPG'
JMEA = '/data/dataset/hda/Datasets/JMEA'
#KPG = '/data/dataset/hda/Datasets/KPG_test'
RCV = '/data/dataset/hda/Datasets/RCV'
#----------------------------------------------------------------#
# Office-Home Datasets
SArD = OH + '/Source_Ar_DeCAF6.mat'
SClD = OH + '/Source_Cl_DeCAF6.mat'
SPrD = OH + '/Source_Pr_DeCAF6.mat'
SReD = OH + '/Source_Re_DeCAF6.mat'
#------------------------------#
TArR = OH + '/Target_Ar_ResNet50.mat'
TClR = OH + '/Target_Cl_ResNet50.mat'
TPrR = OH + '/Target_Pr_ResNet50.mat'
TReR = OH + '/Target_Re_ResNet50.mat'
#----------------------------------------------------------------#
# Office31 Datasets
SAD = OF + '/Source_A31_DeCAF6.mat'
SDD = OF + '/Source_D31_DeCAF6.mat'
SWD = OF + '/Source_W31_DeCAF6.mat'
#------------------------------#
TAR = OF + '/Target_A31_ResNet50.mat'
TDR = OF + '/Target_D31_ResNet50.mat'
TWR = OF + '/Target_W31_ResNet50.mat'
#----------------------------------------------------------------#
# TextCategorization43
E = TC + '/Source_EN.mat'
F = TC + '/Source_FR.mat'
G = TC + '/Source_GR.mat'
I = TC + '/Source_IT.mat'
S5 = TC + '/Target_SP_5.mat'
S10 = TC + '/Target_SP_10.mat'
S15 = TC + '/Target_SP_15.mat'
S20 = TC + '/Target_SP_20.mat'
#----------------------------------------------------------------#
# T2IClassification
SN = TI + '/Source_N.mat'
TI = TI + '/Target_I.mat'

#----------------------------------------------------------------#

JMEA_E2S5 = JMEA + '/Source_E2S5.mat'
JMEA_I2S5 = JMEA + '/Source_I2S5.mat'
JMEA_G2S5 = JMEA + '/Source_G2S5.mat'
JMEA_F2S5 = JMEA + '/Source_F2S5.mat'
JMEA_TE2S5 = JMEA + '/Target_E2S5.mat'
JMEA_TI2S5 = JMEA + '/Target_I2S5.mat'
JMEA_TG2S5 = JMEA + '/Target_G2S5.mat'
JMEA_TF2S5 = JMEA + '/Target_F2S5.mat'

JMEA_SN2TI = JMEA + '/Source_SN2TI.mat'
JMEA_TI = JMEA + '/Target_TI.mat'

JMEA_E2S10 = JMEA + '/Source_E2S10.mat'
JMEA_I2S10 = JMEA + '/Source_I2S10.mat'
JMEA_G2S10 = JMEA + '/Source_G2S10.mat'
JMEA_F2S10 = JMEA + '/Source_F2S10.mat'
JMEA_S10 = JMEA + '/Target_S10.mat'

#----------------------------------------------------------------#


#----------------------------------------------------------------#

KPG_E2S5 = KPG + '/Source_E2S5.mat'
KPG_I2S5 = KPG + '/Source_I2S5.mat'
KPG_G2S5 = KPG + '/Source_G2S5.mat'
KPG_F2S5 = KPG + '/Source_F2S5.mat'
KPG_S5 = KPG + '/Target_S5.mat'

KPG_SN2TI = KPG + '/Source_SN2TI.mat'
KPG_TI = KPG + '/Target_TI.mat'

KPG_E2S10 = KPG + '/Source_E2S10.mat'
KPG_I2S10 = KPG + '/Source_I2S10.mat'
KPG_G2S10 = KPG + '/Source_G2S10.mat'
KPG_F2S10 = KPG + '/Source_F2S10.mat'
KPG_S10 = KPG + '/Target_S10.mat'

#----------------------------------------------------------------#

SrcvE = TC + '/Source_EN.mat'
SrcvF = TC + '/Source_FR.mat'
SrcvG = TC + '/Source_GE.mat'
SrcvI = TC + '/Source_IT.mat'
SrcvS = TC + '/Source_SP.mat'

TrcvE = TC + '/Target_EN.mat'
TrcvF = TC + '/Target_FR.mat'
TrcvG = TC + '/Target_GE.mat'
TrcvI = TC + '/Target_IT.mat'
