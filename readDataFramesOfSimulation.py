import pickle

dataset0_AseelsFields = pickle.load(open('dataFrame_AseelsFields_DoylePatientsDataset_noNoise_noControl.pt', "rb"))
dataset1_AseelsFields = pickle.load(open('dataFrame_AseelsFields_DoylePatientsDataset_noNoise.pt', "rb"))

# the next dataFrames should be plug and play into your code.
# I used the fields as follows:

# 'Time',
# 'HR_electrical' <= HR,
# 'HR_mechanical' <= HR,
# 'DBP' <= Pvs,
# 'MBP' <= Pas,
# 'SBP' <= Pap,
# 'RR' <= workload,
# 'ID',
# 'PP' <= Combination,
# 'CO' <= O2

# there are two types of datas, type0 and type1, one is with the controller active the other one is with the controller disabled.
DataFrame_type0_lowWorkload_AseelsFields = dataset0_AseelsFields[0]
DataFrame_type0_highWorkload_AseelsFields = dataset0_AseelsFields[1]

DataFrame_type1_lowWorkload_AseelsFields = dataset1_AseelsFields[0]
DataFrame_type1_highWorkload_AseelsFields = dataset1_AseelsFields[1]

# I also saved the same dataset using the field names from the simulation, so this is not plug and play
# it can be found in
# dataFrame_OriginalFields_DoylePatientsDataset_noNoise_noControl.pt
# dataFrame_OriginalFields_DoylePatientsDataset_noNoise.pt