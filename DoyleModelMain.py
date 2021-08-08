import numpy as np
import torchsde
from cardio_func import *

DoylePatient = DoyleSDE()

batch_size, state_size = 1, DoylePatient.state_size
simDuration = 400  # sec

fs = DoylePatient.paramsDict["displayParamsDict"]["fs"]

x_0 = torch.full((batch_size, state_size), 0.1)
tVec = torch.tensor(np.linspace(0, simDuration - 1/fs, fs*simDuration))

# Initial state x0, the SDE is solved over the interval [tVec[0], tVec[-1]].
# x_k will have shape (tVec.shape[0], batch_size, state_size)
with torch.no_grad():
    x_k = torchsde.sdeint(DoylePatient, x_0, tVec)