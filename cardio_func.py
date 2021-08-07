import torch
from torch import nn

class DoyleSDE(torch.nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()
        self.paramsDict = self.DoyleParams()
        self.state_size = 4

    def DoyleParams(self):
        heartParamsDict = {
            # parameters for subject 1
            "c_l": 0.03,    #  [L / mmHg]
            "c_r": 0.05     #  [L / mmHg]
        }

        circulationParamsDict = {
            "c_as": 0.01016,    #  [L / mmHg]
            "c_vs": 0.65,       #  [L / mmHg]
            "c_ap": 0.0361,     #  [L / mmHg]
            "c_vp": 0.1408,     #  [L / mmHg]
            "R_p": 0.5,         #  [mmHg min/L]
            "R_s0": 6.5,        #  [mmHg min/L]
            "A": 18,            #  [mmHg min/L]
            "V_tot": 5.058,     #  [L]
            "V_TO2": 6.0,       #  [L]
            "O_2a": 0.2,        #  [L O2 / L blood]
            "rho": 0.012,       #  [L/min/Watt]
            "M_0": 0.36         #  [L/min]
        }

        controlParamsDict = {
            "q_as": 30,         # weighting factor, [1 / mmHg]
            "q_o2": 100000,     # weighting factor, [L blood / L O2]
            "q_H": 1            # weighting factor, [1 / min]
        }

        displayParamsDict = {
            "fs": 100  # [Hz] - figures time resolution
        }

        paramsDict = {
            "heartParamsDict": heartParamsDict,
            "circulationParamsDict": circulationParamsDict,
            "controlParamsDict": controlParamsDict,
            "displayParamsDict": displayParamsDict
        }

        return paramsDict

    def algebricEquations(self, x, d):
        # x is the state vector: [batch_size, state_size], [Pas, Pvs, Pap, O2v]
        # d is the workload input: [batch_size, 1], [W]

        Pas, Pvs, Pap, O2v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        W = d

        circulationParamsDict = self.paramsDict["circulationParamsDict"]
        c_as, c_vs, c_ap, c_vp = circulationParamsDict["c_as"], circulationParamsDict["c_vs"], circulationParamsDict["c_ap"], circulationParamsDict["c_vp"]
        V_tot, A, R_s0, rho, M_0, R_p, O_2a, V_TO2 = circulationParamsDict["V_tot"], circulationParamsDict["A"], circulationParamsDict["R_s0"], circulationParamsDict["rho"], circulationParamsDict["M_0"], circulationParamsDict["R_p"], circulationParamsDict["O_2a"], circulationParamsDict["V_TO2"]

        Vvp = V_tot - (c_as*Pas + c_vs*Pvs + c_ap*Pap)
        Pvp = torch.div(Vvp, c_vp)
        Rs = A*O2v + R_s0
        Fs, Fp = torch.div(Pas - Pvs, Rs), torch.div(Pap - Pvp, R_p)
        M = rho*W + M_0
        delta_O2 = O_2a - O2v

        return Pvp, Rs, Fs, Fp, M, delta_O2

    def calc_A_L_B_L(self, x, d, u):
        # x_L is the state vector at low exercise: [batch_size, state_size], [Pas, Pvs, Pap, O2v]
        # d is the workload input: [batch_size, 1], [W]

        Pas, Pvs, Pap, O2v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        W, H = d, u

        heartParamsDict, circulationParamsDict = self.paramsDict["heartParamsDict"], self.paramsDict["circulationParamsDict"]
        c_as, c_vs, c_ap, c_vp = circulationParamsDict["c_as"], circulationParamsDict["c_vs"], circulationParamsDict["c_ap"], circulationParamsDict["c_vp"]
        V_tot, A, R_s0, rho, M_0, R_p, O_2a, V_TO2 = circulationParamsDict["V_tot"], circulationParamsDict["A"], circulationParamsDict["R_s0"], circulationParamsDict["rho"], circulationParamsDict["M_0"], circulationParamsDict["R_p"], circulationParamsDict["O_2a"], circulationParamsDict["V_TO2"]
        c_l, c_r = heartParamsDict["c_l"], heartParamsDict["c_r"]

        Pvp, Rs, Fs, Fp, M, delta_O2 = self.algebricEquations(x, d)

        A_L, B_L = torch.zeros(self.state_size, self.state_size), torch.zeros(self.state_size, 1)

        A_L[0, 0] = - 1/(c_as*Rs)
        A_L[0, 1] = - A_L[0, 0]
        A_L[0, 3] = A*Fs/(c_as*Rs)

        A_L[1, 0] = 1/(c_vs*Rs)
        A_L[1, 1] = - (1/c_vs)*(c_r*H + 1/Rs)
        A_L[1, 3] = - A*Fs/(c_vs*Rs)

        A_L[2, 1] = c_r*H/c_ap
        A_L[2, 2] = -1/(c_ap*R_p)

        A_L[3, 0] = delta_O2/(Rs*V_TO2)
        A_L[3, 1] = -A_L[3, 0]
        A_L[3, 3] = -(Fs/V_TO2)*(A*delta_O2/Rs - 1)

        B_L[0, 0] = c_l*Pvp/c_as
        B_L[1, 0] = -c_r*Pvs/c_vs
        B_L[2, 0] = c_r*Pvs/c_ap

        return A_L, B_L

    # Drift
    def f(self, t, y):
        state_size = y.shape[1]
        for s in range(state_size):
            if s == 0:
                out = self.mu1(y[:, s:s+1])
            else:
                out = torch.cat((out, self.mu1(y[:, s:s+1])), dim=1)
        return out  # shape (batch_size, state_size)


    # Diffusion
    def g(self, t, y):
        batch_size = y.shape[0]
        return self.noiseStd[None, :].repeat(batch_size,1)


