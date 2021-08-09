import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from scipy.linalg import solve_continuous_are

class DoyleSDE(torch.nn.Module):

    def __init__(self, d, d_tVec):
        super().__init__()
        self.d = d  # [Watt] d is the workload input of shape # [time-vec, batchSize, 1]; the time-vec should be dense
        self.d_tVec = d_tVec

        self.noise_type = "diagonal"
        self.sde_type = "ito"

        self.paramsDict = self.DoyleParams()
        self.state_size = 4
        self.control_size = 1
        self.input_size = self.d.shape[2]

        self.enableController = True

        self.solveIvpRun = False
        self.solveIvpRunBatchNo = 0

        self.dotFactor = 1/60  # all units are in minutes

        self.noiseStd = torch.tensor([0, 0, 0, 0])  # for state_size=4

        Pas_L, Pvs_L, Pap_L, O2v_L, W_L, H_L = 77.5, 4.250, 11.6, 154/1000, 0, 41
        x_L, W_L, H_L = torch.tensor([Pas_L, Pvs_L, Pap_L, O2v_L], dtype=torch.float)[:, None], torch.tensor([W_L], dtype=torch.float)[:, None], torch.tensor([H_L], dtype=torch.float)[:, None]

        # x_L is [state_size, 1], W_L is [input_size, 1], H_L is [control_size, 1]
        self.referenceValues = {"x_L": x_L, "d_L": W_L, "u_L": H_L}
        self.K = self.calc_gain_K(self.referenceValues)

        self.controlBias = torch.matmul(self.K, x_L) + H_L

    def DoyleParams(self):
        heartParamsDict = {
            # parameters for subject 1 cl,cr = 0.03, 0.05
            # parameters for subject 2 cl,cr = 0.025, 0.045
            "c_l": 0.025,    #  [L / mmHg]
            "c_r": 0.045     #  [L / mmHg]
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
            # parameters for subject 1,2 @ 0-50 Watt:
            "q_as": 30,         # weighting factor, [1 / mmHg]
            "q_o2": 100000,     # weighting factor, [L blood / L O2]
            "q_H": 1            # weighting factor, [1 / min]
        }

        displayParamsDict = {
            "fs": 1.0  # [Hz] - figures time resolution
        }

        paramsDict = {
            "heartParamsDict": heartParamsDict,
            "circulationParamsDict": circulationParamsDict,
            "controlParamsDict": controlParamsDict,
            "displayParamsDict": displayParamsDict
        }

        return paramsDict

    def algebricEquations(self, x, u, d):
        # x is the state vector: [batch_size, state_size, 1], [Pas, Pvs, Pap, O2v]
        # u is the control: [batch_size, control_size, 1]
        # d is the workload input: [batch_size, input_size, 1], [W]

        Pas, Pvs, Pap, O2v = x[:, 0:1, 0:1], x[:, 1:2, 0:1], x[:, 2:3, 0:1], x[:, 3:4, 0:1]
        W = d
        H = u

        if self.solveIvpRun:
            W = W[self.solveIvpRunBatchNo:self.solveIvpRunBatchNo+1]

        # Pas, Pvs, Pap, O2v, W are of shape [batch_size, 1, 1]

        heartParamsDict, circulationParamsDict = self.paramsDict["heartParamsDict"], self.paramsDict["circulationParamsDict"]
        c_as, c_vs, c_ap, c_vp = circulationParamsDict["c_as"], circulationParamsDict["c_vs"], circulationParamsDict["c_ap"], circulationParamsDict["c_vp"]
        V_tot, A, R_s0, rho, M_0, R_p, O_2a, V_TO2 = circulationParamsDict["V_tot"], circulationParamsDict["A"], circulationParamsDict["R_s0"], circulationParamsDict["rho"], circulationParamsDict["M_0"], circulationParamsDict["R_p"], circulationParamsDict["O_2a"], circulationParamsDict["V_TO2"]
        c_l, c_r = heartParamsDict["c_l"], heartParamsDict["c_r"]

        Vvp = V_tot - (c_as*Pas + c_vs*Pvs + c_ap*Pap)
        Pvp = torch.div(Vvp, c_vp)
        Rs = A*O2v + R_s0
        Fs, Fp = torch.div(Pas - Pvs, Rs), torch.div(Pap - Pvp, R_p)
        M = rho*W + M_0
        delta_O2 = O_2a - O2v
        Q_l = c_l*H*Pvp
        Q_r = c_r*H*Pvs

        return Pvp, Rs, Fs, Fp, M, delta_O2, Q_l, Q_r

    def calc_gain_K(self, referenceValues):
        A_L, B_L = self.calc_A_L_B_L(referenceValues)

        controlParamsDict = self.paramsDict["controlParamsDict"]
        Q, R = torch.zeros(self.state_size, self.state_size), torch.zeros(self.control_size, 1)

        Q[0, 0] = torch.pow(torch.tensor(controlParamsDict["q_as"]), 2)
        Q[3, 3] = torch.pow(torch.tensor(controlParamsDict["q_o2"]), 2)

        R[0, 0] = torch.pow(torch.tensor(controlParamsDict["q_H"]), 2)

        P = torch.tensor(solve_continuous_are(a=A_L, b=B_L, q=Q, r=R), dtype=torch.float)
        K = torch.matmul(torch.linalg.inv(R), torch.matmul(torch.transpose(B_L, 1, 0), P))
        return K


    def calc_A_L_B_L(self, referenceValues):
        x_L, d_L, u_L = referenceValues["x_L"], referenceValues["d_L"], referenceValues["u_L"]
        # x_L is the reference state vector at low exercise: [state_size, 1], [Pas, Pvs, Pap, O2v]
        # d_L is the reference workload input: [input_size, 1], [W]
        # u_L is the reference heart rate at low exercise [control_size, 1], [H]

        Pas_L, Pvs_L, Pap_L, O2v_L = x_L[0], x_L[1], x_L[2], x_L[3]
        W_L, H_L = d_L, u_L

        heartParamsDict, circulationParamsDict = self.paramsDict["heartParamsDict"], self.paramsDict["circulationParamsDict"]
        c_as, c_vs, c_ap, c_vp = circulationParamsDict["c_as"], circulationParamsDict["c_vs"], circulationParamsDict["c_ap"], circulationParamsDict["c_vp"]
        V_tot, A, R_s0, rho, M_0, R_p, O_2a, V_TO2 = circulationParamsDict["V_tot"], circulationParamsDict["A"], circulationParamsDict["R_s0"], circulationParamsDict["rho"], circulationParamsDict["M_0"], circulationParamsDict["R_p"], circulationParamsDict["O_2a"], circulationParamsDict["V_TO2"]
        c_l, c_r = heartParamsDict["c_l"], heartParamsDict["c_r"]

        Pvp_L, Rs_L, Fs_L, Fp_L, M_L, delta_O2_L, Q_l, Q_r = self.algebricEquations(x_L[None, :, :], u_L[None, :, :], d_L[None, :, :])
        Pvp_L, Rs_L, Fs_L, Fp_L, M_L, delta_O2_L = Pvp_L.squeeze(), Rs_L.squeeze(), Fs_L.squeeze(), Fp_L.squeeze(), M_L.squeeze(), delta_O2_L.squeeze()

        A_L, B_L = torch.zeros(self.state_size, self.state_size), torch.zeros(self.state_size, 1)

        A_L[0, 0] = - 1/(c_as*Rs_L)
        A_L[0, 1] = - A_L[0, 0]
        A_L[0, 3] = A*Fs_L/(c_as*Rs_L)

        A_L[1, 0] = 1/(c_vs*Rs_L)
        A_L[1, 1] = - (1/c_vs)*(c_r*H_L + 1/Rs_L)
        A_L[1, 3] = - A*Fs_L/(c_vs*Rs_L)

        A_L[2, 1] = c_r*H_L/c_ap
        A_L[2, 2] = -1/(c_ap*R_p)

        A_L[3, 0] = delta_O2_L/(Rs_L*V_TO2)
        A_L[3, 1] = -A_L[3, 0]
        A_L[3, 3] = -(Fs_L/V_TO2)*(A*delta_O2_L/Rs_L - 1)

        B_L[0, 0] = c_l*Pvp_L/c_as
        B_L[1, 0] = -c_r*Pvs_L/c_vs
        B_L[2, 0] = c_r*Pvs_L/c_ap

        return A_L, B_L

    def calcControl(self, x):
        # x is the state vector: [batch_size, state_size, 1], [Pas, Pvs, Pap, O2v]
        batch_size = x.shape[0]
        x_L, u_L = self.referenceValues["x_L"], self.referenceValues["u_L"]
        # x_L is the reference state vector at low exercise: [state_size, 1], [Pas, Pvs, Pap, O2v]
        # d_L is the reference workload input: [input_size, 1], [W]
        # u_L is the reference heart rate at low exercise [control_size, 1], [H]

        if self.enableController:
            K = self.K
            x_L, H_L = x_L[None, :, :].repeat(batch_size, 1, 1), u_L[None, :, :].repeat(batch_size, 1, 1)

            u = - torch.matmul(K, x) + self.controlBias[None, :, :].repeat(batch_size, 1, 1)
            # u is the control [batch_size, control_size, 1]
        else:
            u = u_L[None, :, :].repeat(batch_size, 1, 1)
        return u

    def runSolveIvp(self, x_0, simDuration):
        self.solveIvpRun = True
        batchSize = x_0.shape[0]
        solList = list()
        for b in range(batchSize):
            self.solveIvpRunBatchNo = b
            solList.append(solve_ivp(self.calc_dx, [0, simDuration], x_0[b].tolist(), args=[self.paramsDict], dense_output=True, rtol=1e-4))

        fs = self.paramsDict["displayParamsDict"]["fs"]
        tVec = np.linspace(0, simDuration - 1 / fs, int(fs * simDuration))

        x_k = np.zeros((tVec.shape[0], batchSize, self.state_size))
        for b, sol in enumerate(solList):
            x_k[:, b] = np.transpose(sol.sol(tVec))

        self.solveIvpRun = False
        return torch.tensor(x_k, dtype=torch.float)

    def calc_dx(self, t, x, *args):
        dx = self.f(torch.tensor(t), torch.tensor(x, dtype=torch.float)[None, :])
        return dx.numpy()[0]

    # Drift
    def f(self, t, x):
        # x is the state vector: [batch_size, state_size], [Pas, Pvs, Pap, O2v]
        # d is the workload input: [batch_size, 1], [W]

        batch_size = x.shape[0]

        dIndex = torch.argmin(torch.abs(self.d_tVec - t))
        W = self.d[dIndex]

        Pas, Pvs, Pap, O2v = x[:, 0:1, None], x[:, 1:2, None], x[:, 2:3, None], x[:, 3:4, None]

        circulationParamsDict, heartParamsDict = self.paramsDict["circulationParamsDict"], self.paramsDict["heartParamsDict"]
        c_as, c_vs, c_ap, c_vp = circulationParamsDict["c_as"], circulationParamsDict["c_vs"], circulationParamsDict["c_ap"], circulationParamsDict["c_vp"]
        V_tot, A, R_s0, rho, M_0, R_p, O_2a, V_TO2 = circulationParamsDict["V_tot"], circulationParamsDict["A"], circulationParamsDict["R_s0"], circulationParamsDict["rho"], circulationParamsDict["M_0"], circulationParamsDict["R_p"], circulationParamsDict["O_2a"], circulationParamsDict["V_TO2"]
        c_l, c_r = heartParamsDict["c_l"], heartParamsDict["c_r"]

        H = self.calcControl(x[:, :, None])
        Pvp, Rs, Fs, Fp, M, delta_O2, Q_l, Q_r = self.algebricEquations(x[:, :, None], u=H, d=W[:, :, None])


        dot_Pas = self.dotFactor*(1/c_as)*(Q_l - Fs)
        dot_Pvs = self.dotFactor*(1/c_vs)*(Fs - Q_r)
        dot_Pap = self.dotFactor*(1/c_ap)*(Q_r - Fp)
        dot_O2v = self.dotFactor*(1/V_TO2)*(-M + torch.matmul(Fs, delta_O2))

        return torch.cat((dot_Pas, dot_Pvs, dot_Pap, dot_O2v), dim=1)[:, :, 0]  # shape (batch_size, state_size)


    # Diffusion
    def g(self, t, x):
        batch_size = x.shape[0]
        return self.noiseStd[None, :].repeat(batch_size, 1)

    def plot(self, x_k):
        # x_k shape (nSamples, batch_size, state_size)
        nSamples, batch_size = x_k.shape[0], x_k.shape[1]
        # recalc the control:
        u_k = torch.zeros(nSamples, batch_size, self.control_size, dtype=torch.float)
        for k in range(nSamples):
            u_k[k] = self.calcControl(x_k[k][:, :, None])[:, :self.control_size, 0]

        x_k[:, :, 3] = 1000*x_k[:, :, 3] # to converts the units from [L O2 / L blood] to [mL O2 / L blood]

        x_u_k = torch.cat((x_k, u_k), dim=2)
        x_u_k = x_u_k.cpu().numpy()

        tVec = np.arange(0, nSamples)/self.paramsDict["displayParamsDict"]["fs"]

        ylabels = ["Pas", "Pvs", "Pap", "O2v", "HR", "W"]
        colors = ['magenta', 'yellow', 'yellow', 'green', 'black', 'blue']
        plt.figure(figsize=(16, 12))
        for s in range(self.state_size + self.control_size + self.input_size):
            plt.subplot(2, 3, s+1)
            for b in range(batch_size):
                if s == (self.state_size + self.control_size + self.input_size - 1):
                    plt.plot(self.d_tVec, self.d[:, b, 0], label=f'sample {s}')
                else:
                    plt.plot(tVec, x_u_k[:, b, s], label=f'sample {s}')

            plt.xlabel('sec')
            plt.ylabel(ylabels[s])
            plt.legend()
            plt.grid()

        sList = [0, 3, 4, 5]
        for b in range(batch_size):
            plt.figure()
            for s in sList:
                if s == 5:
                    plt.plot(self.d_tVec, self.d[:, b, 0], label=ylabels[s], color=colors[s])
                else:
                    plt.plot(tVec, x_u_k[:, b, s], label=ylabels[s], color=colors[s])
            plt.xlabel('sec')
            plt.legend()
            plt.grid()





def plot(ts, samples, xlabel, ylabel, title=''):
    ts = ts.cpu()
    batch_size = samples.shape[1]
    samples = samples.squeeze().t().cpu()
    plt.figure()
    if batch_size > 1:
        for i, sample in enumerate(samples):
            plt.plot(ts, sample, marker='x', label=f'sample {i}')
    else:
        plt.plot(ts, samples, marker='x', label=f'sample {0}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    #plt.show()



