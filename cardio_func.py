import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
import numpy as np
import control
import numdifftools as nd
from scipy.linalg import solve_continuous_are

class DoyleSDE(torch.nn.Module):

    def __init__(self, u_L, d_L, q_as, q_o2, q_H, c_l, c_r, pinnedList=list()):
        super().__init__()

        self.pinnedList = pinnedList  # this is for matching control parameters to figures at article
        self.hyperParamSearchCounter = 0

        self.d = torch.zeros(1, 1, 1, dtype=torch.float)  # [Watt] d is the workload input of shape # [time-vec, batchSize, 1]; the time-vec should be dense
        self.d_tVec = 0

        self.noise_type = "diagonal"
        self.sde_type = "ito"

        self.paramsDict = self.DoyleParams(q_as, q_o2, q_H, c_l, c_r)
        self.state_size = 4
        self.control_size = 1
        self.input_size = self.d.shape[2]

        # debugging:
        self.enableController = True
        self.enableExternalControlValue = False
        self.externalControlValue = 0
        self.scalarFuncNo = 0

        self.solveIvpRun = False
        self.solveIvpRunBatchNo = 0

        self.dotFactor = 1/60  # all units are in minutes

        self.noiseStd = torch.tensor([0, 0, 0, 0])  # for state_size=4

        Pas_L, Pvs_L, Pap_L, O2v_L = 82, 4.250, 11.6, 154/1000  # some values that are not used later
        x_L = torch.tensor([Pas_L, Pvs_L, Pap_L, O2v_L], dtype=torch.float)[:, None]

        # x_L is [state_size, 1], W_L is [input_size, 1], H_L is [control_size, 1]
        self.referenceValues = {"x_L": x_L}
        self.referenceValues["d_L"], self.referenceValues["u_L"] = torch.tensor([d_L], dtype=torch.float)[:, None], torch.tensor([u_L], dtype=torch.float)[:, None]
        self.referenceValues["x_L"] = self.calcFixedPoint(u=self.referenceValues["u_L"], d=self.referenceValues["d_L"])

        self.K, self.controlBias = self.calc_gain_K(self.referenceValues)

    def DoyleParams(self, q_as, q_o2, q_H, c_l, c_r):
        heartParamsDict = {
            # parameters for subject 1 cl,cr = 0.03, 0.05
            # parameters for subject 2 cl,cr = 0.025, 0.045
            "c_l": c_l,    #  [L / mmHg]
            "c_r": c_r     #  [L / mmHg]
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
            "q_as": q_as,    # 30         # weighting factor, [1 / mmHg]
            "q_o2": q_o2,  # 100000,     # weighting factor, [L blood / L O2]
            "q_H": q_H   # 1            # weighting factor, [1 / min]
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

    def calc_Q_R(self):
        controlParamsDict = self.paramsDict["controlParamsDict"]
        Q, R = torch.zeros(self.state_size, self.state_size), torch.zeros(self.control_size, 1)

        Q[0, 0] = torch.pow(torch.tensor(controlParamsDict["q_as"]), 2)
        Q[3, 3] = torch.pow(torch.tensor(controlParamsDict["q_o2"]), 2)

        R[0, 0] = torch.pow(torch.tensor(controlParamsDict["q_H"]), 2)

        return Q, R

    def calc_gain_K(self, referenceValues):
        # the two solvers do not agree. different solutions to the riccati equation. Matlab solver agrees with my solver
        enableLibraryGainCalc = False

        A_L, B_L = self.calc_A_L_B_L(self.referenceValues)
        Q, R = self.calc_Q_R()

        if enableLibraryGainCalc:
            # calculating the gain using a control library
            C_L, D_L = torch.zeros_like(A_L), torch.zeros(self.state_size, self.control_size)  # aux parameters
            linearizedSys = control.ss(A_L.numpy(), B_L.numpy(), C_L.numpy(), D_L.numpy())

            K, S, E = control.lqr(linearizedSys, Q.numpy(), R.numpy(), N=np.zeros((self.state_size, self.control_size)))
            # the calculation made by self.calc_gain_K and by control.lqr lead to the same gain K.
            # we will take the K from control.lqr for formality...
            K = torch.tensor(K, dtype=torch.float)
        else:
            P = torch.tensor(solve_continuous_are(a=A_L.numpy(), b=B_L.numpy(), q=Q.numpy(), r=R.numpy()), dtype=torch.float)
            K = torch.matmul(torch.linalg.inv(R), torch.matmul(torch.transpose(B_L, 1, 0), P))

        controlBias = torch.matmul(K, referenceValues["x_L"]) + referenceValues["u_L"]
        return K, controlBias


    def calc_A_L_B_L(self, referenceValues):
        enableAnalytic = True
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

        if enableAnalytic:
            B_L = torch.zeros(self.state_size, 1)
            B_L[0, 0] = c_l * Pvp_L / c_as
            B_L[1, 0] = - c_r * Pvs_L / c_vs
            B_L[2, 0] = c_r * Pvs_L / c_ap

            B_L = self.dotFactor * B_L

            A_L = torch.zeros(self.state_size, self.state_size)

            A_L[0, 0] = - 1/(c_as)*(c_as*c_l*H_L/c_vp + 1/Rs_L)
            A_L[0, 1] = 1/(c_as)*(-c_vs*c_l*H_L/c_vp + 1/Rs_L)
            A_L[0, 2] = -c_ap*c_l*H_L/(c_as*c_vp)
            A_L[0, 3] = A*Fs_L/(c_as*Rs_L)

            A_L[1, 0] =  1/(c_vs*Rs_L)
            A_L[1, 1] = - (1/c_vs)*(c_r*H_L + 1/Rs_L)
            A_L[1, 3] = - A*Fs_L/(c_vs*Rs_L)

            A_L[2, 0] = - c_as/(c_ap*c_vp*R_p)
            A_L[2, 1] = (1/c_ap)*(c_r*H_L - c_vs/(c_vp*R_p))
            A_L[2, 2] = - 1/(c_ap*R_p)*(1 + c_ap/c_vp)

            A_L[3, 0] = delta_O2_L/(Rs_L*V_TO2)
            A_L[3, 1] = -A_L[3, 0]
            A_L[3, 3] = -(Fs_L/V_TO2)*(A*delta_O2_L/Rs_L + 1)

            A_L = self.dotFactor*A_L
        else:
            enableControllerOrigVal = self.enableController
            self.enableController = False
            fun = lambda x: self.f(0, torch.tensor(x)[None, :, :][:, :, 0])[0, :][:, None].numpy()
            A_L = torch.tensor(nd.Jacobian(fun)(x_L.numpy()), dtype=torch.float)
            self.enableController = enableControllerOrigVal

            B_L = torch.zeros(self.state_size, 1, dtype=torch.float)
            for s in range(self.state_size):
                self.scalarFuncNo = s
                f = nd.Derivative(self.scalar_f_externalControl, full_output=True)
                val, info = f(self.referenceValues["u_L"][0].numpy())
                B_L[s, 0] = torch.tensor(val, dtype=torch.float)

        return A_L, B_L

    def calcControl(self, x, t):
        # x is the state vector: [batch_size, state_size, 1], [Pas, Pvs, Pap, O2v]
        batch_size = x.shape[0]

        if self.enableController:
            K = self.K
            u = - torch.matmul(K, x) + self.controlBias[None, :, :].repeat(batch_size, 1, 1)
            # u is the control [batch_size, control_size, 1]
        elif self.enableExternalControlValue:
            u = torch.tensor(self.externalControlValue, dtype=torch.float)[None, None].repeat(batch_size, 1, 1)
        else: # using reference value
            u_L = self.referenceValues["u_L"]
            # u_L is the reference heart rate at low exercise [control_size, 1], [H]
            u = u_L[None, :, :].repeat(batch_size, 1, 1)
        '''
        # manual control
        # print(f'input time is {t}')
        tStart = 22.5
        a = 5
        b = 0.8
        if t > tStart:
            u = self.referenceValues["u_L"] + a * np.log(b*(1 + t - tStart))
        else:
            u = self.referenceValues["u_L"]
        u = u[None, :, :].repeat(batch_size, 1, 1)
        print(f't: {t}, HR: {u}')
        '''
        return u

    def controlParamsOptimizeSciPy(self, controlParams):
        return self.controlParamsOptimize(controlParams.tolist())

    def controlParamsOptimize(self, controlParamList):
        q_as, q_o2, c_l, c_r = controlParamList
        q_H = 1

        self.paramsDict["controlParamsDict"]["q_as"] = q_as
        self.paramsDict["controlParamsDict"]["q_o2"] = q_o2
        self.paramsDict["controlParamsDict"]["q_H"] = q_H
        self.paramsDict["heartParamsDict"]["c_l"] = c_l
        self.paramsDict["heartParamsDict"]["c_r"] = c_r

        self.K, self.controlBias = self.calc_gain_K(self.referenceValues)

        pinned_Times, pinned_HR, pinned_Pas, pinned_O2, simDuration = self.pinnedList

        with torch.no_grad():
            x_k = self.runSolveIvp(self.referenceValues["x_L"][None, :, :][:, :, 0], simDuration)
            u_k = self.reCalcControl(x_k)
            tVec = self.getTvec(x_k)
        score = self.calcScore(x_k, u_k, tVec, pinned_Times, pinned_HR, pinned_Pas, pinned_O2)

        print(f'q_as = {q_as}, q_o2 = {q_o2}, q_H = {q_H}, c_l = {c_l}, c_r = {c_r}, score = {score}')

        return score

    def runSolveIvp(self, x_0, d, d_tVec, simDuration):
        self.d, self.d_tVec = d, d_tVec
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

    def rootWrapper(self, x, *args):
        paramsDict = args[0]
        t=0 # not in use
        return self.calc_dx(t, x.tolist(), paramsDict)

    def calcFixedPoint(self, u, d):
        # u,d - scalars

        # save original values:
        enableControllerOrigVal = self.enableController
        controlOrigVal = self.referenceValues["u_L"]
        inputOrigVal = self.d

        # change to requested fixed-point environment
        self.enableController = False  # Now H has the value of H_L, W has the value of d(t=0)
        self.d = d*torch.ones_like(self.d)
        self.referenceValues["u_L"] = u*torch.ones_like(self.referenceValues["u_L"])

        fixedPoint = root(self.rootWrapper, self.referenceValues["x_L"].numpy()[:, 0], self.paramsDict)

        # restore original values:
        self.enableController = enableControllerOrigVal
        self.d = inputOrigVal
        self.referenceValues["u_L"] = controlOrigVal

        return torch.tensor(fixedPoint.x, dtype=torch.float)[:, None]

    def staticOptimization(self, u_star, d_star, d):
        x_star = self.calcFixedPoint(u=u_star, d=d_star)
        Q, R = self.calc_Q_R()
        # optimization:
        H_values = torch.linspace(40, 170, 2*(170-40+1), dtype=torch.float)

        minJsWasWritten = False
        for u in H_values:
            x = self.calcFixedPoint(u=u, d=d)
            if x.min() < 0:  # all the physiological values in the state vec are positive in a valid solution
                continue
            uDiff, xDiff = (u - u_star)[None, None], x - x_star
            Js = torch.matmul(torch.transpose(xDiff, 1, 0), torch.matmul(Q, xDiff)) + torch.matmul(torch.transpose(uDiff, 1, 0), torch.matmul(R, uDiff))
            if not minJsWasWritten:
                minJs = Js
                tilde_H = u
                minJsWasWritten = True
            elif Js < minJs:
                minJs = Js
                tilde_H = u

        tilde_x = self.calcFixedPoint(u=tilde_H, d=d)
        return tilde_x, tilde_H

    def create_figure_S4(self):
        origControlParams = self.paramsDict["controlParamsDict"]

        W_values = torch.linspace(0, 250, 2*(250-0+1), dtype=torch.float)

        tilde_x, tilde_H = torch.zeros(W_values.shape[0], self.state_size, 1), torch.zeros(W_values.shape[0], self.control_size, 1)
        for i, d in enumerate(W_values):
            if d < 110:
                d_star = 0  # watt
                u_star = 48  # heart-rate @ min
                self.paramsDict["controlParamsDict"]["q_as"], self.paramsDict["controlParamsDict"]["q_o2"], self.paramsDict["controlParamsDict"]["q_H"] = np.sqrt(2), np.sqrt(1e7), np.sqrt(1.5)
            else:
                d_star = 100  # watt
                u_star = 114  # heart-rate @ min
                self.paramsDict["controlParamsDict"]["q_as"], self.paramsDict["controlParamsDict"]["q_o2"], self.paramsDict["controlParamsDict"]["q_H"] = np.sqrt(4), np.sqrt(1e7), np.sqrt(4)
            tilde_x[i], tilde_H[i] = self.staticOptimization(u_star=u_star, d_star=d_star, d=d)

        self.paramsDict["controlParamsDict"] = origControlParams

        # plot:
        W_values, tilde_x, tilde_H = W_values.numpy(), tilde_x.numpy(), tilde_H.numpy()
        plt.figure()
        plt.plot(W_values, tilde_x[:, 0, 0], label="Pas", color='blue')
        delta_O2 = self.paramsDict["circulationParamsDict"]["O_2a"] - tilde_x[:, 3, 0]
        plt.plot(W_values, 1000*delta_O2, label=r'$\Delta O_2*1000$', color='green')
        plt.plot(W_values, tilde_H[:, 0, 0], label="H", color='red')
        plt.grid()
        plt.legend()
        plt.xlabel('Workload [watts]')
        plt.show()

    def scalar_f_externalControl(self, externalControlValue):
        origEnableControlVal, origEnableExternalControlVal = self.enableController, self.enableExternalControlValue
        self.enableController, self.enableExternalControlValue = False, True
        self.externalControlValue = externalControlValue

        xDot = self.f(0, self.referenceValues["x_L"][None, :, :][:, :, 0])

        self.enableController, self.enableExternalControlValue = origEnableControlVal, origEnableExternalControlVal

        return xDot[0, self.scalarFuncNo:self.scalarFuncNo+1].numpy()

    # Drift
    def f(self, t, x):
        # x is the state vector: [batch_size, state_size], [Pas, Pvs, Pap, O2v]
        # d is the workload input: [batch_size, 1], [W]

        dIndex = torch.argmin(torch.abs(self.d_tVec - t))
        W = self.d[dIndex]

        circulationParamsDict, heartParamsDict = self.paramsDict["circulationParamsDict"], self.paramsDict["heartParamsDict"]
        c_as, c_vs, c_ap, c_vp = circulationParamsDict["c_as"], circulationParamsDict["c_vs"], circulationParamsDict["c_ap"], circulationParamsDict["c_vp"]
        V_tot, A, R_s0, rho, M_0, R_p, O_2a, V_TO2 = circulationParamsDict["V_tot"], circulationParamsDict["A"], circulationParamsDict["R_s0"], circulationParamsDict["rho"], circulationParamsDict["M_0"], circulationParamsDict["R_p"], circulationParamsDict["O_2a"], circulationParamsDict["V_TO2"]

        H = self.calcControl(x[:, :, None], t)

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

    def reCalcControl(self, x_k):
        # x_k shape (nSamples, batch_size, state_size)
        nSamples, batch_size = x_k.shape[0], x_k.shape[1]
        tVec = self.getTvec(x_k)
        # recalc the control:
        u_k = torch.zeros(nSamples, batch_size, self.control_size, dtype=torch.float)
        for k in range(nSamples):
            u_k[k] = self.calcControl(x_k[k][:, :, None], tVec[k])[:, :self.control_size, 0]
        return u_k

    def getTvec(self, x_k):
        nSamples, batch_size = x_k.shape[0], x_k.shape[1]
        fs = self.paramsDict["displayParamsDict"]["fs"]
        tVec = np.arange(0, nSamples) / fs
        return torch.tensor(tVec, dtype=torch.float)

    def calcScore(self, x_k, u_k, tVec, pinned_Times, pinned_HR, pinned_Pas, pinned_O2):
        tVecQueryIndexes = torch.zeros_like(pinned_Times)
        for i, time in enumerate(pinned_Times):
            tVecQueryIndexes[i] = torch.argmin(torch.abs(tVec - time))

        Pas_at_queryTimes = x_k[tVecQueryIndexes.long()][:, 0, 0]
        O2_at_queryTimes = 1000 * x_k[tVecQueryIndexes.long()][:, 0, 3]
        HR_at_queryTimes = u_k[tVecQueryIndexes.long()][:, 0, 0]

        Pas_score = torch.norm(Pas_at_queryTimes - pinned_Pas)
        O2_score = torch.norm(O2_at_queryTimes - pinned_O2)
        HR_score = torch.norm(HR_at_queryTimes - pinned_HR)
        score = Pas_score + O2_score + HR_score
        #score = HR_score

        return score

    def plot(self, x_k):
        # x_k shape (nSamples, batch_size, state_size)
        nSamples, batch_size = x_k.shape[0], x_k.shape[1]
        tVec = self.getTvec(x_k)
        # recalc the control:
        u_k = self.reCalcControl(x_k)

        x_k[:, :, 3] = 1000*x_k[:, :, 3] # to converts the units from [L O2 / L blood] to [mL O2 / L blood]

        x_u_k = torch.cat((x_k, u_k), dim=2)
        x_u_k = x_u_k.cpu().numpy()



        ylabels = ["Pas", "Pvs", "Pap", "O2v", "HR", "W"]
        colors = ['magenta', 'yellow', 'yellow', 'green', 'black', 'blue']
        '''
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
        '''
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


def generate_workload_profile(batch_size, simDuration, enableInputWorkload):
    dfs = 100  # [hz]
    dVal = 53.3  # [watt]
    d_tVec = torch.tensor(np.arange(0, np.ceil(simDuration * dfs)) / dfs)
    d = torch.zeros(d_tVec.shape[0], batch_size, 1, dtype=torch.float)
    if enableInputWorkload:
        workStartTimes = np.array([33.17, 100, 225])  # [sec]
        workStopTimes = np.array([74.4, 160, 310])  # [sec]
        workStartIndexes = np.round(dfs * workStartTimes)
        workStopIndexes = np.round(dfs * workStopTimes)
        for i, startIndex in enumerate(workStartIndexes):
            stopIndex = workStopIndexes[i]
            d[int(startIndex):int(stopIndex) + 1, :, 0] = dVal

        # Pinned values from figure S14:
        workPinnedTimes = np.array([28.43, 28.9, 29.383, 29.857, 33.17])
        workPinnedValues = np.array([0, 9.765, 28.544, 40.1877, 53.33])
        workStartIndexes = np.round(dfs * workPinnedTimes)
        workStopTimes = workPinnedTimes[1:]
        workStopIndexes = np.round(dfs * workStopTimes)
        for i, startIndex in enumerate(workStartIndexes):
            if i == 4:
                stopIndex = workStopIndexes[i - 1]
                d[int(startIndex):int(stopIndex) + 1, :, 0] = torch.linspace(workPinnedValues[i], workPinnedValues[i], int(stopIndex) - int(startIndex) + 1)[:, None]
            else:
                stopIndex = workStopIndexes[i]
                d[int(startIndex):int(stopIndex) + 1, :, 0] = torch.linspace(workPinnedValues[i], workPinnedValues[i + 1], int(stopIndex) - int(startIndex) + 1)[:, None]

    return d, d_tVec



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



