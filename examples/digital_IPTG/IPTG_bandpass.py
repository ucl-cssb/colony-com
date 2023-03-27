from plate import Plate
from species import Species
import numpy as np
import math
import helper_functions as hf


def get_default_params(w, environment_size):
    ## experimental parameters
    D_I = 1e-4 / w ** 2  # mm^2 per min ***** IPTG DIFFUSION RATE
    T7_0 = 1  # ***** a.u. initial T7RNAP concentration per cell
    R_0 = 1  # ***** a.u. initial REPRESSOR concentration per cell
    GFP_0 = 1  # a.u. ***** initial GFP concentration per cell
    I_0 = 7.5e-3 * 1e-6 * 6.022e23  # ***** initial inducer molecule number - concentration * volume * Avogadro
    X_0 = 0.3 * 1e8 * 10 / (environment_size[0] * environment_size[1])  # ***** initial cell count per grid position - 0.3 OD in 10mL agar ~ 0.3 * 1e8 * 10 / (environment size)

    ## growth parameters (Gompertz growth curves)
    A = 2.5e-1
    um = 3e-4
    lam = 3.75e2

    ## From Zong paper
    alpha_T = 6223  #
    beta_T = 12.8  #
    K_IT = 1400  # 1.4e-6 M @ 1 molecule per nM
    n_IT = 2.3  #
    K_lacT = 15719  #
    alpha_R = 8025
    beta_R = 30.6
    K_IR = 1200  # 1.2e-6 M @ 1 molecule per nM
    n_IR = 2.2
    K_lacR = 14088
    alpha_G = 16462
    beta_G = 19
    n_A = 1.34
    K_A = 2532
    n_R = 3.9
    K_R = 987
    G_s = 1  # GFP scaling parameter

    params = [X_0, A, um, lam,
              I_0, D_I,
              T7_0, alpha_T, beta_T, K_IT, n_IT, K_lacT,
              R_0, alpha_R, beta_R, K_IR, n_IR, K_lacR,
              GFP_0, alpha_G, beta_G, n_A, K_A, n_R, K_R, G_s]

    return params


def get_neythen_params(w, environment_size):
    ## experimental parameters
    D_I = 4.35462879e-02  # mm^2 per min ***** IPTG DIFFUSION RATE
    T7_0 = 1.10207308e-05  # ***** a.u. initial T7RNAP concentration per cell
    R_0 = 2.18075133e-01  # ***** a.u. initial REPRESSOR concentration per cell
    GFP_0 = 0  # a.u. ***** initial GFP concentration per cell
    agar_thickness = 3.12  # mm
    I_0 = 7.5 / (w ** 2 * agar_thickness) # ***** initial inducer concentration - concentration * volume * Avogadro
    X_0 = 5.970081581135449e-05  # ***** initial cell count per grid position

    ## growth parameters (Gompertz fitted growth curves)
    A = 2.5e-1
    um = 3e-4
    lam = 3.75e2

    ## From Zong paper
    alpha_T = 4.88625617e+04  #
    beta_T = 1.83905487e-05  #
    K_IT = 3.95081261e-05  #
    n_IT = 4.47402392e-01  #
    K_lacT = 1.24947521e+04  #
    alpha_R = 1.04554814e+05
    beta_R = 9.67789421e-06
    K_IR = 7.18971464e-03  #
    n_IR = 1.08965612e+01
    K_lacR = 2.45219227e+04
    alpha_G = 1.40349891e+01
    beta_G = 1.01251668e+00
    n_A = 8.56144749e+00
    K_A = 3.70436050e+00
    n_R = 4.49477997e+00
    K_R = 1.87324583e+01

    G_s = 2.13140568e-01  # GFP scaling parameter

    params = [X_0, A, um, lam,
              I_0, D_I,
              T7_0, alpha_T, beta_T, K_IT, n_IT, K_lacT,
              R_0, alpha_R, beta_R, K_IR, n_IR, K_lacR,
              GFP_0, alpha_G, beta_G, n_A, K_A, n_R, K_R, G_s]

    return params


def build_model(w, environment_size, params, inducer_positions, receiver_positions):
    X_0, A, um, lam, \
    I_0, D_I, \
    T7_0, alpha_T, beta_T, K_IT, n_IT, K_lacT, \
    R_0, alpha_R, beta_R, K_IR, n_IR, K_lacR, \
    GFP_0, alpha_G, beta_G, n_A, K_A, n_R, K_R, G_s = params

    ## Create our environment
    plate = Plate(environment_size)

    ## add receiver to the plate
    # receiver_position = [[int(j * (4.5 / w)) for j in i] for i in receiver_positions]  # convert position to specified dims
    # U_X = np.zeros(environment_size)
    # U_X[receiver_position[0], receiver_position[1]] = X_0
    U_X = np.ones(environment_size) * X_0

    def X_behaviour(t, species, params):
        mu = um * np.exp(um * (np.e * lam - np.e * t) / A - np.exp(um * (np.e * lam - np.e * t) / A + 1) + 2)
        dX = mu * species['X']
        return dX

    receiver = Species("X", U_X)
    receiver.set_behaviour(X_behaviour)
    plate.add_species(receiver)

    ## add IPTG to plate
    inducer_position = [[int(j * (4.5 / w)) for j in i] for i in inducer_positions]  # convert position to specified dims

    U_I = np.zeros(environment_size)
    U_I[inducer_position[0], inducer_position[1]] = I_0

    def I_behaviour(t, species, params):
        dI = D_I * hf.ficks(species['I'], w)
        return dI

    iptg = Species("I", U_I)
    iptg.set_behaviour(I_behaviour)
    plate.add_species(iptg)

    # add T7RNAP to the plate
    U_T7 = np.ones(environment_size) * T7_0

    def T7_behaviour(t, species, params):
        mu = um * np.exp(um * (np.e * lam - np.e * t) / A - np.exp(um * (np.e * lam - np.e * t) / A + 1) + 2)
        dT7 = alpha_T * mu * (1 + (species['I'] / K_IT) ** n_IT) / (1 + (species['I'] / K_IT) ** n_IT + K_lacT) + beta_T * mu - mu * species['T7']

        return dT7

    t7 = Species("T7", U_T7)
    t7.set_behaviour(T7_behaviour)
    plate.add_species(t7)

    ## add repressor R to the plate
    U_R = np.ones(environment_size) * R_0

    def R_behaviour(t, species, params):
        mu = um * np.exp(um * (np.e * lam - np.e * t) / A - np.exp(um * (np.e * lam - np.e * t) / A + 1) + 2)
        dR = alpha_R * mu * (1 + (species['I'] / K_IR) ** n_IR) / (1 + (species['I'] / K_IR) ** n_IR + K_lacR) + beta_R * mu - mu * species['R']
        return dR

    repressor = Species("R", U_R)
    repressor.set_behaviour(R_behaviour)
    plate.add_species(repressor)

    ## add GFP to plate
    U_G = np.zeros(environment_size)

    def G_behaviour(t, species, params):
        mu = um * np.exp(um * (np.e * lam - np.e * t) / A - np.exp(um * (np.e * lam - np.e * t) / A + 1) + 2)
        dGFP = alpha_G * mu * species['T7'] ** n_A / (K_A ** n_A + species['T7'] ** n_A) * K_R ** n_R / (
                    K_R ** n_R + species['R'] ** n_R) + beta_G * mu - species['G'] * mu * G_s

        return dGFP

    gfp = Species("G", U_G)
    gfp.set_behaviour(G_behaviour)
    plate.add_species(gfp)

    return plate


def main():
    ## 1536 well plate
    # environment_size = (32, 48)
    environment_size = (16, 16)
    w = 2.25  # inter-well spacing in mm

    # ## 384 well plate
    # # environment_size = (16, 24)
    # environment_size = (8, 8)
    # w = 4.5
    #
    # ## 96 well plate
    # environment_size = (8, 12)
    # w = 9

    params = get_neythen_params(w, environment_size)
    inducer_positions = [[4, 4], [3, 5]]  # positions specified on 384 well plate [[row], [col]]
    receiver_positions = [[4], [4]]  # positions specified on 384 well plate [[row], [col]]

    plate = build_model(w, environment_size, params, inducer_positions, receiver_positions)

    sim = plate.run(t_final=1200 + 1,
                    dt=1,
                    params=params)

    ## plotting
    plate.plot_simulation(sim, 2, 'linear')

main()
