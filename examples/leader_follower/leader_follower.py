from plate import Plate
from species import Species
import numpy as np
import math
import helper_functions as hf


def main():
    ## 1536 well plate
    environment_size = (32, 48)
    w = 2.25  # inter-well spacing in mm

    # ## 384 well plate
    # environment_size = (16, 24)
    # w = 4.5
    #
    # ## 96 well plate
    # environment_size = (8, 12)
    # w = 9

    ## experimental parameters
    # growth
    mu_max = 0.02  # per min  *****  max growth rate
    K_mu = 6e-3  # g  ***** growth Michaelis-Menten coeffecient
    gamma = 1E8  # cells per g  ***** yield
    # diffusion
    D_N = 800 * (1e-3) ** 2 * 60 / w ** 2  # mm^2 per min  ***** nutrient diffusion rate (800 um2/s)
    D_A = 400 * (1e-3) ** 2 * 60 / w ** 2  # mm^2 per min ***** IPTG DIFFUSION RATE
    D_B = 400 * (1e-3) ** 2 * 60 / w ** 2  # mm^2 per min ***** AHL DIFFUSION RATE (400 um2/s)
    # chemotaxis
    D_L = 450 * (1e-3) ** 2 * 60 / w ** 2  # mm^2 per min  ***** max leader movement rate (450 um^2/s)
    D_F = 450 * (1e-3) ** 2 * 60 / w ** 2  # mm^2 per min  ***** max follower movement rate
    K_dL = 1e-3 * 1e9  # half-max induction: 1e-3 M @ 1 molecule per nM
    lam_dL = 2  # Hill coefficient of leader chemotaxis induction
    K_dF = 1e-8 * 1e9  # half-max induction: 1e-8 M @ 1 molecule per nM
    lam_dF = 2  # Hill coefficient of follower chemotaxis induction
    # expression
    k_B = 1  # molecules per cell per min

    ## initial conditions
    N_0 = 0.04 / (environment_size[0] * environment_size[
        1])  # g ***** initial nutrient per grid position - 0.4% = 0.4g per 100 mL -> in 10mL agar = 0.04 / (environment_size)
    L_0 = 0.1 * 1e8 * 1e-3  # ***** initial cell count per grid position - 0.1 OD in 1 uL inoculum @ 1e8 cells per mL when OD = 1
    F_0 = 0.1 * 1e8 * 1e-3  # ***** initial cell count per grid position - 0.1 OD in 1 uL inoculum
    leader_positions = [[8], [11]]  # positions specified on 384 well plate [[row], [col]]
    follower_positions = [[8], [13]]  # positions specified on 384 well plate [[row], [col]]
    inducer_positions = [[4], [11]]

    ## Create our environment
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_N = np.ones(environment_size) * N_0
    N = Species("N", U_N)

    def N_behaviour(t, species, params):
        # unpack params
        w, mu_max, K_mu, gamma, D_N, D_L, D_F, D_A, D_B, K_dL, lam_dL, K_dF, lam_dF, k_B = params

        mu = mu_max * species['N'] / (K_mu + species['N'])
        dN = D_N * hf.ficks(species['N'], w) - (mu * species['L'] / gamma) - (mu * species['F'] / gamma)
        return dN

    N.set_behaviour(N_behaviour)
    plate.add_species(N)

    ## add leader strain to the plate
    U_L = np.zeros(environment_size)
    leader_position = [[int(j * (4.5 / w)) for j in i] for i in leader_positions]  # convert position to specified dims
    U_L[leader_position[0], leader_position[1]] = L_0
    leader = Species("L", U_L)

    def L_behaviour(t, species, params):
        ## unpack params
        w, mu_max, K_mu, gamma, D_N, D_L, D_F, D_A, D_B, K_dL, lam_dL, K_dF, lam_dF, k_B = params

        mu = mu_max * species['N'] / (K_mu + species['N'])

        dL = D_L * hf.ficks(species['L'] * hf.hill(s=species['A'], K=K_dL, lam=lam_dL), w) + mu * species['L']
        return dL

    leader.set_behaviour(L_behaviour)
    plate.add_species(leader)

    ## add follower strain to the plate
    U_F = np.zeros(environment_size)
    follower_position = [[int(j * (4.5 / w)) for j in i] for i in
                         follower_positions]  # convert position to specified dims
    U_F[follower_position[0], follower_position[1]] = F_0
    follower = Species("F", U_F)

    def F_behaviour(t, species, params):
        ## unpack params
        w, mu_max, K_mu, gamma, D_N, D_L, D_F, D_A, D_B, K_dL, lam_dL, K_dF, lam_dF, k_B = params

        mu = mu_max * species['N'] / (K_mu + species['N'])

        dF = D_F * hf.ficks(species['F'] * hf.hill(s=species['B'], K=K_dF, lam=lam_dF), w) + mu * species['F']
        return dF

    follower.set_behaviour(F_behaviour)
    plate.add_species(follower)

    ## add A molecule to plate
    U_A = np.zeros(environment_size)
    inducer_position = [[int(j * (4.5 / w)) for j in i] for i in
                        inducer_positions]  # convert position to specified dims
    U_A[inducer_position[0], inducer_position[1]] = 1e6
    # grad_values = np.logspace(0, 9, environment_size[0])
    # for idx, value in enumerate(grad_values):
    #     U_A[idx,:] = value
    A = Species("A", U_A)

    def A_behaviour(t, species, params):
        ## unpack params
        w, mu_max, K_mu, gamma, D_N, D_L, D_F, D_A, D_B, K_dL, lam_dL, K_dF, lam_dF, k_B = params

        dA = D_A * hf.ficks(species['A'], w)
        return dA

    A.set_behaviour(A_behaviour)
    plate.add_species(A)

    ## add B molecule to plate
    U_B = np.zeros(environment_size)
    B = Species("B", U_B)

    def B_behaviour(t, species, params):
        ## unpack params
        w, mu_max, K_mu, gamma, D_N, D_L, D_F, D_A, D_B, K_dL, lam_dL, K_dF, lam_dF, k_B = params

        dB = D_B * hf.ficks(species['B'], w) + hf.leaky_hill(s=species['A'], max=0.1, min=0, lam=2, K=1e-3 * 1e9) * species['L']
        return dB

    B.set_behaviour(B_behaviour)
    plate.add_species(B)

    ## run the experiment
    params = (w, mu_max, K_mu, gamma, D_N, D_L, D_F, D_A, D_B, K_dL, lam_dL, K_dF, lam_dF, k_B)
    sim = plate.run(t_final=24 * 60 + 1,
                    dt=1,
                    params=params)

    ## plotting
    plate.plot_simulation(sim, 6, cols=2)


main()
