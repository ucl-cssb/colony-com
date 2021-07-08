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

    ## growth parameters (currently Monod growth but can be replaced with fitted growth curves)
    D_N = 1e-4 / w**2  # mm^2 per min  ***** nutrient diffusion rate
    mu_max = 0.02  # per min  *****  max growth rate
    K_mu = 1  # g  ***** growth Michaelis-Menten coeffecient
    gamma = 1E12  # cells per g  ***** yield

    ## experimental parameters
    D_A = 1e-4 / w**2  # mm^2 per min ***** IPTG DIFFUSION RATE
    T7_0 = 1   # ***** a.u. initial T7RNAP concentration per cell
    R_0 = 1   # ***** a.u. initial REPRESSOR concentration per cell
    GFP_0 = 1  # a.u. ***** initial GFP concentration per cell
    X_0 = 0.3 * 1e8 * 10 / (environment_size[0] * environment_size[1])  # ***** initial cell count per grid position - 0.3 OD in 10mL agar ~ 0.3 * 1e8 * 10 / (environment size)
    N_0 = 0.04 / (environment_size[0] * environment_size[1])  # g ***** initial nutrient per grid position - 0.4% = 0.4g per 100 mL = 0.04 / (environment_size)

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

    ## Create our environment
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_N = np.ones(environment_size) * N_0
    N = Species("N", U_N)
    def N_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, \
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        n = D_N * hf.ficks(species['N'], w) - mu_max * K_mu / (K_mu + species['N']) * species['X'] / gamma
        return n
    N.set_behaviour(N_behaviour)
    plate.add_species(N)

    ## add one strain to the plate
    U_X = np.ones(environment_size) * X_0
    strain = Species("X", U_X)
    def X_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, \
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        x = mu_max * K_mu / (K_mu + species['N']) * species['X']
        return x
    strain.set_behaviour(X_behaviour)
    plate.add_species(strain)

    ## add IPTG to plate
    inducer_num = 10e-3 * 1e-6 * 6.022e23  # concentration * volume * Avogadro

    inducer_position = [[8, 8], [11, 13]]  # positions specified on 384 well plate [[row], [col]]
    inducer_position = [[int(j * (4.5/w)) for j in i] for i in inducer_position]  # convert position to specified dims

    U_A = np.ones(environment_size)
    U_A[inducer_position[0], inducer_position[1]] = inducer_num

    A = Species("A", U_A)
    def A_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0, \
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        a = D_A * hf.ficks(species['A'], w)
        return a
    A.set_behaviour(A_behaviour)
    plate.add_species(A)

    ## add GFP to plate
    U_G = np.ones(environment_size) * GFP_0
    G = Species("G", U_G)
    def G_behaviour(t, species, params):
        ## unpack params
        w, D_N, mu_max, K_mu, gamma, D_A, \
        alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0, \
        alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0,\
        alpha_G, beta_G, n_A, K_A, n_R, K_R = params

        mu = mu_max * K_mu / (K_mu + species['N'])

        T7 = alpha_T + beta_T - alpha_T * K_lacT / (1 + (species['A'] / K_IT)**n_IT + K_lacT) + T7_0 * np.exp(-mu * t)
        R = alpha_R + beta_R - alpha_R * K_lacR / (1 + (species['A'] / K_IR)**n_IR + K_lacR) + R_0 * np.exp(-mu * t)
        # R = 0  # produces treshold

        dGFP = (alpha_G * mu * T7**n_A / (K_A**n_A + T7**n_A) * K_R**n_R / (K_R**n_R + R**n_R) + beta_G * mu - species['G'] * mu)

        return dGFP
    G.set_behaviour(G_behaviour)
    plate.add_species(G)

    ## run the experiment
    params = (w, D_N, mu_max, K_mu, gamma, D_A,
              alpha_T, beta_T, K_IT, n_IT, K_lacT, T7_0,
              alpha_R, beta_R, K_IR, n_IR, K_lacR, R_0,
              alpha_G, beta_G, n_A, K_A, n_R, K_R)
    sim = plate.run(t_final = 12*60 + 1,
                    dt = 1,
                    params = params)

    ## plotting
    # plate.plot_simulation(sim, 3, 'log10', 2)

    # calculate total GFP (rather than GFP per cell) = X * GFP
    plate_view = sim[1] * sim[3]

    import matplotlib.pyplot as plt
    from matplotlib import cm

    timpoints = 10
    fig, axs = plt.subplots(timpoints, sharex='all', sharey='all')

    tps = np.linspace(0, plate_view.shape[2] - 1, timpoints)
    for idx, ax in enumerate(axs.flatten()):
        im = ax.imshow(plate_view[:, :, int(tps[idx])],
                       interpolation="none",
                       cmap=cm.gist_gray,
                       vmin=np.min(plate_view),
                       vmax=np.max(plate_view))

        ax.set_ylabel(int(tps[idx]))
    fig.show()

    # import matplotlib.animation as animation
    # fig, ax = plt.subplots()
    # ims = []
    # for idx in range(plate_view.shape[2]):
    #     im = ax.imshow(np.log10(plate_view[:, :, idx]),
    #                    interpolation="none",
    #                    cmap=cm.gist_gray,
    #                    vmin=np.min(np.log10(plate_view)),
    #                    vmax=np.max(np.log10(plate_view)),
    #                    animated=True)
    #     ims.append([im])
    #
    # ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
    #                                 repeat_delay=1000)
    # ani.save("movie_banpdass_far.mp4")

main()