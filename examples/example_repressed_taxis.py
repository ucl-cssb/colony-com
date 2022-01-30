from plate import Plate
from species import Species
import numpy as np
import helper_functions as hf


def main():
    ## experimental parameters
    w = 0.5

    D_p = 500 * (6 * 10**-5)    # max chemotaxis rate (450 um2/s)
    D_p0 = 10 * (6 * 10**-5)    # min chemotaxis rate (10 um2/s)
    D_h = 400 * (6 * 10**-5)   # AHL diffusion rate
    D_n = 800 * (6 * 10**-5)  # nutrient diffusion coeff (800 um2/s)

    gamma = 0.7 / 60  # growth rate (0.7 hr-1)
    beta = 1.04 / 60  # AHL half-life
    m = 20  #
    n_0 = 15E8  #
    k_n = 1  #
    K_n = 1E9   #
    K_h = 4E8   #
    alpha = beta    # AHL production rate

    dim_mm = 90
    dim = int(dim_mm / w)
    environment_size = (dim, dim)
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_n = np.ones(environment_size) * n_0
    n = Species("n", U_n)
    def n_behaviour(t, species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, gamma, beta, m, n_0, k_n, K_n, K_h, alpha = params

        dn = D_n * hf.ficks(species['n'], w) - (k_n * gamma * species['n']**2 * species['p']) / (species['n']**2 + K_n**2)
        return dn
    n.set_behaviour(n_behaviour)
    plate.add_species(n)

    ## add strain to the plate
    U_p = np.zeros(environment_size)
    for r in np.arange(3./w, -0.001/w, -1):
        for i in np.arange((dim/2) - r, (dim/2) + r):
            for j in np.arange((dim / 2) - r, (dim / 2) + r):
                U_p[int(i), int(j)] = 2 * np.exp(-(r*w) ** 2 / 4) * 10 ** 8

    #U_p[50, 50] = 2E8
    p = Species("p", U_p)
    def p_behaviour(t, species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, gamma, beta, m, n_0, k_n, K_n, K_h, alpha = params

        mu_h = (D_p + D_p0 * (species['h'] / K_h)**m) / (1 + (species['h'] / K_h)**m)
        dp = hf.ficks(mu_h * species['p'], w) + (gamma * species['n']**2 * species['p']) / (species['n']**2 + K_n**2)
        return dp
    p.set_behaviour(p_behaviour)
    plate.add_species(p)

    ## add AHL to plate
    U_h = np.zeros(environment_size)
    h = Species("h", U_h)
    def h_behaviour(t, species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, gamma, beta, m, n_0, k_n, K_n, K_h, alpha = params

        dh = D_h * hf.ficks(species['h'], w) + alpha * species['p'] - beta * species['h']
        return dh
    h.set_behaviour(h_behaviour)
    plate.add_species(h)

    # plate.plot_plate()

    ## run the experiment
    params = (w, D_p, D_p0, D_h, D_n, gamma, beta, m, n_0, k_n, K_n, K_h, alpha)
    sim = plate.run(t_final=2000,
                    dt=20,
                    params=params)

    ## plotting
    plate.plot_simulation(sim, 10)

    ## make video of P over time
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import cm

    plate_view = sim[1]

    fig, ax = plt.subplots()
    plt.axis('off')
    ims = []
    for idx in range(plate_view.shape[2]):
        im = ax.imshow(plate_view[:, :, idx],
                       interpolation="none",
                       cmap=cm.gist_gray,
                       vmin=0,
                       vmax=np.max(plate_view),
                       animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=5000/len(ims), blit=True,
                                    repeat_delay=1000)
    ani.save("movie_repressed_taxis.mp4")


main()
