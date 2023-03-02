from plate import Plate
from species import Species
import numpy as np
import helper_functions as hf


def main():
    ## experimental parameters
    w = 1

    D_p = 500 * (6 * 10 ** -5)  # max chemotaxis rate (450 um2/s)
    D_p0 = 10 * (6 * 10 ** -5)  # min chemotaxis rate (10 um2/s)
    D_h = 400 * (6 * 10 ** -5)  # AHL diffusion rate
    D_n = 800 * (6 * 10 ** -5)  # nutrient diffusion coeff (800 um2/s)
    D_b = 100 * (6 * 10 ** -5)  # bacteriocin diffusion rate

    gamma = 0.7 / 60  # growth rate (0.7 hr-1)
    beta = 1.04 / 60  # AHL half-life
    m = 20  #
    n_0 = 15E8  #
    k_n = 1  #
    K_n = 1E9  #
    K_h = 4E8  #
    alpha = beta  # AHL production rate
    omega = 5 * 10 ** -1

    dim_mm = 120
    dim = int(dim_mm / w)
    environment_size = (int(dim * (2 / 3)), dim)
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_n = np.ones(environment_size) * n_0
    n = Species("n", U_n)

    def n_behaviour(t, species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega = params

        dn = D_n * hf.ficks(species['n'], w) - (k_n * gamma * species['n'] ** 2 * species['p']) / (species['n'] ** 2 + K_n ** 2)
        return dn

    n.set_behaviour(n_behaviour)
    plate.add_species(n)

    ## add strain to the plate
    U_p = np.ones(environment_size) * 1E7
    p = Species("p", U_p)

    def p_behaviour(t, species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega = params

        mu_h = (D_p + D_p0 * (species['h'] / K_h) ** m) / (1 + (species['h'] / K_h) ** m)
        dp = hf.ficks(mu_h * species['p'], w) + (gamma * species['n'] ** 2 * species['p']) / (
                    species['n'] ** 2 + K_n ** 2) - omega * species['b'] * species['p']
        return dp

    p.set_behaviour(p_behaviour)
    plate.add_species(p)

    ## add AHL to plate
    U_h = np.zeros(environment_size)
    h = Species("h", U_h)

    def h_behaviour(t, species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega = params

        dh = D_h * hf.ficks(species['h'], w) + alpha * species['p'] - beta * species['h']
        return dh

    h.set_behaviour(h_behaviour)
    plate.add_species(h)

    ## add bacteriocin to the plate
    U_b = np.zeros(environment_size)
    # U_b[int(dim/2), int(dim/2)] = 1 * 10**0
    U_b[[20, 40, 60], 15] = 0.1
    U_b[[20, 40, 60], 30] = 0.2
    U_b[[20, 40, 60], 45] = 0.5
    U_b[[20, 40, 60], 60] = 1
    U_b[[20, 40, 60], 75] = 2
    U_b[[20, 40, 60], 90] = 5
    U_b[[20, 40, 60], 105] = 10
    b = Species('b', U_b)

    def b_behaviour(t, species, params):
        ## unpack params
        w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega = params

        b = D_b * hf.ficks(species['b'], w) - beta * species['b']
        return b

    b.set_behaviour(b_behaviour)
    plate.add_species(b)

    plate.plot_plate()

    ## run the experiment
    params = (w, D_p, D_p0, D_h, D_n, D_b, gamma, beta, m, n_0, k_n, K_n, K_h, alpha, omega)
    sim = plate.run(t_final=24 * 60,
                    dt=1,
                    params=params)

    ## plotting
    plate.plot_simulation(sim, 10)


main()
