from plate import Plate
from species import Species
import numpy as np
import helper_functions as hf


def main():
    ## experimental parameters
    D = 3E-3  # nutrient diffusion coeff (#mm2/min)
    rho_n = 0.3  # consumption rate of nutrients by X
    rc = 6E-3  # growth rate of X on N
    Dc = 1E-5  # cell diffusion coefficient
    w = 1
    Da = 0.03
    rho_A = 0.1  # production rate of AHL

    environment_size = (50, 50)
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_N = np.ones(environment_size)
    N = Species("N", U_N)

    def N_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        n = D * hf.ficks(species['N'], w) - rho_n * species['N'] * (species['R'] + species['S'])
        return n

    N.set_behaviour(N_behaviour)
    plate.add_species(N)

    ## add receiver strain to the plate
    U_R = np.zeros(environment_size)
    for i in np.linspace(5, 45, 9):
        for j in np.linspace(5, 45, 9):
            if (i == 25) & (j == 25):
                continue
            U_R[int(i), int(j)] = 0.001
    # U_R = np.ones(environment_size) * 0.001
    R = Species("R", U_R)

    def R_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        r = Dc * hf.ficks(species['R'], w) + rc * species['N'] * species['R']
        return r

    R.set_behaviour(R_behaviour)
    plate.add_species(R)

    ## add sender strain to the plate
    U_S = np.zeros(environment_size)
    U_S[25, 25] = 0.001
    S = Species("S", U_S)

    def S_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        r = Dc * hf.ficks(species['S'], w) + rc * species['N'] * species['S']
        return r

    S.set_behaviour(S_behaviour)
    plate.add_species(S)

    ## add AHL to plate
    U_A = np.zeros(environment_size)
    A = Species("A", U_A)

    def A_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        a = Da * hf.ficks(species['A'], w) + rho_A * species['S']
        return a

    A.set_behaviour(A_behaviour)
    plate.add_species(A)

    ## add GFP to plate
    U_G = np.zeros(environment_size)
    G = Species("G", U_G)

    def G_behaviour(t, species, params):
        ## unpack params
        D, rho_n, Dc, rc, w, rho_A, Da = params
        g = Dc * hf.ficks(species['G'], w) + hf.leaky_hill(s=species['A'], K=1E-3, lam=2, min=1E-3, max=1) * species['R']
        return g

    G.set_behaviour(G_behaviour)
    plate.add_species(G)

    ## run the experiment
    params = (D, rho_n, Dc, rc, w, rho_A, Da)
    sim = plate.run(t_final=100 * 60,
                    dt=1,
                    params=params)

    ## plotting
    plate.plot_simulation(sim, 10)


main()
