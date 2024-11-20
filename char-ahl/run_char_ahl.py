from plate import Plate
from species import Species
import numpy as np
import helper_functions as hf


def main():

    ## plate parameters
    w  = 4.5   # distance between wells in mm

    ## experimental parameters
    D = 1E-2     # nutrient diffusion coeff (#mm2/min), range 6E−3 to 6E-2 mm2/min
    Da = 6E-3    # AHL diffusion coeff (#mm2/min)

    rho_n = 0.3  # consumption rate of nutrients by X
    rc = 6E-3    # growth rate of X on N
    
    rho_A = 0.1  # production rate of AHL

    R0 = 0.001   # initial concentration of receiver strain
    S0 = 0.001   # initial concentration of sender strain

    # GFP production
    K = 1E-3     # half-maximal AHL concentration
    lam = 2      # Hill coefficient
    min = 1E-3   # minimum GFP production rate
    max = 1      # maximum GFP production rate

    R_pos = [ [1, 4], [2, 2], [3, 5], [4, 3], [5, 1], [5, 6], [6, 4], [7, 2] ]
    S_pos = [ [4, 4] ]

    def N_behaviour(t, species, params):
        D, rho_n, rc, w, rho_A, Da = params
        n = D * hf.ficks(species['N'], w) - rho_n * species['N'] * (species['R'] + species['S'])
        return n

    def R_behaviour(t, species, params):
        D, rho_n, rc, w, rho_A, Da = params
        r = rc * species['N'] * species['R']
        return r

    def S_behaviour(t, species, params):
        D, rho_n, rc, w, rho_A, Da = params
        r = rc * species['N'] * species['S']
        return r
    
    def A_behaviour(t, species, params):
        D, rho_n, rc, w, rho_A, Da = params
        a = Da * hf.ficks(species['A'], w) + rho_A * species['S']
        return a

    def G_behaviour(t, species, params):
        D, rho_n, rc, w, rho_A, Da = params
        g = hf.leaky_hill(species['A'], K, lam, min, max) * species['R']
        return g

    #############################################

    environment_size = (7, 7)
    plate = Plate(environment_size)

    ## add nutrient to the plate
    U_N = np.ones(environment_size)
    N = Species("N", U_N)
    N.set_behaviour(N_behaviour)
    plate.add_species(N)

    ## add receiver strain to the plate
    U_R = np.zeros(environment_size)
    for xy in R_pos:
        U_R[xy[0]-1, xy[1]-1] = R0
    
    R = Species("R", U_R)
    R.set_behaviour(R_behaviour)
    plate.add_species(R)

    ## add sender strain to the plate
    U_S = np.zeros(environment_size)
    #U_S[3, 3] = S0
    for xy in S_pos:
        U_S[xy[0]-1, xy[1]-1] = S0

    S = Species("S", U_S)
    S.set_behaviour(S_behaviour)
    plate.add_species(S)

    ## add AHL to plate
    U_A = np.zeros(environment_size)
    A = Species("A", U_A)
    A.set_behaviour(A_behaviour)
    plate.add_species(A)

    ## add GFP to plate
    U_G = np.zeros(environment_size)
    G = Species("G", U_G)
    G.set_behaviour(G_behaviour)
    plate.add_species(G)

    # plot the setup
    plate.plot_plate()
    #for x in plate.get_all_species():
    #    print(x.get_name())

    ## run the experiment
    params = (D, rho_n, rc, w, rho_A, Da)
    sim = plate.run(t_final=20 * 60,
                    dt=60,
                    params=params)

    print(sim.shape)

    ## plotting
    #plate.plot_simulation(sim, 10)

    # GFP and N timecourse
    plate.plot_timecourse(sim, 4, R_pos, "plot-GFP.pdf")
    plate.plot_timecourse(sim, 1, R_pos, "plot-R.pdf")

    plate.plot_timecourse(sim, 4, R_pos, "plot-GFP-o.pdf", type="o")
    plate.plot_timecourse(sim, 1, R_pos, "plot-R-o.pdf", type="o")

main()
