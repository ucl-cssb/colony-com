from plate import Plate
from species import Species
import numpy as np
import helper_functions


def main():
    ## experimental parameters
    w = 0.1
    D = 0.1  # nutrient diffusion coeff (#mm2/min)

    dim_mm = 40
    dim = int(dim_mm / w)
    environment_size = (dim, dim)
    plate = Plate(environment_size)

    ## add morphogen to the plate
    U_N = np.zeros(environment_size)
    U_N[int(environment_size[0]/2), int(environment_size[1]/2)] = 1000

    # positions = [1/2]
    # U_N = np.zeros(environment_size)
    # for p in positions:
    #     for r in np.arange(3/w, -0.001/w, -1):
    #         for i in np.arange((dim*p) - r, (dim*p) + r):
    #             for j in np.arange((dim / 2) - r, (dim / 2) + r):
    #                 U_N[int(i), int(j)] = 1000 * np.exp(-(r*w) ** 2 / 4)
    N = Species("N", U_N)

    def N_behaviour(t, species, params):
        ## unpack params
        D, w = params
        dN = D * helper_functions.ficks(species['N'], w)

        # hold source constant
        # dN[int(environment_size[0] / 2), int(environment_size[1] / 2)] = 0
        return dN

    N.set_behaviour(N_behaviour)
    plate.add_species(N)

    ## run the experiment
    params = (D, w)
    sim = plate.run(t_final=10,
                    dt=.01,
                    params=params)

    ## plotting
    # plate.plot_simulation(sim, 10, scale_range='dynamic', cols=1)
    plate.plot_simulation_3D(sim, 11, cols=1)

    ## plot 1D cross section
    # sim_1D = sim[0, int(environment_size[0] / 2), :, :]



main()
