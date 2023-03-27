import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.integrate import solve_ivp


class Plate:
    def __init__(self, size):
        self.size = size  # dimensions of plate (row, column)
        self.species = []  # list of species objects

    def get_size(self):
        return self.size

    def get_num_species(self):
        return len(self.species)

    def get_all_species(self):
        return self.species

    def get_species_by_name(self, name):
        for s in self.species:
            if s.get_name() == name:
                return s
        else:
            return None

    def get_all_species_U(self):
        U = np.zeros((self.get_num_species(), self.size[0], self.size[1]))
        for idx, s in enumerate(self.species):
            U[idx] = s.get_U()
        return U

    def add_species(self, new_species):
        self.species.append(new_species)

    def set_species(self, species):
        self.species = species

    def model(self, t, y, params):
        ## reshape vector 'y' into a 3-dimensional matrix with dimensions [num_species, row_length, column_length]
        U = y.reshape(self.get_all_species_U().shape)

        dU = np.zeros(U.shape)

        ## for each species, get their U matrix and behaviour function
        species_dict = {}
        behaviour_dict = {}
        for idx, s in enumerate(self.species):
            species_dict[s.get_name()] = U[idx]
            behaviour_dict[s.get_name()] = s.behaviour

        ## for each species, run the behaviour function to determine change of timestep
        for idx, s in enumerate(self.species):
            dU[idx] = behaviour_dict[s.get_name()](t, species_dict, params)

        ## flatten and return dU
        return dU.flatten()

    def run(self, t_final, dt, params):
        ## get timepoints
        t = np.arange(0, t_final, dt)

        ## flatten all species U matrix (solver takes state as a vector not matrix)
        U_init = self.get_all_species_U().flatten()

        ## numerically solve model
        sim_ivp = solve_ivp(self.model, [0, t_final], U_init,
                            t_eval=t, args=(params,))

        ## reshape species into matrix [num_species, row_length, column_length, num_timepoints]
        sim_ivp = sim_ivp.y.reshape(self.get_num_species(),
                                    self.size[0], self.size[1],
                                    len(t))

        return sim_ivp

    def plot_simulation(self, sim, num_timepoints, scale='linear', scale_range='fixed', cols=3):
        """
        plots the simulated species at equally spaced timepoints

        args: sim: The 4-dimensional ndarray simulation output num_timepoints: The number of timepoints to plot.
        Equally spaced timepoints are calculated. scale: The scaling of the colourmap for the species: 'linear' or
        'log10' are the only ones allowed currently. scale_range: 'dynamic' or 'fixed'. If 'fixed', the same range
        for the colourmap is used for all timepoints, for 'dynamic' a new range is calculated for each timepoint.
        cols: the number of columns to use for arranging the subplots of each species.

        """
        tps = np.linspace(0, sim.shape[3] - 1, num_timepoints)
        for tp in tps:
            tp = int(tp)

            rows = int(np.ceil(len(self.species) / cols))
            gs = gridspec.GridSpec(rows, cols)
            fig = plt.figure()

            for idx in range(len(self.species)):
                ax = fig.add_subplot(gs[idx])

                if scale == "log10":
                    if scale_range == 'fixed':
                        scale_min = np.min(np.log10(sim[idx, :, :, :]))
                        scale_max = np.max(np.log10(sim[idx, :, :, :]))
                    elif scale_range == 'dynamic':
                        scale_min = np.min(np.log10(sim[idx, :, :, tp]))
                        scale_max = np.max(np.log10(sim[idx, :, :, tp]))

                    im = ax.imshow(np.log10(sim[idx, :, :, tp]), interpolation="none",
                                   cmap=cm.viridis,
                                   vmin=scale_min,
                                   vmax=scale_max)
                elif scale == "linear":
                    if scale_range == 'fixed':
                        scale_min = 0
                        scale_max = np.max(sim[idx, :, :, :])
                    elif scale_range == 'dynamic':
                        scale_min = np.min(sim[idx, :, :, tp])
                        scale_max = np.max(sim[idx, :, :, tp])

                    im = ax.imshow(sim[idx, :, :, tp], interpolation="none",
                                   cmap=cm.viridis,
                                   vmin=scale_min,
                                   vmax=scale_max)

                ax.set_title(self.species[idx].get_name() + ' : ' + str(tp))

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax, shrink=0.8)

            plt.subplots_adjust(wspace=0.6)
            fig.savefig('fig_' + str(tp) + '.png')

            fig.show()

    def plot_plate(self, cols=3):
        print("plotting plate")

        rows = int(np.ceil(len(self.species) / cols))
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure()

        for idx in range(len(self.species)):
            ax = fig.add_subplot(gs[idx])

            im = ax.imshow(self.species[idx].get_U(), interpolation="none", cmap=cm.viridis, vmin=0)
            ax.set_title(self.species[idx].get_name())

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, shrink=0.8)

        fig.savefig('fig_setup.png')
        fig.show()
