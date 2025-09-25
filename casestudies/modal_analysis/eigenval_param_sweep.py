import src.dynamic as dps
import src.modal_analysis as dps_mdl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap


class EigenSweep:
    def __init__(self, eigenvalues_list, participation_factors_list, state_desc, parameter_values):
        self.eigenvalues_list = eigenvalues_list
        self.participation_factors_list = participation_factors_list
        self.state_desc = state_desc
        self.parameter_values = parameter_values

        self.root = tk.Tk()
        self.root.title("Eigenvalue Plot")

        self.search_frame = tk.Frame(self.root)
        self.search_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.search_entries = []
        self.add_search_field()

        self.add_button = tk.Button(self.search_frame, text="Add Search Field", command=self.add_search_field)
        self.add_button.pack()

        self.update_button = tk.Button(self.search_frame, text="Update Plot", command=self.update_plot)
        self.update_button.pack()

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect('resize_event', self.on_resize)
        self.plot_eigenvalues()

    def plot_eigenvalues(self, search_terms=None):
        self.ax.clear()
        norm = plt.Normalize(min(self.parameter_values), max(self.parameter_values))
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'red'])

        for eigs, pfs_abs, param in zip(self.eigenvalues_list, self.participation_factors_list, self.parameter_values):
            color = cmap(norm(param))
            if search_terms:
                bool_lists = []
                for terms in search_terms:
                    bool_list = [any(term in state for term in terms) for state in self.state_desc]
                    bool_lists.append(bool_list)

                combined_bool_list = np.logical_and.reduce(bool_lists)
                state_indices = [i for i, flag in enumerate(combined_bool_list) if flag]
                opacities = np.max(pfs_abs[state_indices, :], axis=0)
            else:
                opacities = np.ones(len(eigs))

            self.ax.scatter(eigs.real, eigs.imag, color=color, alpha=opacities)

        self.ax.set_xlabel('Real Part')
        self.ax.set_ylabel('Imaginary Part')
        self.ax.set_title('Eigenvalues with Participation Factors')

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def update_plot(self):
        search_terms = []
        for entry in self.search_entries:
            search_term = entry[0].get()
            search_terms.append([term.strip() for term in search_term.split(',')])

        self.plot_eigenvalues(search_terms)

    def add_search_field(self):
        frame = tk.Frame(self.search_frame)
        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT)
        remove_button = tk.Button(frame, text="X", command=lambda: self.remove_search_field(frame))
        remove_button.pack(side=tk.LEFT)
        frame.pack()
        self.search_entries.append((entry, frame))

    def remove_search_field(self, frame):
        for entry, frm in self.search_entries:
            if frm == frame:
                self.search_entries.remove((entry, frm))
                frame.destroy()
                break

    def on_resize(self, event):
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def run(self):
        self.root.mainloop()


def main():
    import casestudies.ps_data.ma_4bus as model_data

    eigenvalues_list = []
    participation_factors_list = []

    # Define the parameter values to sweep
    parameter_values = np.linspace(3,10,10)  # Example parameter values

    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)

    for param in parameter_values:
        # Set the parameter value in the power system model
        ps.gen['GEN'].par['H'] = param

        # Linearize
        ps_lin.linearize()
        ps_lin.eigenvalue_decomposition()

        eigenvalues_list.append(ps_lin.eigs)
        pfs_abs = np.abs(ps_lin.lev.T * ps_lin.rev) / np.max(np.abs(ps_lin.lev.T * ps_lin.rev), axis=0)
        participation_factors_list.append(pfs_abs)

    plotter = EigenSweep(eigenvalues_list, participation_factors_list, ps.state_desc, parameter_values)
    plotter.run()


if __name__ == '__main__':
    main()