from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import src.dynamic as dps
import src.solvers as dps_sol
import importlib
importlib.reload(dps)


if __name__ == '__main__':

    # region Model loading and initialisation stage
    import casestudies.ps_data.ma_4bus as model_data
    model = model_data.load()
    ps = dps.PowerSystemModel(model=model)  # Load into a PowerSystemModel object

    ps.power_flow()  # Power flow calculation
    # Print load flow solution
    for bus, v in zip(ps.buses['name'], ps.v_0):
        print(f'{bus}: {np.abs(v):.2f} /_ {np.angle(v):.2f}')

    ps.init_dyn_sim()  # Initialise dynamic variables
    x0 = ps.x0.copy()  # Initial states

    # List of machine parameters for easy access
    gen_pars = ps.gen['GEN'].par # Access like this: S_n_gen = genpars['S_n']

    t = 0
    t_end = 15  # Simulation time

    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x0, t_end, max_step=5e-3)  # solver
    # endregion

    # region Runtime variables
    result_dict = defaultdict(list)
    # Additional plot variables
    P_m_stored = []
    P_e_stored = []
    E_f_stored = []
    v_bus = []
    I_stored = []
    modal_stored = []

    event_flag1 = True
    # endregion

    # Simulation loop starts here!
    while t < t_end:
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        if 1 < t < 1.05:
            ps.y_bus_red_mod[3,3] = 1e6
        else:
            ps.y_bus_red_mod[3,3] = 0

        # region Store variables
        result_dict['Global', 't'].append(sol.t)
        [result_dict[tuple(desc)].append(state) for desc, state in zip(ps.state_desc, x)]
        # Store additional variables
        P_m_stored.append(ps.gen['GEN'].P_m(x, v).copy())
        P_e_stored.append(ps.gen['GEN'].P_e(x, v).copy())
        E_f_stored.append(ps.gen['GEN'].E_f(x, v).copy())
        I_gen = ps.y_bus_red_full[0, 1] * (v[0] - v[1])
        I_stored.append(np.abs(I_gen))
        # endregion

    # Convert dict to pandas dataframe
    result = pd.DataFrame(result_dict, columns=pd.MultiIndex.from_tuples(result_dict))

    # region Plotting
    fig, ax = plt.subplots(3, sharex=True)
    fig.suptitle('Generator speed, power angle and electric power')
    ax[0].plot(result[('Global', 't')], result.xs(key='speed', axis='columns', level=1), label=gen_pars['name'])
    ax[0].legend()
    ax[0].set_ylabel('Speed (p.u.)')
    ax[1].plot(result[('Global', 't')], result.xs(key='angle', axis='columns', level=1))
    ax[1].set_ylabel('Rotor angle (rad)')
    ax[2].plot(result[('Global', 't')], np.array(P_e_stored)/gen_pars['S_n'])
    ax[2].set_ylabel('Elec. power (p.u.)')
    ax[2].set_xlabel('time (s)')
    plt.show()
    # endregion
