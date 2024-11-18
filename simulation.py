import config
import numpy as np
from util import calculate_all_metrics

def run_simulation(scenario, env, controllers, logger, plotter):
    x_cum = [[], []]
    u_cum = [[], []]
    metrics = []

    controllers[0].initialize_controller(env)
    controllers[1].initialize_controller(env)

    for sim_iteration in range(config.sim_steps):
        print(f"\nIteration: {sim_iteration}")
        for agent_idx in range(config.n):
            x_cum[agent_idx].append(env.initial_states[agent_idx])

        metrics.append(calculate_all_metrics(x_cum[0][-1], x_cum[1][-1]))
        new_states, outputted_controls = env.run_simulation(sim_iteration, controllers, logger)

        for agent_idx in range(config.n):
            u_cum[agent_idx].append(outputted_controls[agent_idx])

        # Plots
        if sim_iteration % config.plot_rate == 0 and config.plot_live and plotter is not None:
            plotter.plot_live(sim_iteration, scenario, x_cum, u_cum, metrics)

    # Discard the first element of both x1 and x2
    x_cum = np.array(x_cum)
    u_cum = np.array(u_cum)
    if config.plot_end and plotter is not None:
        plotter.plot(scenario, x_cum, u_cum, metrics)
