import config
import numpy as np

def run_simulation(scenario, env, controllers, logger, plotter):
    x_cum = [[], []]
    u_cum = [[], []]

    controllers[0].initialize_controller(env)
    controllers[1].initialize_controller(env)

    for sim_iteration in range(config.sim_steps):
        print(f"\nIteration: {sim_iteration}")
        for agent_idx in range(config.n):
            x_cum[agent_idx].append(env.initial_states[agent_idx])

        new_states, outputted_controls = env.run_simulation(sim_iteration, controllers, logger)

        for agent_idx in range(config.n):
            u_cum[agent_idx].append(outputted_controls[agent_idx])

        # Plots
        if sim_iteration % config.plot_rate == 0 and config.plot_live:
            plotter.plot_live(scenario, controllers, x_cum, u_cum)

    # Discard the first element of both x1 and x2
    x_cum = np.array(x_cum)
    u_cum = np.array(u_cum)
    if config.plot_end:
        plotter.plot(scenario, controllers, x_cum, u_cum)

