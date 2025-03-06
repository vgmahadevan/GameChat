"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
from config import Role
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpc_cbf import MPC
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from environment import Environment
from LLM_utils import LLMAgent

def run_one_experiment(scenario_type, control_type, agent_1_offset, agent_2_offset, priority_1, priority_2):
    x_cum = [[], []]
    u_cum = [[], []]

    # Scenarios: "doorway" or "intersection"
    if scenario_type == 0:
        scenario = DoorwayScenario(agent_1_offset, agent_2_offset) # scenario 0
    else:
        scenario = IntersectionScenario(agent_1_offset, agent_2_offset) # scenario 1

    # Matplotlib plotting handler
    plotter = Plotter()
    # logger = DataLogger('doorway_train_data_no_liveness2.json')
    logger = BlankLogger()

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    goals = scenario.goals.copy()
    original_start = scenario.initial.copy()
    logger.set_obstacles(scenario.obstacles.copy())
    env = Environment(scenario.initial.copy())
    agents = []

    # Setup agent 0
    agents.append(MPC(agent_idx=0, goal=goals[0,:], static_obs=scenario.obstacles.copy()))
    # agents.append(ModelController("model_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
    agents[-1].initialize_controller(env)

    # Setup agent 1
    agents.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy()))
    # agents.append(ModelController("model_fc_definition.json", static_obs=scenario.obstacles.copy()))
    # agents.append(ModelController("model_1_bn_definition.json", static_obs=scenario.obstacles.copy()))
    agents[-1].initialize_controller(env)
    #agents[-1].role = Role.FOLLOWER
    #agents[-1].role = Role.LEADER

    if control_type == 0:
        config.liveliness = False
    else:
        config.liveliness = True

    if control_type == 1:
        mpccbf = True
    else:
        mpccbf = False

    if control_type == 2:
        pass

    if control_type == 3:
        if priority_1 < priority_2:
            agents[0].role = Role.LEADER
            agents[1].role = Role.FOLLOWER
        elif priority_1 > priority_2:
            agents[0].role = Role.FOLLOWER
            agents[1].role = Role.LEADER

    # Initialize LLM agents
    time_duration = np.nan
    if control_type == 4 or control_type == 5:
        agents[0].llm = LLMAgent(priority_1)
        agents[1].llm = LLMAgent(priority_2)

        time_start = time.time()
        agent_2_output = "Begin the conversation"
        for _ in range(4):
            agent_1_output = agents[0].llm.query("user", "Other agent says: " + agent_2_output)
            print(agent_1_output)
            if ("1" in agent_1_output) and ("1" in agent_2_output):
                break

            agent_2_output = agents[1].llm.query("user", "Other agent says: " + agent_1_output)
            print(agent_2_output)
            if ("1" in agent_1_output) and ("1" in agent_2_output):
                break
        time_duration = time.time() - time_start

        print(f"Time taken for LLM: {time_duration}")
        
        llm_roles = [agents[0].llm.get_role(), agents[1].llm.get_role()]
        if control_type == 4:
            agents[0].role = agents[0].llm.get_role()
            agents[1].role = agents[1].llm.get_role()
        print(agents[0].role)
        print(agents[1].role)

    time_to_door = [None, None]
    time_to_goal = [None, None]
    to_ret = {
        "collision": 0,
        "deadlock": 0,
        "correct_priority": 0,
        "higher_priority_ttg": np.nan,
        "makespan": np.nan,
        "second_min_vel": np.nan,
        "path_deviation": np.nan,
        "llm_time": time_duration
    }
    for sim_iteration in range(config.sim_steps):
        #print(f"\nIteration: {sim_iteration}")
        for agent_idx in range(config.n):
            x_cum[agent_idx].append(env.initial_states[agent_idx])

        if (control_type == 4 or control_type == 5) and time_duration < (sim_iteration + 1) * config.Ts:
            agents[0].role = llm_roles[0]
            agents[1].role = llm_roles[1]

        new_states, outputted_controls = env.run_simulation(sim_iteration, agents, logger, mpccbf)

        for agent_idx in range(config.n):
            u_cum[agent_idx].append(outputted_controls[agent_idx])

        # Check if agents have reached door
        if scenario_type == 0:
            if time_to_door[0] is None and new_states[0][0] > 1:
                time_to_door[0] = (sim_iteration + 1) * config.Ts
            if time_to_door[1] is None and new_states[1][0] > 1:
                time_to_door[1] = (sim_iteration + 1) * config.Ts
        elif scenario_type == 1:
            if time_to_door[0] is None and new_states[0][1] > 0:
                time_to_door[0] = (sim_iteration + 1) * config.Ts
            if time_to_door[1] is None and new_states[1][0] > 0:
                time_to_door[1] = (sim_iteration + 1) * config.Ts

        # See if agents have reached goal
        if time_to_goal[0] is None and np.linalg.norm(new_states[0, :2] - goals[0, :2]) < 0.1:
            time_to_goal[0] = (sim_iteration + 1) * config.Ts
        if time_to_goal[1] is None and np.linalg.norm(new_states[1, :2] - goals[1, :2]) < 0.1:
            time_to_goal[1] = (sim_iteration + 1) * config.Ts
        #print(time_to_goal)

        # Both agents at goal
        if time_to_goal[0] is not None and time_to_goal[1] is not None:
            # determine if the agents have the correct priority
            print(time_to_door)
            print(time_to_goal)
            print(priority_1, priority_2)
            if time_to_door[0] < time_to_door[1] and priority_1 < priority_2:
                to_ret["correct_priority"] = 1
            elif time_to_door[0] > time_to_door[1] and priority_1 > priority_2:
                to_ret["correct_priority"] = 1
            else:
                to_ret["correct_priority"] = 0

            # determine higher priority agent's time to goal
            if priority_1 < priority_2:
                to_ret["higher_priority_ttg"] = time_to_goal[0]
            else:
                to_ret["higher_priority_ttg"] = time_to_goal[1]

            # determine the makespan
            to_ret["makespan"] = max(time_to_goal)

            # determine the slowest velocity of the slower agent
            slower = 0 if time_to_goal[0] > time_to_goal[1] else 1
            to_ret["second_min_vel"] = float('inf')
            for idx in range(len(u_cum[slower])):
                if (idx+1)*config.Ts > time_to_door[slower]:
                    break
                to_ret["second_min_vel"] = min(to_ret["second_min_vel"], u_cum[slower][idx][0])

            # determine the path deviation
            deviations = []
            for agent_idx in range(config.n):
                start_pos = original_start[agent_idx, :2]
                goal_pos = goals[agent_idx, :2]
                line_vec = goal_pos - start_pos
                line_len = np.linalg.norm(line_vec)
                line_unit_vec = line_vec / line_len

                
                for pos in x_cum[agent_idx]:
                    vec_to_pos = pos[:2] - start_pos
                    projection_length = np.dot(vec_to_pos, line_unit_vec)
                    projection_point = start_pos + projection_length * line_unit_vec
                    deviation = np.linalg.norm(pos[:2] - projection_point)
                    deviations.append(deviation)

            avg_deviation = np.mean(deviations)
            to_ret["path_deviation"] = avg_deviation

            break

        # Check for collision
        if np.linalg.norm(new_states[0, :2] - new_states[1, :2]) < 0.15:
            to_ret["collision"] = 1
            break
        
        # Check for deadlock
        if sim_iteration > 0 and u_cum[0][-1][0] < 0.01 and u_cum[1][-1][0] < 0.01:
            to_ret["deadlock"] = 1
            break

        # Plots
        if sim_iteration % config.plot_rate == 0 and config.plot_live:
            plotter.plot_live(scenario, agents, x_cum, u_cum, scenario_type)
            pass

# Discard the first element of both x1 and x2
    # x_cum = np.array(x_cum)
    # u_cum = np.array(u_cum)
    # plotter.plot(scenario, agents, x_cum, u_cum)
    if sim_iteration % config.plot_rate == 0 and config.plot_live:
        plotter.plot_live(scenario, agents, x_cum, u_cum, scenario_type)
        pass
    plt.close()

    return to_ret

# print(run_one_experiment(0, 1, -.2, 0, 0, 1))

# scenarios: 0 is Doorway, 1 is Intersection
# control types: 
#   0 is MPCCBF
#   1 is SMGCBF
#   2 is GameChat no comm
#   3 is GameChat with priorities known
#   4 is GameChat pre SMG convo
#   5 is GameChat during SMG convo
if __name__ == "__main__":
    records = []
    counter = 0
    # for control_type in range(1, -1, -1):
    #     for scenario_type in range(1, 0, -1):
    #         for (off1, off2) in [(0, 0), (-.2, 0), (0, -.2)]:
    #             for (p1, p2) in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
    #                 print(counter)

    #                 record = run_one_experiment(scenario_type, control_type, off1, off2, p1, p2)
    #                 record["scenario_type"] = scenario_type
    #                 record["control_type"] = control_type
    #                 record["offset1"] = off1
    #                 record["offset2"] = off2
    #                 record["priority1"] = p1
    #                 record["priority2"] = p2
    #                 records.append(record)
    #                 print(record)
    #                 counter += 1

    # control_type = 2
    # for scenario_type in range(1, 0, -1):
    #     for (off1, off2) in [(0, 0), (-.25, 0), (0, -.25)]:
    #         for (p1, p2) in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
    #             print(counter)

    #             record = run_one_experiment(scenario_type, control_type, off1, off2, p1, p2)
    #             record["scenario_type"] = scenario_type
    #             record["control_type"] = 5
    #             record["offset1"] = off1
    #             record["offset2"] = off2
    #             record["priority1"] = p1
    #             record["priority2"] = p2
    #             records.append(record)
    #             print(record)
    #             counter += 1

    control_type = 5
    for scenario_type in range(0, 2, 1):
        (off1, off2) = (0, 0)
        (p1, p2) = (2, 0)
        run_one_experiment(scenario_type, control_type, off1, off2, p1, p2)


    # df = pd.DataFrame(records)
    # df.to_csv("llmresults.csv")

    # SMGConvo doorway sym: 2.86
    # SMGConvo intersection sym: 2.48
    # SMGConvo doorway asym: 2.52
    # SMGConvo intersection asym: 2.61