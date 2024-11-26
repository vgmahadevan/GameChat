import config
from data_logger import DataGenerator

generator = DataGenerator(['intersection_scenario_suite2/s_intersection_0.8_0.8_l_0_faster_off0.json'], config.x_is_d_goal, config.add_liveness_as_input, config.fixed_liveness_input, config.n_opponents, config.static_obs_xy_only, config.ego_frame_inputs)
inputs = generator.get_inputs([0], False)[0]

for iteration in range(len(inputs)):
    print("\nIteration", iteration)
    print(inputs[iteration])
