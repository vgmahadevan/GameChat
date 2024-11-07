import json
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    # Denotes the total number of samples
    def __len__(self):
        return len(self.features)

    # Generates one sample of data
    def __getitem__(self, index):
        X = self.features[index]
        y = self.labels[index]

        return X, y


class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = {
            'agent_inputs': {},
            'agent_outputs': {},
            'obstacles': []
        }
    
    def set_obstacles(self, obstacles):
        self.data['obstacles'] = obstacles


    def log_iteration(self, agent_idx, ego_state, opp_state, ego_goal, controls):
        inputs = np.append(ego_state, opp_state)
        inputs = np.append(inputs, [ego_goal])
        inputs = inputs.reshape(-1).tolist()
        outputs = controls.reshape(-1).tolist()
        agent_idx = str(agent_idx)
        if agent_idx not in self.data['agent_inputs']:
            self.data['agent_inputs'][agent_idx] = []
            self.data['agent_outputs'][agent_idx] = []
        self.data['agent_inputs'][agent_idx].append(inputs)
        self.data['agent_outputs'][agent_idx].append(outputs)
        json.dump(self.data, open(self.filename, 'w'))


    def get_inputs(self, agent_idx, normalize):
        agent_idx = str(agent_idx)
        data = np.array(self.data['agent_inputs'][agent_idx])
        if not normalize:
            return data

        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std


    def get_outputs(self, agent_idx, normalize):
        agent_idx = str(agent_idx)
        data = np.array(self.data['agent_outputs'][agent_idx])
        if not normalize:
            return data

        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std


    def get_obstacles(self):
        return self.data['obstacles'].copy()


    @staticmethod
    def load_data(filename):
        logger = DataLogger(filename)
        logger.data = json.load(open(filename))
        return logger


class BlankLogger:
    def __init__(self):
        pass
    
    def set_obstacles(self, obstacles):
        pass

    def log_iteration(self, agent_idx, ego_state, opp_state, ego_goal, controls):
        pass
