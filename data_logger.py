import json
import numpy as np

class DataLogger:

    def __init__(self, filename):
        self.filename = filename
        self.data = {
            'inputs': [],
            'outputs': [],
            'obstacles': []
        }
    
    def set_obstacles(self, obstacles):
        self.data['obstacles'] = obstacles


    def log_iteration(self, inputs, outputs):
        inputs = inputs.reshape(-1).tolist()
        outputs = outputs.reshape(-1).tolist()
        self.data['inputs'].append(inputs)
        self.data['outputs'].append(outputs)
        json.dump(self.data, open(self.filename, 'w'))


    def get_inputs(self, normalize):
        data = np.array(self.data['inputs'])
        if not normalize:
            return data

        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std


    def get_outputs(self, normalize):
        data = np.array(self.data['outputs'])
        if not normalize:
            return data

        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std


    def get_obstacles(self):
        return np.array(self.data['obstacles'])


    @staticmethod
    def load_data(filename):
        logger = DataLogger(filename)
        logger.data = json.load(open(filename))
        return logger
