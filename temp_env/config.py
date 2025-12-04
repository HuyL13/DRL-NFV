import numpy as np

VNF_TYPES = ['NAT', 'FW', 'VOC', 'TM', 'WO', 'IDPS']

VNF_SPECS = {
    'NAT': {'cpu': 2, 'ram': 4, 'storage': 10, 'process_time': 5},
    'FW': {'cpu': 3, 'ram': 6, 'storage': 15, 'process_time': 8},
    'VOC': {'cpu': 5, 'ram': 8, 'storage': 20, 'process_time': 12},
    'TM': {'cpu': 2, 'ram': 3, 'storage': 8, 'process_time': 4},
    'WO': {'cpu': 4, 'ram': 7, 'storage': 18, 'process_time': 10},
    'IDPS': {'cpu': 6, 'ram': 10, 'storage': 25, 'process_time': 15}
}

SFC_SPECS = {
    'CG': {'chain': ['NAT', 'FW', 'VOC', 'WO', 'IDPS'], 'bw': 4, 'delay': 80, 'bundle': (40, 55)},
    'AR': {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 100, 'delay': 10, 'bundle': (1, 4)},
    'VoIP': {'chain': ['NAT', 'FW', 'TM', 'FW', 'NAT'], 'bw': 0.064, 'delay': 100, 'bundle': (100, 200)},
    'VS': {'chain': ['NAT', 'FW', 'TM', 'VOC', 'IDPS'], 'bw': 4, 'delay': 100, 'bundle': (50, 100)},
    'MIoT': {'chain': ['NAT', 'FW', 'IDPS'], 'bw': 25, 'delay': 5, 'bundle': (10, 15)},
    'Ind4.0': {'chain': ['NAT', 'FW'], 'bw': 70, 'delay': 8, 'bundle': (1, 4)}
}

DC_CONFIG = {
    'storage': 2000,
    'cpu_range': (12, 120),
    'ram': 256,
    'link_bw': 1000
}

TRAINING_CONFIG = {
    'episodes': 1000,
    'batch_size': 64,
    'gamma': 0.99,
    'lr_dqn': 0.0001,
    'lr_vae': 0.001,
    'buffer_size': 100000,
    'target_update': 10,
    'vae_latent_dim': 32,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'P3_CONSTANT': 50.0,
}