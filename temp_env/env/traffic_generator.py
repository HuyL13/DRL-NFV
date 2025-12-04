import numpy as np
from config import SFC_SPECS

class SFCRequest:
    def __init__(self, sfc_id, sfc_type, src_dc, dst_dc):
        self.id = sfc_id
        self.type = sfc_type
        self.spec = SFC_SPECS[sfc_type]
        self.chain = self.spec['chain']
        self.bw = self.spec['bw']
        self.max_delay = self.spec['delay']
        self.src = src_dc
        self.dst = dst_dc
        self.current_vnf_idx = 0
        self.placement = []
        self.elapsed_time = 0
        self.active = True
        self.shortest_path=[]
        

        
    def get_current_vnf(self):
        if self.current_vnf_idx < len(self.chain):
            return self.chain[self.current_vnf_idx]
        return None
    
    def advance_vnf(self, dc_id, process_time):
        self.placement.append(dc_id)
        self.elapsed_time += process_time
        self.current_vnf_idx += 1
        
    def is_complete(self):
        return self.current_vnf_idx >= len(self.chain)
    
    def check_delay_violation(self):
        return self.elapsed_time > self.max_delay
    
    def get_remaining_delay(self):
        return max(0, self.max_delay - self.elapsed_time)
    
    def get_state(self):
        state = [
            self.current_vnf_idx / len(self.chain),
            self.elapsed_time / self.max_delay,
            self.bw / 100.0,
            self.src / 10.0,
            self.dst / 10.0
        ]
        vnf_encoding = [0] * 6
        if self.current_vnf_idx < len(self.chain):
            from config import VNF_TYPES
            vnf_encoding[VNF_TYPES.index(self.chain[self.current_vnf_idx])] = 1
        state.extend(vnf_encoding)
        return np.array(state, dtype=np.float32)

class TrafficGenerator:
    def __init__(self, num_dcs):
        self.num_dcs = num_dcs
        self.sfc_counter = 0
        self.active_sfcs = []
        
    def generate_bundle(self, request_count=1):
        new_sfcs = []
        for _ in range(request_count):
            for sfc_type, spec in SFC_SPECS.items():
                bundle_size = np.random.randint(*spec['bundle'])
                for _ in range(bundle_size):
                    src = np.random.randint(0, self.num_dcs)
                    dst = np.random.randint(0, self.num_dcs)
                    while dst == src:
                        dst = np.random.randint(0, self.num_dcs)
                    sfc = SFCRequest(self.sfc_counter, sfc_type, src, dst)
                    new_sfcs.append(sfc)
                    self.sfc_counter += 1
        self.active_sfcs.extend(new_sfcs)
        return new_sfcs
    
    def remove_completed(self):
        self.active_sfcs = [sfc for sfc in self.active_sfcs if sfc.active]
        
    def get_active_count(self):
        return len(self.active_sfcs)