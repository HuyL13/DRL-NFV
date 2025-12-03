import numpy as np
from config import DC_CONFIG, VNF_SPECS, VNF_TYPES

class DataCenter:
    def __init__(self, dc_id, cpu, storage=DC_CONFIG['storage'], ram=DC_CONFIG['ram']):
        self.id = dc_id
        self.max_cpu = cpu
        self.max_storage = storage
        self.max_ram = ram
        self.cpu = cpu
        self.storage = storage
        self.ram = ram
        self.installed_vnfs = {vnf: 0 for vnf in VNF_TYPES}
        self.allocated_vnfs = {}
 
        
    def can_install(self, vnf_type):
        spec = VNF_SPECS[vnf_type]
        return (self.storage >= spec['storage'] and 
                self.cpu >= spec['cpu'] and 
                self.ram >= spec['ram'])
    
    def install_vnf(self, vnf_type):
        if self.can_install(vnf_type):
            spec = VNF_SPECS[vnf_type]
            self.storage -= spec['storage']
            self.cpu -= spec['cpu']
            self.ram -= spec['ram']
            self.installed_vnfs[vnf_type] += 1
            return True
        return False
    
    def uninstall_vnf(self, vnf_type):
        if self.installed_vnfs[vnf_type] > 0:
            spec = VNF_SPECS[vnf_type]
            self.storage += spec['storage']
            self.cpu += spec['cpu']
            self.ram += spec['ram']
            self.installed_vnfs[vnf_type] -= 1
            return True
        return False
    
    def can_allocate(self, vnf_type, sfc_id):
        return self.installed_vnfs[vnf_type] > sum(1 for k, v in self.allocated_vnfs.items() 
                                                     if v == vnf_type)
    
    def allocate_vnf(self, vnf_type, sfc_id):
        if self.can_allocate(vnf_type, sfc_id):
            self.allocated_vnfs[sfc_id] = vnf_type
            return True
        return False
    
    def deallocate_vnf(self, sfc_id):
        if sfc_id in self.allocated_vnfs:
            del self.allocated_vnfs[sfc_id]
            return True
        return False
    
    def get_state(self):
        state = [
            self.cpu / self.max_cpu,
            self.storage / self.max_storage,
            self.ram / self.max_ram
        ]
        state.extend([self.installed_vnfs[vnf] / 10.0 for vnf in VNF_TYPES])
        state.append(len(self.allocated_vnfs) / 50.0)
        return np.array(state, dtype=np.float32)

class CoreNetwork:
    def __init__(self, num_dcs=4):
        self.num_dcs = num_dcs
        self.dcs = [DataCenter(i, np.random.randint(*DC_CONFIG['cpu_range'])) 
                    for i in range(num_dcs)]
        self.link_bw = np.full((num_dcs, num_dcs), DC_CONFIG['link_bw'], dtype=np.float32)
        self.link_used_bw = np.zeros((num_dcs, num_dcs), dtype=np.float32)
        self.sfc_placements = {}
        
    def get_available_bw(self, dc1, dc2):
        return self.link_bw[dc1][dc2] - self.link_used_bw[dc1][dc2]
    
    def allocate_bw(self, path, bw):
        for i in range(len(path) - 1):
            self.link_used_bw[path[i]][path[i+1]] += bw
            
    def deallocate_bw(self, path, bw):
        for i in range(len(path) - 1):
            self.link_used_bw[path[i]][path[i+1]] -= bw
            
    # calculate shortest path based on link_bw and link_used_bw
    def get_shortest_path(self, src, dst, bw):
        path=[]
        # dijktra algorithm
        visited = set()
        distances = {i: float('inf') for i in range(self.num_dcs)}
        previous_nodes = {i: None for i in range(self.num_dcs)}
        distances[src] = 0
        for _ in range(self.num_dcs):
            current_node = min((node for node in range(self.num_dcs) if node not in visited), 
                               key=lambda node: distances[node], default=None)
            if current_node is None or current_node == dst:
                break
            visited.add(current_node)
      
            for neighbor in range(self.num_dcs):
                if neighbor in visited:
                    continue
                available_bw = self.get_available_bw(current_node, neighbor)
                if available_bw >= bw:
                    alt_distance = distances[current_node] + 1
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous_nodes[neighbor] = current_node
        
        # reconstruct path
        current_node = dst
        while current_node is not None:
            path.insert(0, current_node)
            current_node = previous_nodes[current_node]
        if path[0] == src:
            return path
        return []
    
    def reset(self):
        self.dcs = [DataCenter(i, np.random.randint(*DC_CONFIG['cpu_range'])) 
                    for i in range(self.num_dcs)]
        self.link_used_bw = np.zeros((self.num_dcs, self.num_dcs), dtype=np.float32)
        self.sfc_placements = {}
        
    def get_network_state(self):
        dc_states = [dc.get_state() for dc in self.dcs]
        link_utilization = (self.link_used_bw / self.link_bw).flatten()
        return np.concatenate([np.concatenate(dc_states), link_utilization])