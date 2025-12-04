"""
Pretrain DQN for SFC Provisioning - Following Algorithm 1 from the paper:
"Unlocking Reconfigurability for Deep Reinforcement Learning in SFC Provisioning"

This implementation follows exactly Algorithm 1 and the training procedure described in the paper:
- 350 updates (U)
- 20 episodes per update (E)
- 100 actions per step (A)
- Step duration T = 1ms
- Action inference time = 0.01ms
- SFC generation every N = 4 steps

Key components from Algorithm 1:
- DC Priority based on resources, E2E delay, and path availability
- VNF Priority using P1 (remaining time), P2 (SFC-based), P3 (urgency)
- Proper Allocation action that selects highest priority VNF
"""

import numpy as np
import sys
import os

sys.path.append('.')

from env.core_network import CoreNetwork, DataCenter
from env.traffic_generator import TrafficGenerator, SFCRequest
from models.dqn_model import DQNModel
from utils.buffers import ReplayBuffer
from config import VNF_TYPES, VNF_SPECS, SFC_SPECS, TRAINING_CONFIG, DC_CONFIG

# ============================================================================
# Training Hyperparameters from Paper
# ============================================================================
NUM_UPDATES = 350              # U = 350 updates
EPISODES_PER_UPDATE = 20       # E = 20 episodes per update
ACTIONS_PER_STEP = 100         # A = 100 actions per step (as per paper)
STEP_DURATION = 1              # T = 1 ms
ACTION_INFERENCE_TIME = 0.01   # 0.01 ms per action inference
SFC_GENERATION_INTERVAL = 4    # N = 4 steps between SFC generation
P3_CONSTANT = TRAINING_CONFIG.get('P3_CONSTANT', 50.0)  # Constant C for P3 calculation
P3_URGENCY_THRESHOLD = 0.3     # Threshold Thr for urgency (30% of E2E delay remaining)

# ============================================================================
# Reward values from Paper (Section III-A-3)
# ============================================================================
REWARD_SFC_SATISFIED = 2.0
REWARD_SFC_DROPPED = -1.5
REWARD_INVALID_ACTION = -1.0
REWARD_UNINSTALL_REQUIRED_VNF = -0.5


class SFCProvisioningEnv:
    """
    Environment for SFC Provisioning following the paper's system model and Algorithm 1.
    
    Key differences from simplified version:
    1. Processes ALL pending SFCs, not just one
    2. DC priority based on resources + E2E delay + path
    3. VNF priority using P1 + P2 + P3 formula
    4. Generates new SFCs every N=4 steps
    """
    
    def __init__(self, num_dcs=4):
        self.num_dcs = num_dcs
        self.network = CoreNetwork(num_dcs)
        self.traffic_gen = TrafficGenerator(num_dcs)
        
        self.step_count = 0
        self.current_time = 0  # Current simulation time in ms
        
        # State dimensions from paper
        self.num_vnf_types = len(VNF_TYPES)  # |V| = 6
        self.num_sfc_types = len(SFC_SPECS)   # |S| = 6
        
        # State 1: [1 × (2*|V| + 2)] - Current DC info
        self.state1_dim = 2 * self.num_vnf_types + 2
        
        # State 2: [|S| × (1 + 2*|V|)] - SFC processing by current DC
        self.state2_dim = self.num_sfc_types * (1 + 2 * self.num_vnf_types)
        
        # State 3: [|S| × (4 + |V|)] - Overall pending SFC info
        self.state3_dim = self.num_sfc_types * (4 + self.num_vnf_types)
        
        # Action space: 2*|V| + 1 (Install + Uninstall + Wait)
        self.num_actions = 2 * self.num_vnf_types + 1
        
        # Track statistics
        self.sfcs_satisfied = 0
        self.sfcs_dropped = 0
        self.sfcs_total = 0
        
    def reset(self):
        """Reset environment for new episode."""
        self.network.reset()
        self.traffic_gen = TrafficGenerator(self.num_dcs)
        self.step_count = 0
        self.current_time = 0
        self.sfcs_satisfied = 0
        self.sfcs_dropped = 0
        self.sfcs_total = 0
        
        # Generate initial SFC requests (request_count=1 means one bundle)
        new_sfcs = self.traffic_gen.generate_bundle(request_count=1)
        self.sfcs_total += len(new_sfcs)
        
        # Assign shortest paths to all SFCs
        for sfc in self.traffic_gen.active_sfcs:
            sfc.shortest_path = self.network.get_shortest_path(sfc.src, sfc.dst, sfc.bw)
        
        return self._get_state(0)
    
    def _get_state1(self, dc_id):
        """
        Get State 1: Current DC information (Algorithm 1, Step 3)
        [1 × (2*|V| + 2)] = installed VNFs + available VNFs + storage + CPU
        """
        dc = self.network.dcs[dc_id]
        state = []
        
        # Number of installed VNFs for each type (normalized)
        for vnf_type in VNF_TYPES:
            state.append(dc.installed_vnfs[vnf_type] / 10.0)
        
        # Number of available (idle) VNFs for each type
        for vnf_type in VNF_TYPES:
            installed = dc.installed_vnfs[vnf_type]
            allocated = sum(1 for v in dc.allocated_vnfs.values() if v == vnf_type)
            available = max(0, installed - allocated)
            state.append(available / 10.0)
        
        # Available storage and computational power (normalized)
        state.append(dc.storage / dc.max_storage)
        state.append(dc.cpu / dc.max_cpu)
        
        return np.array(state, dtype=np.float32)
    
    def _get_state2(self, dc_id):
        """
        Get State 2: SFC processing stages by current DC (Algorithm 1, Step 4)
        [|S| × (1 + 2*|V|)] - For each SFC type: type encoding + allocated VNFs + remaining VNFs
        """
        sfc_types = list(SFC_SPECS.keys())
        state = []
        
        for sfc_type in sfc_types:
            # Find SFCs of this type that have VNFs allocated at this DC
            sfcs_at_dc = [sfc for sfc in self.traffic_gen.active_sfcs 
                         if sfc.type == sfc_type and sfc.active and dc_id in sfc.placement]
            
            if sfcs_at_dc:
                # Use the first one (could aggregate, but paper uses per-type info)
                sfc = sfcs_at_dc[0]
                
                # SFC type encoding
                state.append(sfc_types.index(sfc_type) / len(sfc_types))
                
                # Already allocated VNFs (one-hot for each VNF type)
                allocated_vnfs = [0] * self.num_vnf_types
                for vnf_idx in range(sfc.current_vnf_idx):
                    vnf_type = sfc.chain[vnf_idx]
                    allocated_vnfs[VNF_TYPES.index(vnf_type)] = 1
                state.extend(allocated_vnfs)
                
                # Remaining VNFs waiting for allocation
                remaining_vnfs = [0] * self.num_vnf_types
                for vnf_idx in range(sfc.current_vnf_idx, len(sfc.chain)):
                    vnf_type = sfc.chain[vnf_idx]
                    remaining_vnfs[VNF_TYPES.index(vnf_type)] = 1
                state.extend(remaining_vnfs)
            else:
                # No SFCs of this type at this DC
                state.append(sfc_types.index(sfc_type) / len(sfc_types))
                state.extend([0] * self.num_vnf_types)  # No allocated VNFs
                state.extend([0] * self.num_vnf_types)  # No remaining VNFs
        
        return np.array(state, dtype=np.float32)
    
    def _get_state3(self):
        """
        Get State 3: Overall pending SFC requests information (Algorithm 1, Step 5)
        [|S| × (4 + |V|)] - For each SFC type: type + count + remaining delay + BW + waiting VNFs
        """
        sfc_types = list(SFC_SPECS.keys())
        state = []
        
        for sfc_type in sfc_types:
            # Get all active SFCs of this type
            sfcs_of_type = [sfc for sfc in self.traffic_gen.active_sfcs 
                          if sfc.type == sfc_type and sfc.active]
            
            if sfcs_of_type:
                # SFC type encoding
                state.append(sfc_types.index(sfc_type) / len(sfc_types))
                
                # Request count (normalized)
                state.append(len(sfcs_of_type) / 100.0)
                
                # Remaining minimum E2E latency time (normalized)
                min_remaining = min(sfc.get_remaining_delay() for sfc in sfcs_of_type)
                state.append(min_remaining / 100.0)
                
                # BW requirement (normalized)
                state.append(SFC_SPECS[sfc_type]['bw'] / 100.0)
                
                # Total VNFs waiting for allocation for each VNF type
                waiting_vnfs = [0] * self.num_vnf_types
                for sfc in sfcs_of_type:
                    current_vnf = sfc.get_current_vnf()
                    if current_vnf:
                        waiting_vnfs[VNF_TYPES.index(current_vnf)] += 1
                # Normalize
                waiting_vnfs = [v / 50.0 for v in waiting_vnfs]
                state.extend(waiting_vnfs)
            else:
                # No pending SFCs of this type
                state.append(sfc_types.index(sfc_type) / len(sfc_types))
                state.append(0)  # Count
                state.append(0)  # Remaining delay
                state.append(0)  # BW
                state.extend([0] * self.num_vnf_types)  # No waiting VNFs
        
        return np.array(state, dtype=np.float32)
    
    def _get_state(self, dc_id):
        """Get all three input states for the DRL model."""
        state1 = self._get_state1(dc_id)
        state2 = self._get_state2(dc_id)
        state3 = self._get_state3()
        return state1, state2, state3
    
    def set_dc_priority(self):
        """
        Set DC priority based on Algorithm 1 description:
        
        "DCs' iteration order is defined by their priority points which depend on 
        the available resources and incoming SFC requests' E2E delay, and it picks 
        the source DC with minimum E2E delay SFC requests to the highest priority. 
        The DCs on the shortest available path based on BW resource availability 
        between source and destination of that SFC request are provided with 
        decreasing priority values."
        
        Returns: List of DC IDs ordered by priority (highest first)
        """
        if not self.traffic_gen.active_sfcs:
            return list(range(self.num_dcs))
        
        # Find the SFC with minimum remaining E2E delay
        active_sfcs = [sfc for sfc in self.traffic_gen.active_sfcs if sfc.active]
        if not active_sfcs:
            return list(range(self.num_dcs))
        
        # Sort SFCs by remaining delay (most urgent first)
        active_sfcs.sort(key=lambda sfc: sfc.get_remaining_delay())
        most_urgent_sfc = active_sfcs[0]
        
        # Get shortest path for the most urgent SFC
        path = most_urgent_sfc.shortest_path
        if not path:
            path = self.network.get_shortest_path(
                most_urgent_sfc.src, 
                most_urgent_sfc.dst, 
                most_urgent_sfc.bw
            )
            most_urgent_sfc.shortest_path = path
        
        # Build priority list
        priority_list = []
        dc_scores = {}
        
        for dc_id in range(self.num_dcs):
            dc = self.network.dcs[dc_id]
            score = 0
            
            # Higher score for DCs in the path (decreasing by position)
            if path and dc_id in path:
                position = path.index(dc_id)
                score += (len(path) - position) * 100  # Path priority
            
            # Higher score for DCs with more available resources
            score += (dc.cpu / dc.max_cpu) * 50  # CPU availability
            score += (dc.storage / dc.max_storage) * 30  # Storage availability
            
            # Higher score if DC is source of urgent SFC
            if dc_id == most_urgent_sfc.src:
                score += 200  # Source DC gets highest priority
            
            # Bonus for DCs that already have required VNF installed
            current_vnf = most_urgent_sfc.get_current_vnf()
            if current_vnf and dc.installed_vnfs.get(current_vnf, 0) > 0:
                score += 50
            
            dc_scores[dc_id] = score
        
        # Sort by score (highest first)
        priority_list = sorted(dc_scores.keys(), key=lambda x: dc_scores[x], reverse=True)
        
        return priority_list
    
    def calculate_vnf_priority(self, vnf_type, dc_id):
        """
        Calculate VNF priority using P1 + P2 + P3 formula from Algorithm 1 (Steps 22-29).
        
        Returns: List of (sfc, priority) tuples sorted by priority (highest first)
        """
        vnf_priority_list = []
        
        for sfc in self.traffic_gen.active_sfcs:
            if not sfc.active:
                continue
            
            # Check if this SFC needs this VNF type next
            current_vnf = sfc.get_current_vnf()
            if current_vnf != vnf_type:
                continue
            
            # Calculate P1: Remaining time priority (higher when less time remaining)
            # P1 = TE_s - D_s (time elapsed - E2E delay tolerance)
            # Higher value = more urgent
            remaining_delay = sfc.get_remaining_delay()
            max_delay = sfc.max_delay
            p1 = (max_delay - remaining_delay) / max(max_delay, 1)  # Normalized
            
            # Calculate P2: SFC-based priority
            # Higher if previous VNFs in chain are allocated to this DC
            # Lower if previous VNFs are allocated to other DCs
            p2 = 0
            for vnf_idx in range(sfc.current_vnf_idx):
                if vnf_idx < len(sfc.placement):
                    if sfc.placement[vnf_idx] == dc_id:
                        p2 += 1  # Same DC bonus
                    else:
                        p2 -= 0.5  # Different DC penalty
            
            # Normalize P2
            if sfc.current_vnf_idx > 0:
                p2 = p2 / sfc.current_vnf_idx
            
            # Calculate P3: Urgency priority
            # If remaining time < threshold, increase priority significantly
            # P3 = C / (D_s - TE_s + epsilon)
            epsilon = 0.001
            urgency_threshold = P3_URGENCY_THRESHOLD * max_delay
            if remaining_delay < urgency_threshold:
                p3 = P3_CONSTANT / (remaining_delay + epsilon)
            else:
                p3 = 0
            
            # Normalize P3
            p3 = min(p3 / P3_CONSTANT, 1.0)
            
            # Total priority
            priority = p1 + p2 + p3
            
            # Bonus if this DC is on the SFC's shortest path
            if sfc.shortest_path and dc_id in sfc.shortest_path:
                priority += 0.5
            
            vnf_priority_list.append((sfc, priority))
        
        # Sort by priority (highest first)
        vnf_priority_list.sort(key=lambda x: x[1], reverse=True)
        
        return vnf_priority_list
    
    def step(self, dc_id, action):
        """
        Execute action following Algorithm 1 (Steps 7-32).
        
        Action types:
        - 0 to |V|-1: Allocation of VNF type (Install if needed + Allocate to highest priority SFC)
        - |V| to 2*|V|-1: Uninstall VNF type
        - 2*|V|: Wait
        """
        reward = 0
        dc = self.network.dcs[dc_id]
        
        # Parse action type (Algorithm 1, Step 7)
        if action < self.num_vnf_types:
            # ALLOCATION action (Algorithm 1, Steps 17-32)
            vnf_type = VNF_TYPES[action]
            
            # Get VNFs of this type waiting for allocation with priorities
            vnf_priority_list = self.calculate_vnf_priority(vnf_type, dc_id)
            
            if not vnf_priority_list:
                # No VNFs waiting to be allocated with this type - invalid action
                reward = REWARD_INVALID_ACTION
            else:
                # Check if we can install/allocate this VNF type
                # First, check if there's an available VNF instance
                installed = dc.installed_vnfs.get(vnf_type, 0)
                allocated_count = sum(1 for v in dc.allocated_vnfs.values() if v == vnf_type)
                available = installed - allocated_count
                
                if available <= 0:
                    # Need to install new VNF instance
                    if dc.can_install(vnf_type):
                        dc.install_vnf(vnf_type)
                        available = 1
                    else:
                        # Cannot install - no resources - invalid action
                        reward = REWARD_INVALID_ACTION
                        return self._get_state(dc_id), reward, False
                
                # Select highest priority VNF to allocate (Algorithm 1, Step 30)
                selected_sfc, priority = vnf_priority_list[0]
                
                # Perform allocation (Algorithm 1, Step 31)
                if dc.can_allocate(vnf_type, selected_sfc.id):
                    dc.allocate_vnf(vnf_type, selected_sfc.id)
                    
                    # Advance the SFC
                    process_time = VNF_SPECS[vnf_type]['process_time']
                    selected_sfc.advance_vnf(dc_id, process_time)
                    
                    # Check if SFC is complete
                    if selected_sfc.is_complete():
                        if not selected_sfc.check_delay_violation():
                            reward = REWARD_SFC_SATISFIED
                            self.sfcs_satisfied += 1
                        else:
                            reward = REWARD_SFC_DROPPED
                            self.sfcs_dropped += 1
                        selected_sfc.active = False
                        # Deallocate VNFs for this SFC
                        self._deallocate_sfc(selected_sfc)
                    else:
                        # Partial reward for successful allocation
                        reward = 0.1
                else:
                    reward = REWARD_INVALID_ACTION
                    
        elif action < 2 * self.num_vnf_types:
            # UNINSTALL action (Algorithm 1, Steps 11-16)
            vnf_type = VNF_TYPES[action - self.num_vnf_types]
            
            # Check if there's an idle VNF of this type (Algorithm 1, Step 13)
            installed = dc.installed_vnfs.get(vnf_type, 0)
            allocated_count = sum(1 for v in dc.allocated_vnfs.values() if v == vnf_type)
            idle = installed - allocated_count
            
            if idle <= 0:
                # No idle VNF to uninstall - invalid action
                reward = REWARD_INVALID_ACTION
            else:
                # Check if any VNF of this type is waiting to be allocated (Algorithm 1, Step 14)
                vnfs_waiting = any(
                    sfc.get_current_vnf() == vnf_type 
                    for sfc in self.traffic_gen.active_sfcs if sfc.active
                )
                
                if vnfs_waiting:
                    # Uninstalling a VNF that is still needed - penalty
                    reward = REWARD_UNINSTALL_REQUIRED_VNF
                else:
                    reward = 0  # Valid uninstall, no penalty
                
                # Perform uninstall (Algorithm 1, Step 15)
                dc.uninstall_vnf(vnf_type)
                
        else:
            # WAIT action (Algorithm 1, Steps 8-10)
            reward = 0
        
        # Update time for all active SFCs
        self._update_sfc_times(ACTION_INFERENCE_TIME)
        
        # Check for dropped SFCs due to delay violation
        self._check_dropped_sfcs()
        
        next_state = self._get_state(dc_id)
        
        # Episode done when no active SFCs
        done = not any(sfc.active for sfc in self.traffic_gen.active_sfcs)
        
        return next_state, reward, done
    
    def _deallocate_sfc(self, sfc):
        """Deallocate all VNF resources for a completed/dropped SFC."""
        for dc in self.network.dcs:
            keys_to_remove = [k for k, v in dc.allocated_vnfs.items() if k == sfc.id]
            for key in keys_to_remove:
                del dc.allocated_vnfs[key]
    
    def _update_sfc_times(self, time_delta):
        """Update elapsed time for all active SFCs."""
        for sfc in self.traffic_gen.active_sfcs:
            if sfc.active:
                sfc.elapsed_time += time_delta
    
    def _check_dropped_sfcs(self):
        """Check and mark SFCs that have exceeded their E2E delay."""
        for sfc in self.traffic_gen.active_sfcs:
            if sfc.active and sfc.check_delay_violation():
                sfc.active = False
                self.sfcs_dropped += 1
                self._deallocate_sfc(sfc)
    
    def advance_step(self):
        """
        Advance simulation by one step (T = 1ms).
        Generate new SFCs every N = 4 steps.
        """
        self.step_count += 1
        self.current_time += STEP_DURATION
        
        # Update time for all SFCs
        self._update_sfc_times(STEP_DURATION)
        
        # Generate new SFCs every N steps (Algorithm 1 training description)
        if self.step_count % SFC_GENERATION_INTERVAL == 0:
            new_sfcs = self.traffic_gen.generate_bundle(request_count=1)
            self.sfcs_total += len(new_sfcs)
            
            # Assign shortest paths to new SFCs
            for sfc in new_sfcs:
                sfc.shortest_path = self.network.get_shortest_path(sfc.src, sfc.dst, sfc.bw)
        
        # Check for dropped SFCs
        self._check_dropped_sfcs()
    
    def has_pending_sfcs(self):
        """Check if there are any pending SFC requests."""
        return any(sfc.active for sfc in self.traffic_gen.active_sfcs)
    
    def get_stats(self):
        """Get current statistics."""
        return {
            'satisfied': self.sfcs_satisfied,
            'dropped': self.sfcs_dropped,
            'total': self.sfcs_total,
            'active': sum(1 for sfc in self.traffic_gen.active_sfcs if sfc.active),
            'step': self.step_count,
            'time': self.current_time
        }


class PretrainDQN:
    """
    Pretrain DQN following Algorithm 1 and Section III-A-4 of the paper.
    
    Training procedure:
    - U = 350 updates
    - E = 20 episodes per update  
    - A = 100 actions per step
    - Step duration T = 1ms
    - SFC generation every N = 4 steps
    """
    
    def __init__(self, num_dcs=4):
        self.num_dcs = num_dcs
        self.env = SFCProvisioningEnv(num_dcs)
        
        # Initialize DQN model with proper state dimensions
        self.dqn = DQNModel(
            dc_state_dim=self.env.state1_dim,
            sfc_state_dim=self.env.state2_dim,
            network_state_dim=self.env.state3_dim
        )
        
        self.replay_buffer = ReplayBuffer()
        
        # Statistics
        self.stats = {
            'episodes': [],
            'rewards': [],
            'acceptance': [],
            'dropped': []
        }
    
    def train(self, num_updates=NUM_UPDATES, episodes_per_update=EPISODES_PER_UPDATE):
        """
        Main training loop following the paper's procedure.
        
        Training has U updates, each update after E episodes.
        Each episode processes SFC requests until none pending.
        Each step performs A actions.
        """
        print(f"\n{'='*60}")
        print("Pretrain DQN for SFC Provisioning")
        print(f"Following Algorithm 1 from the paper")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  - Number of DCs: {self.num_dcs}")
        print(f"  - Updates (U): {num_updates}")
        print(f"  - Episodes per update (E): {episodes_per_update}")
        print(f"  - Actions per step (A): {ACTIONS_PER_STEP}")
        print(f"  - Step duration (T): {STEP_DURATION} ms")
        print(f"  - SFC generation interval (N): {SFC_GENERATION_INTERVAL} steps")
        print(f"{'='*60}")
        print(f"Starting training...\n")
        
        total_episodes = 0
        
        for update in range(num_updates):
            update_rewards = []
            update_acceptance = []
            update_dropped = []
            
            for ep in range(episodes_per_update):
                episode_reward, acceptance_rate, drop_rate = self._run_episode()
                
                update_rewards.append(episode_reward)
                update_acceptance.append(acceptance_rate)
                update_dropped.append(drop_rate)
                
                total_episodes += 1
                
                # Store stats
                self.stats['episodes'].append(total_episodes)
                self.stats['rewards'].append(episode_reward)
                self.stats['acceptance'].append(acceptance_rate)
                self.stats['dropped'].append(drop_rate)
                
                # Decay epsilon
                self.dqn.decay_epsilon()
                
                # Log every 20 episodes
                if total_episodes % 20 == 0:
                    avg_reward_20 = np.mean(self.stats['rewards'][-20:])
                    avg_acceptance_20 = np.mean(self.stats['acceptance'][-20:])
                    avg_dropped_20 = np.mean(self.stats['dropped'][-20:])
                    print(f"  Episode {total_episodes} | "
                          f"Reward: {episode_reward:.2f} | "
                          f"Avg Reward (20): {avg_reward_20:.2f} | "
                          f"Avg Acceptance (20): {avg_acceptance_20:.2%} | "
                          f"Avg Dropped (20): {avg_dropped_20:.2%} | "
                          f"Epsilon: {self.dqn.epsilon:.3f}")
            
            # Update target network after each update cycle
            self.dqn.update_target_model()
            
            # Train on replay buffer
            if len(self.replay_buffer) >= TRAINING_CONFIG['batch_size']:
                for _ in range(10):  # Multiple training iterations per update
                    batch = self.replay_buffer.sample(TRAINING_CONFIG['batch_size'])
                    states, actions, rewards, next_states_batch, dones = batch
                    self.dqn.train_step(states, actions, rewards, next_states_batch, dones)
            
            # Print progress per update
            avg_reward = np.mean(update_rewards)
            avg_acceptance = np.mean(update_acceptance)
            avg_dropped = np.mean(update_dropped)
            
            print(f"Update {update+1}/{num_updates} | "
                  f"Episodes: {total_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Acceptance: {avg_acceptance:.2%} | "
                  f"Dropped: {avg_dropped:.2%} | "
                  f"Epsilon: {self.dqn.epsilon:.3f}")
        
        print(f"\n{'='*60}")
        print("Training Completed!")
        print(f"{'='*60}")
        
        return self.dqn, self.stats
    
    def _run_episode(self):
        """
        Run a single episode following Algorithm 1.
        
        Episode starts with incoming SFC requests and ends once there are no pending requests.
        At each step, performs A actions across prioritized DCs.
        
        Paper: "Each episode starts with incoming SFC requests and ends once there are 
        no pending SFC requests."
        
        We generate SFCs only at specific intervals (first 3 request counts as in paper evaluation).
        """
        self.env.reset()
        episode_reward = 0
        max_steps = 50  # Reduced for faster episodes
        request_counts_generated = 1  # Already generated 1 at reset
        max_request_counts = 1  # Only initial bundle for faster training
        
        while self.env.has_pending_sfcs() and self.env.step_count < max_steps:
            # Perform A actions in this step
            step_reward = 0
            
            for action_idx in range(ACTIONS_PER_STEP):
                if not self.env.has_pending_sfcs():
                    break
                
                # Get DC priority list (Algorithm 1, Step 1)
                dc_priority_list = self.env.set_dc_priority()
                
                # Select highest priority DC (Algorithm 1, Step 2)
                dc_id = dc_priority_list[0]
                
                # Get states (Algorithm 1, Steps 3-5)
                dc_state, sfc_state, network_state = self.env._get_state(dc_id)
                
                # Get action from DRL model (Algorithm 1, Step 6)
                action = self.dqn.get_action(dc_state, sfc_state, network_state)
                
                # Execute action and get reward (Algorithm 1, Steps 7-32)
                next_states, reward, done = self.env.step(dc_id, action)
                
                # Store experience in replay buffer (Algorithm 1, Step 34)
                self.replay_buffer.push(
                    dc_state, sfc_state, network_state,
                    action, reward,
                    next_states[0], next_states[1], next_states[2],
                    done
                )
                
                step_reward += reward
                
                if done:
                    break
            
            episode_reward += step_reward
            
            # Advance step - but limit SFC generation to first few intervals
            self.env.step_count += 1
            self.env.current_time += STEP_DURATION
            self.env._update_sfc_times(STEP_DURATION)
            
            # Generate new SFCs at intervals, but only up to max_request_counts
            if (self.env.step_count % SFC_GENERATION_INTERVAL == 0 and 
                request_counts_generated < max_request_counts):
                new_sfcs = self.env.traffic_gen.generate_bundle(request_count=1)
                self.env.sfcs_total += len(new_sfcs)
                for sfc in new_sfcs:
                    sfc.shortest_path = self.env.network.get_shortest_path(sfc.src, sfc.dst, sfc.bw)
                request_counts_generated += 1
            
            self.env._check_dropped_sfcs()
            
            # Train periodically within episode
            if len(self.replay_buffer) >= TRAINING_CONFIG['batch_size']:
                batch = self.replay_buffer.sample(TRAINING_CONFIG['batch_size'])
                states, actions, rewards, next_states_batch, dones = batch
                self.dqn.train_step(states, actions, rewards, next_states_batch, dones)
        
        # Calculate acceptance and drop rates
        stats = self.env.get_stats()
        total = max(stats['total'], 1)
        acceptance_rate = stats['satisfied'] / total
        drop_rate = stats['dropped'] / total
        
        return episode_reward, acceptance_rate, drop_rate
    
    def save_model(self, path="checkpoints/pretrained_dqn"):
        """Save trained model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.dqn.save_weights(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="checkpoints/pretrained_dqn"):
        """Load model weights."""
        self.dqn.load_weights(path)
        print(f"Model loaded from {path}")


def evaluate_model(dqn, num_dcs=4, num_episodes=50):
    """
    Evaluate trained model performance.
    """
    env = SFCProvisioningEnv(num_dcs)
    
    results = {
        'acceptance_ratios': [],
        'drop_ratios': [],
        'rewards': []
    }
    
    # Disable exploration during evaluation
    original_epsilon = dqn.epsilon
    dqn.epsilon = 0
    
    for ep in range(num_episodes):
        env.reset()
        episode_reward = 0
        max_steps = 200
        
        while env.has_pending_sfcs() and env.step_count < max_steps:
            for _ in range(ACTIONS_PER_STEP):
                if not env.has_pending_sfcs():
                    break
                
                dc_priority_list = env.set_dc_priority()
                dc_id = dc_priority_list[0]
                
                dc_state, sfc_state, network_state = env._get_state(dc_id)
                action = dqn.get_action(dc_state, sfc_state, network_state)
                
                next_states, reward, done = env.step(dc_id, action)
                episode_reward += reward
                
                if done:
                    break
            
            env.advance_step()
        
        stats = env.get_stats()
        total = max(stats['total'], 1)
        
        results['acceptance_ratios'].append(stats['satisfied'] / total)
        results['drop_ratios'].append(stats['dropped'] / total)
        results['rewards'].append(episode_reward)
    
    # Restore epsilon
    dqn.epsilon = original_epsilon
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Average Acceptance Ratio: {np.mean(results['acceptance_ratios']):.2%}")
    print(f"Average Drop Ratio: {np.mean(results['drop_ratios']):.2%}")
    print(f"Average Reward: {np.mean(results['rewards']):.2f}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pretrain DQN for SFC Provisioning')
    parser.add_argument('--num_dcs', type=int, default=4, help='Number of data centers')
    parser.add_argument('--updates', type=int, default=NUM_UPDATES, help='Number of updates')
    parser.add_argument('--episodes_per_update', type=int, default=EPISODES_PER_UPDATE, 
                        help='Episodes per update')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--eval_episodes', type=int, default=50, help='Number of evaluation episodes')
    parser.add_argument('--save_path', type=str, default='checkpoints/pretrained_dqn',
                        help='Path to save model')
    
    args = parser.parse_args()
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize trainer
    trainer = PretrainDQN(num_dcs=args.num_dcs)
    
    # Train
    dqn, stats = trainer.train(
        num_updates=args.updates,
        episodes_per_update=args.episodes_per_update
    )
    
    # Save model
    trainer.save_model(args.save_path)
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("Final Training Statistics")
    print(f"{'='*60}")
    print(f"Average Reward (last 100 episodes): {np.mean(stats['rewards'][-100:]):.2f}")
    print(f"Average Acceptance (last 100 episodes): {np.mean(stats['acceptance'][-100:]):.2%}")
    print(f"Average Dropped (last 100 episodes): {np.mean(stats['dropped'][-100:]):.2%}")
    
    # Run evaluation if requested
    if args.eval:
        evaluate_model(dqn, num_dcs=args.num_dcs, num_episodes=args.eval_episodes)
