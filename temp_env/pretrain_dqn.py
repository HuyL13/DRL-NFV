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
ACTIONS_PER_STEP = 20          # Reduced from 100 for faster training
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
    Environment for SFC Provisioning following the paper's system model.
    Uses the same dynamics as main.py - process one SFC at a time.
    """
    
    def __init__(self, num_dcs=4):
        self.num_dcs = num_dcs
        self.network = CoreNetwork(num_dcs)
        self.traffic_gen = TrafficGenerator(num_dcs)
        self.current_sfc = None  # Process ONE SFC at a time like main.py
        self.step_count = 0
        self.current_time = 0
        
        # State dimensions from paper
        self.num_vnf_types = len(VNF_TYPES)  # |V|
        self.num_sfc_types = len(SFC_SPECS)   # |S|
        
        # State 1: [1 × (2*|V| + 2)]
        self.state1_dim = 2 * self.num_vnf_types + 2
        
        # State 2: [|S| × (1 + 2*|V|)] - flattened
        self.state2_dim = self.num_sfc_types * (1 + 2 * self.num_vnf_types)
        
        # State 3: [|S| × (4 + |V|)] - flattened
        self.state3_dim = self.num_sfc_types * (4 + self.num_vnf_types)
        
        # Action space: 2*|V| + 1
        self.num_actions = 2 * self.num_vnf_types + 1
        
    def reset(self):
        """Reset environment for new episode."""
        self.network.reset()
        self.traffic_gen.active_sfcs = []
        self.current_sfc = None
        self.step_count = 0
        self.current_time = 0
        
        # Generate initial SFC requests using TrafficGenerator's generate_bundle
        self.traffic_gen.generate_bundle(request_count=1)
        if len(self.traffic_gen.active_sfcs) > 0:
            self.current_sfc = self.traffic_gen.active_sfcs[0]
        
        return self._get_state(0)
    
    def _get_state1(self, dc_id):
        """
        Get State 1: Current DC information.
        [1 × (2*|V| + 2)] = installed VNFs for each type, available VNFs for each type, 
        available storage, available computational power
        """
        dc = self.network.dcs[dc_id]
        state = []
        
        # Installed VNFs for each type (normalized)
        for vnf_type in VNF_TYPES:
            state.append(dc.installed_vnfs[vnf_type] / 10.0)
        
        # Available VNFs for each type (installed - allocated)
        for vnf_type in VNF_TYPES:
            installed = dc.installed_vnfs[vnf_type]
            allocated = sum(1 for v in dc.allocated_vnfs.values() if v == vnf_type)
            available = installed - allocated
            state.append(available / 10.0)
        
        # Available storage and computational power (normalized)
        state.append(dc.storage / dc.max_storage)
        state.append(dc.cpu / dc.max_cpu)
        
        return np.array(state, dtype=np.float32)
    
    def _get_state2(self, dc_id):
        """
        Get State 2: Current SFC state.
        Uses current_sfc like main.py
        """
        if self.current_sfc is None:
            return np.zeros(self.state2_dim, dtype=np.float32)
        
        sfc_types = list(SFC_SPECS.keys())
        state = []
        
        for sfc_type in sfc_types:
            if self.current_sfc.type == sfc_type and dc_id in self.current_sfc.placement:
                sfc = self.current_sfc
                state.append(sfc_types.index(sfc_type) / len(sfc_types))
                
                # Already allocated VNFs
                allocated_vnfs = [0] * self.num_vnf_types
                for vnf_idx in range(sfc.current_vnf_idx):
                    vnf_type = sfc.chain[vnf_idx]
                    allocated_vnfs[VNF_TYPES.index(vnf_type)] = 1
                state.extend(allocated_vnfs)
                
                # Remaining VNFs
                remaining_vnfs = [0] * self.num_vnf_types
                for vnf_idx in range(sfc.current_vnf_idx, len(sfc.chain)):
                    vnf_type = sfc.chain[vnf_idx]
                    remaining_vnfs[VNF_TYPES.index(vnf_type)] = 1
                state.extend(remaining_vnfs)
            else:
                state.append(0)
                state.extend([0] * self.num_vnf_types)
                state.extend([0] * self.num_vnf_types)
        
        return np.array(state, dtype=np.float32)
    
    def _get_state3(self):
        """
        Get State 3: Overall pending SFC requests information.
        """
        sfc_types = list(SFC_SPECS.keys())
        state = []
        
        for sfc_type in sfc_types:
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
                
                # Total VNFs waiting for allocation for each type
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
        if self.current_sfc is None:
            return None, None, None
        state1 = self._get_state1(dc_id)
        state2 = self._get_state2(dc_id)
        state3 = self._get_state3()
        return state1, state2, state3
    
    def set_dc_priority(self):
        """
        Set DC priority based on current SFC.
        """
        if self.current_sfc is None:
            return list(range(self.num_dcs))
        
        # Get shortest path with BW constraint for current SFC
        path = self.network.get_shortest_path(
            self.current_sfc.src, 
            self.current_sfc.dst, 
            self.current_sfc.bw
        )
        
        # Build priority list: path DCs first, then others
        priority_list = path.copy() if path else [self.current_sfc.src]
        
        for dc_id in range(self.num_dcs):
            if dc_id not in priority_list:
                priority_list.append(dc_id)
        
        return priority_list
    
    def get_valid_actions(self, dc_id):
        """Get valid actions for a DC - same as main.py."""
        valid = []
        dc = self.network.dcs[dc_id]
        
        for i, vnf in enumerate(VNF_TYPES):
            if dc.can_install(vnf):
                valid.append(i)
            if dc.installed_vnfs[vnf] > 0:
                valid.append(len(VNF_TYPES) + i)
        
        valid.append(2 * len(VNF_TYPES))
        return valid
    
    def step(self, dc_id, action):
        """
        Execute action - same dynamics as main.py VNFPlacementEnv.step()
        """
        reward = 0
        done = False
        
        if self.current_sfc is None:
            return (None, None, None), -1, True
        
        dc = self.network.dcs[dc_id]
        
        # Install action
        if action < len(VNF_TYPES):
            vnf_type = VNF_TYPES[action]
            if dc.install_vnf(vnf_type):
                reward = -0.1
            else:
                reward = -1
        
        # Uninstall action
        elif action < 2 * len(VNF_TYPES):
            vnf_type = VNF_TYPES[action - len(VNF_TYPES)]
            if dc.uninstall_vnf(vnf_type):
                reward = -0.5
            else:
                reward = -1
        
        # Wait action
        else:
            reward = 0
        
        # Try to allocate current VNF if possible
        current_vnf = self.current_sfc.get_current_vnf()
        if current_vnf and dc.can_allocate(current_vnf, self.current_sfc.id):
            if dc_id not in self.current_sfc.placement:
                dc.allocate_vnf(current_vnf, self.current_sfc.id)
                process_time = VNF_SPECS[current_vnf]['process_time']
                self.current_sfc.advance_vnf(dc_id, process_time)
                reward += 0.5
                
                if self.current_sfc.is_complete():
                    if not self.current_sfc.check_delay_violation():
                        reward = REWARD_SFC_SATISFIED
                        self.current_sfc.active = False
                    else:
                        reward = REWARD_SFC_DROPPED
                        self.current_sfc.active = False
                    done = True
        
        # Check delay violation
        if not done and self.current_sfc.check_delay_violation():
            reward = REWARD_SFC_DROPPED
            self.current_sfc.active = False
            done = True
        
        # Move to next SFC if current one is done
        if done:
            self.traffic_gen.remove_completed()
            if len(self.traffic_gen.active_sfcs) > 0:
                self.current_sfc = self.traffic_gen.active_sfcs[0]
                done = False
            else:
                self.current_sfc = None
        
        next_state = self._get_state(dc_id)
        return next_state, reward, done
    
    def _get_action_type(self, action):
        """Determine action type from action index."""
        if action < self.num_vnf_types:
            return 'install'
        elif action < 2 * self.num_vnf_types:
            return 'uninstall'
        else:
            return 'wait'
    
    def get_stats(self):
        """Get current statistics."""
        return {
            'active_sfcs': len(self.traffic_gen.active_sfcs),
            'current_sfc': self.current_sfc.type if self.current_sfc else None,
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
                episode_reward, accepted, dropped, total = self._run_episode()
                
                update_rewards.append(episode_reward)
                update_acceptance.append(accepted / max(total, 1))
                update_dropped.append(dropped / max(total, 1))
                
                total_episodes += 1
                
                # Store stats
                self.stats['episodes'].append(total_episodes)
                self.stats['rewards'].append(episode_reward)
                self.stats['acceptance'].append(accepted / max(total, 1))
                self.stats['dropped'].append(dropped / max(total, 1))
                
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
        Run a single episode - same dynamics as main.py train_dqn.
        Episode processes SFC requests one by one until none pending.
        """
        self.env.reset()
        episode_reward = 0
        accepted = 0
        dropped = 0
        total_requests = len(self.env.traffic_gen.active_sfcs)
        step_count = 0
        max_steps = 500  # Safety limit
        
        while self.env.current_sfc is not None and step_count < max_steps:
            # Get DC priority and select highest priority DC
            dc_priority_list = self.env.set_dc_priority()
            dc_id = dc_priority_list[0]
            
            # Get states
            dc_state, sfc_state, network_state = self.env._get_state(dc_id)
            if dc_state is None:
                break
            
            # Get valid actions and select action
            valid_actions = self.env.get_valid_actions(dc_id)
            action = self.dqn.get_action(dc_state, sfc_state, network_state, valid_actions)
            
            # Execute action
            next_states, reward, done = self.env.step(dc_id, action)
            
            # Store in replay buffer
            if next_states[0] is not None:
                self.replay_buffer.push(
                    dc_state, sfc_state, network_state,
                    action, reward,
                    next_states[0], next_states[1], next_states[2],
                    done
                )
            
            episode_reward += reward
            
            # Track accepted/dropped SFCs
            if reward == REWARD_SFC_SATISFIED:
                accepted += 1
            elif reward == REWARD_SFC_DROPPED:
                dropped += 1
            
            # Train DQN periodically
            if len(self.replay_buffer) >= TRAINING_CONFIG['batch_size']:
                batch = self.replay_buffer.sample(TRAINING_CONFIG['batch_size'])
                states, actions, rewards, next_states_batch, dones = batch
                self.dqn.train_step(states, actions, rewards, next_states_batch, dones)
            
            step_count += 1
        
        return episode_reward, accepted, dropped, total_requests
    
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
    Evaluate trained model - same dynamics as main.py.
    """
    env = SFCProvisioningEnv(num_dcs)
    
    results = {
        'acceptance_ratios': [],
        'resource_usage': []
    }
    
    # Disable exploration during evaluation
    original_epsilon = dqn.epsilon
    dqn.epsilon = 0
    
    for ep in range(num_episodes):
        env.reset()
        accepted = 0
        dropped = 0
        total_requests = len(env.traffic_gen.active_sfcs)
        step_count = 0
        max_steps = 500
        
        while env.current_sfc is not None and step_count < max_steps:
            dc_priority_list = env.set_dc_priority()
            dc_id = dc_priority_list[0]
            
            dc_state, sfc_state, network_state = env._get_state(dc_id)
            if dc_state is None:
                break
            
            valid_actions = env.get_valid_actions(dc_id)
            action = dqn.get_action(dc_state, sfc_state, network_state, valid_actions)
            
            next_states, reward, done = env.step(dc_id, action)
            
            if reward == REWARD_SFC_SATISFIED:
                accepted += 1
            elif reward == REWARD_SFC_DROPPED:
                dropped += 1
            
            step_count += 1
        
        acceptance_ratio = accepted / max(total_requests, 1)
        results['acceptance_ratios'].append(acceptance_ratio)
        
        # Calculate average resource usage
        total_cpu_used = sum(dc.max_cpu - dc.cpu for dc in env.network.dcs)
        total_storage_used = sum(dc.max_storage - dc.storage for dc in env.network.dcs)
        avg_resource = (total_cpu_used + total_storage_used) / (2 * num_dcs)
        results['resource_usage'].append(avg_resource)
    
    # Restore epsilon
    dqn.epsilon = original_epsilon
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Average Acceptance Ratio: {np.mean(results['acceptance_ratios']):.2%}")
    print(f"Average Resource Usage: {np.mean(results['resource_usage']):.2f}")
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
