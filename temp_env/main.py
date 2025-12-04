import numpy as np
import sys
sys.path.append('.')

from env.core_network import CoreNetwork
from env.traffic_generator import TrafficGenerator
from models.dqn_model import DQNModel
from models.vae_model import VAEModel
from utils.buffers import ReplayBuffer, VAEDataset
from config import VNF_TYPES, VNF_SPECS, TRAINING_CONFIG

class VNFPlacementEnv:
    def __init__(self, num_dcs=4):
        self.network = CoreNetwork(num_dcs)
        self.traffic_gen = TrafficGenerator(num_dcs)
        self.current_sfc = None
        self.dc_state_dim = 10
        self.sfc_state_dim = 11
        self.network_state_dim = self.dc_state_dim * num_dcs + num_dcs * num_dcs
        
    def reset(self):
        self.network.reset()
        self.traffic_gen.active_sfcs = []
        self.traffic_gen.generate_bundle(request_count=1)
        if len(self.traffic_gen.active_sfcs) > 0:
            self.current_sfc = self.traffic_gen.active_sfcs[0]
        return self.get_state()
    
    def get_state(self, dc_id=0):
        if self.current_sfc is None:
            return None, None, None
        
        dc_state = self.network.dcs[dc_id].get_state()
        sfc_state = self.current_sfc.get_state()
        network_state = self.network.get_network_state()
        
        return dc_state, sfc_state, network_state
    
    def get_valid_actions(self, dc_id):
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
        reward = 0
        done = False
        
        if self.current_sfc is None:
            return None, -1, True
        
        dc = self.network.dcs[dc_id]
        
        if action < len(VNF_TYPES):
            vnf_type = VNF_TYPES[action]
            if dc.install_vnf(vnf_type):
                reward = -0.1
            else:
                reward = -1
                
        elif action < 2 * len(VNF_TYPES):
            vnf_type = VNF_TYPES[action - len(VNF_TYPES)]
            if dc.uninstall_vnf(vnf_type):
                reward = -0.5
            else:
                reward = -1
        else:
            reward = 0
        
        current_vnf = self.current_sfc.get_current_vnf()
        if current_vnf and dc.can_allocate(current_vnf, self.current_sfc.id):
            if dc_id not in self.current_sfc.placement:
                dc.allocate_vnf(current_vnf, self.current_sfc.id)
                process_time = VNF_SPECS[current_vnf]['process_time']
                self.current_sfc.advance_vnf(dc_id, process_time)
                reward += 0.5
                
                if self.current_sfc.is_complete():
                    if not self.current_sfc.check_delay_violation():
                        reward = 2
                        self.current_sfc.active = False
                    else:
                        reward = -1.5
                        self.current_sfc.active = False
                    done = True
        
        if not done and self.current_sfc.check_delay_violation():
            reward = -1.5
            self.current_sfc.active = False
            done = True
        
        if done:
            self.traffic_gen.remove_completed()
            if len(self.traffic_gen.active_sfcs) > 0:
                self.current_sfc = self.traffic_gen.active_sfcs[0]
                done = False
            else:
                self.current_sfc = None
        
        next_state = self.get_state(dc_id)
        return next_state, reward, done

def collect_vae_data(num_dcs=4, episodes=100):
    env = VNFPlacementEnv(num_dcs)
    dataset = VAEDataset()
    
    print("Collecting VAE training data...")
    for ep in range(episodes):
        env.reset()
        step_count = 0
        
        while env.current_sfc is not None and step_count < 200:
            dc_id = np.random.randint(num_dcs)
            current_dc_state = env.network.dcs[dc_id].get_state()
            
            valid_actions = env.get_valid_actions(dc_id)
            action = np.random.choice(valid_actions)
            
            _, reward, _ = env.step(dc_id, action)
            
            next_dc_state = env.network.dcs[dc_id].get_state()
            
            state_change = np.linalg.norm(next_dc_state - current_dc_state)
            value = reward + 0.5 * state_change
            
            dataset.add(current_dc_state, next_dc_state, value)
            step_count += 1
        
        if (ep + 1) % 20 == 0:
            print(f"Episode {ep+1}/{episodes}, Dataset size: {len(dataset)}")
    
    return dataset

def train_vae(vae_model, dataset, epochs=50):
    print("\nTraining VAE model...")
    batch_size = TRAINING_CONFIG['batch_size']
    
    for epoch in range(epochs):
        losses = []
        num_batches = min(100, len(dataset) // batch_size)
        
        for _ in range(num_batches):
            batch = dataset.get_batch(batch_size)
            if batch is None:
                continue
            current_states, next_states, values = batch
            
            current_states = (current_states - np.mean(current_states, axis=0, keepdims=True)) / (np.std(current_states, axis=0, keepdims=True) + 1e-8)
            next_states = (next_states - np.mean(next_states, axis=0, keepdims=True)) / (np.std(next_states, axis=0, keepdims=True) + 1e-8)
            
            total_loss, recon_loss, kl_loss, val_loss = vae_model.train_step(
                current_states, next_states, values
            )
            losses.append([total_loss.numpy(), recon_loss.numpy(), kl_loss.numpy(), val_loss.numpy()])
        
        if (epoch + 1) % 10 == 0:
            avg_losses = np.mean(losses, axis=0)
            print(f"Epoch {epoch+1}/{epochs} | Total: {avg_losses[0]:.4f} | Recon: {avg_losses[1]:.4f} | KL: {avg_losses[2]:.4f} | Value: {avg_losses[3]:.4f}")

def train_dqn(num_dcs=4, episodes=500):
    env = VNFPlacementEnv(num_dcs)
    
    dqn = DQNModel(env.dc_state_dim, env.sfc_state_dim, env.network_state_dim)
    vae = VAEModel(env.dc_state_dim)
    replay_buffer = ReplayBuffer()
    
    vae_dataset = collect_vae_data(num_dcs, episodes=100)
    train_vae(vae, vae_dataset, epochs=500)
    
    print("\nTraining DQN with VAE assistance...")
    stats = {'episodes': [], 'rewards': [], 'acceptance': []}
    
    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        accepted = 0
        total_requests = len(env.traffic_gen.active_sfcs)
        step_count = 0
        
        while env.current_sfc is not None and step_count < 500:
            dc_states = [env.network.dcs[i].get_state() for i in range(num_dcs)]
            dc_values = vae.get_dc_values(dc_states)
            dc_id = np.argmax(dc_values)
            
            dc_state, sfc_state, network_state = env.get_state(dc_id)
            if dc_state is None:
                break
            
            valid_actions = env.get_valid_actions(dc_id)
            action = dqn.get_action(dc_state, sfc_state, network_state, valid_actions)
            
            next_states, reward, done = env.step(dc_id, action)
            
            if next_states[0] is not None:
                replay_buffer.push(dc_state, sfc_state, network_state, action, reward,
                                 next_states[0], next_states[1], next_states[2], done)
            
            episode_reward += reward
            if reward == 2:
                accepted += 1
            
            if len(replay_buffer) >= TRAINING_CONFIG['batch_size']:
                states, actions, rewards, next_states, dones = replay_buffer.sample(
                    TRAINING_CONFIG['batch_size']
                )
                dqn.train_step(states, actions, rewards, next_states, dones)
            
            step_count += 1
        
        dqn.decay_epsilon()
        
        if episode % TRAINING_CONFIG['target_update'] == 0:
            dqn.update_target_model()
        
        acceptance_ratio = accepted / max(total_requests, 1)
        stats['episodes'].append(episode)
        stats['rewards'].append(episode_reward)
        stats['acceptance'].append(acceptance_ratio)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(stats['rewards'][-20:])
            avg_acceptance = np.mean(stats['acceptance'][-20:])
            print(f"Episode {episode+1}/{episodes} | Reward: {avg_reward:.2f} | "
                  f"Acceptance: {avg_acceptance:.2%} | Epsilon: {dqn.epsilon:.3f}")
    
    dqn.save_weights("checkpoints/model")
    vae.save_weights("checkpoints/vae")
    print("\nTraining completed!")
    return dqn, vae, stats

if __name__ == "__main__":
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    dqn, vae, stats = train_dqn(num_dcs=10, episodes=500)
    
    print(f"\nFinal Stats:")
    print(f"Average Reward (last 50): {np.mean(stats['rewards'][-50:]):.2f}")
    print(f"Average Acceptance (last 50): {np.mean(stats['acceptance'][-50:]):.2%}")