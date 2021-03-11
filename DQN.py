import os
import gym
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from IPython.display import clear_output

from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import MlpPolicy

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps),end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True

        # Create log dir
load = [4000]
holding_time = [400]
for i, j in zip(load, holding_time):
  log_dir = "/home/zceened/ProjectFiles/DQN/Final/CLSE/{}/tmp1/".format(i)
  os.makedirs(log_dir, exist_ok=True)
  callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
  
  
  topology_name = 'NSFNET'
  k_paths = 5
  with open(f"/home/zceened/optical-rl-gym/examples/topologies/NSFNET_chen_eon_5-paths_CLSE.h5", 'rb') as f:
      topology = pickle.load(f)
  
  node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505, #add comma again
         0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
         0.07607608, 0.12012012, 0.01901902, 0.16916917])
  env_args = dict(topology=topology, seed=10, allow_rejection=False,
                 mean_service_holding_time=j, episode_length=50, j=1, node_request_probabilities=node_request_probabilities)
  training_env = gym.make('DeepRMSA-v0', **env_args)
  
  #print(training_env.action_space.n)
  # Logs will be saved in log_dir/monitor.csv
  training_env = Monitor(training_env, log_dir + '500ktraining', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))
  # kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)} # set of parameters for testing
  policy_kwargs = {'layers': [128] * 4, 'dueling': False}
  model = DQN(MlpPolicy, training_env, verbose=0, tensorboard_log="./tb/DQN-DeepRMSA-v0/", double_q=False, gamma=.95, policy_kwargs=policy_kwargs,
             learning_rate=10e-5)
  
  env_args['seed'] = env_args['seed'] + 1
  testing_env = gym.make('DeepRMSA-v0', **env_args)
  
  # Logs will be saved in log_dir/monitor.csv
  testing_env = Monitor(testing_env, log_dir + '500ktesting', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))
  
  from stable_baselines.common.evaluation import evaluate_policy
  mean_reward, std_reward = evaluate_policy(model, testing_env, n_eval_episodes=10)
  print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
  import os
 
  a = model.learn(total_timesteps=500000, callback=callback)
  os.remove("/home/zceened/ProjectFiles/DQN/blockingReason4000.csv")
  #Evaluate the trained agent
  mean_reward, std_reward = evaluate_policy(model, testing_env, n_eval_episodes=1000)
  
  print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
  #from stable_baselines import results_plotter
  model.save("/home/zceened/ProjectFiles/DQN/Final/CLSE/{}/best".format(i))
  # Helper from the library
  #results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "DQN DeepRMSA v0")
