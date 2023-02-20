import time
import os

import numpy as np
import torch
import gym

from typing import Optional, Dict, List
from tqdm import tqdm
from collections import deque
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.policy import BasePolicy


# model-free policy trainer
class MFPolicyTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        eval_env: gym.Env,
        buffer: ReplayBuffer,
        interaction_buffer: ReplayBuffer,
        logger: Logger,
        interaction_tag: bool = False,
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        normalize_obs: bool = False,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        self.logger = logger
        self.interaction_buffer = interaction_buffer

        self.interaction_tag = interaction_tag
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self._normalize_obs = normalize_obs
        if normalize_obs:
            self._obs_mean, self._obs_std = self.buffer.normalize_obs()
        self.lr_scheduler = lr_scheduler

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        # train loop
        for e in range(1, self._epoch + 1):

            self.policy.train()
            
            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            # 每隔十轮和环境交互一次
            interaction_interval = 20
            # if e % interaction_interval != interaction_interval - 1:
                # pass
            for it in pbar:
                batch = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)
                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)
                num_timesteps += 1
            
            # on policy train
            # if e % interaction_interval == interaction_interval - 1:
            #     obs = self.eval_env.reset()
            #     for _ in range(self._step_per_epoch):
                    
            #         action = self.policy.select_action(obs, deterministic=False)
            #         next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            #         # print(f'interaction data \n obs:{obs}\n action:{action}\n next_obs:{next_obs}\n reward:{reward}\n terminal:{terminal}')
            #         # batch = 
            #         self.interaction_buffer.add(
            #             obs=obs,
            #             next_obs=next_obs,
            #             action=action,
            #             reward=reward,
            #             terminal=terminal
            #         )
                    

            #         batch = self.interaction_buffer.sample(batch_size=1)
                    
            #         loss = self.policy.learn(batch)
            #         obs = next_obs
            #         if terminal:
            #             break

            # off policy train

            if self.interaction_tag == True:
                if e % interaction_interval == interaction_interval - 18:
                    interaction_times = 0
                    obs = self.eval_env.reset()
                    for _steps in range(self._step_per_epoch):
                        
                        action = self.policy.select_action(obs, deterministic=False)
                        next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
                        # print(f'interaction data \n obs:{obs}\n action:{action}\n next_obs:{next_obs}\n reward:{reward}\n terminal:{terminal}')
                        # batch = 
                        self.interaction_buffer.add(
                            obs=obs,
                            next_obs=next_obs,
                            action=action,
                            reward=reward,
                            terminal=terminal
                        )
                        obs = next_obs
                        interaction_times += 1
                        if terminal:
                            break

                    Train_times = 5
                    batch_size = min(32, interaction_times)
                    _device = "cuda" if torch.cuda.is_available() else "cpu"
                    for train_time in range(Train_times):


                        batch = self.interaction_buffer.sample(batch_size=batch_size)
                        # print(f"batch['observations'].size():{batch['observations'].size()}")
                        # 数据增强
                        _observations = batch['observations']
                        _next_observations = batch['next_observations']


                        obs_random = np.random.randn(_observations.shape[0], _observations.shape[1]) / 1000
                        obs_random = torch.tensor(data=obs_random, device=_device)
                        # print(f'obs_random:{obs_random}')
                        next_obs_random = np.random.randn(_observations.shape[0], _observations.shape[1]) / 1000
                        next_obs_random = torch.tensor(data=next_obs_random, device=_device)
                        # print(f'next_obs_random:{next_obs_random}')
                        obs_augmentation = obs_random + _observations
                        next_obs_augmentation = next_obs_random + _next_observations
                        
                        
                        batch_augmentaton = {
                            'observations': torch.tensor(data=obs_augmentation, dtype=torch.float32, device=_device),
                            'actions': batch['actions'].clone().detach(),
                            'next_observations': torch.tensor(data=next_obs_augmentation, dtype=torch.float32, device=_device),
                            'terminals': batch['terminals'].clone().detach(),
                            'rewards': batch['rewards'].clone().detach()
                        }

                        total_batch = {}

                        for a_item, b_item in zip(batch.items(), batch_augmentaton.items()):
                            if a_item[0] == b_item[0]:
                                total_batch[a_item[0]] = torch.cat([a_item[1], b_item[1]],dim=0)
                        # print(total_batch)
                        # kkkk
                        # add random data to data augmentation
                        # self.buffer_train.add(obs_augmentation, next_obs_augmentation, action, reward, terminal)
                        # print(f"total_batch['observations'].size():{total_batch['observations'].size()}")
                        # print(f"total_batch['next_observations'].size():{total_batch['next_observations'].size()}")
                        loss = self.policy.learn(total_batch)



            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # evaluate current policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_rew_mean = self.eval_env.get_normalized_score(ep_reward_mean) * 100
            norm_ep_rew_std = self.eval_env.get_normalized_score(ep_reward_std) * 100
            last_10_performance.append(norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward", norm_ep_rew_mean)
            self.logger.logkv("eval/normalized_episode_reward_std", norm_ep_rew_std)
            self.logger.logkv("eval/episode_length", ep_length_mean)
            self.logger.logkv("eval/episode_length_std", ep_length_std)
            self.logger.set_timestep(num_timesteps)
            self.logger.dumpkvs()
        
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.logger.close()

        return {"last_10_performance": np.mean(last_10_performance)}

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0
        episode_reward, episode_length = 0, 0

        while num_episodes < self._eval_episodes:
            if self._normalize_obs:
                obs = (np.array(obs).reshape(1,-1) - self._obs_mean) / self._obs_std
            action = self.policy.select_action(obs, deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes +=1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
        
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }
