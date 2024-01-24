from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import json


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if self.args.env_args['map_name'] == 'search_engine':

            fileQueryPermutaion = './data/baidu/doc15_2/query_permutation_doc15_2.json'
            fileQueryRepresentation = './data/baidu/doc15_2/query_features_doc15_2_acge.json'
            fileDocumentRepresentation = './data/baidu/doc15_2/doc_features_doc15_2_acge.json'
            fileQueryDocumentSubtopics = './data/baidu/doc15_2/query_doc_subtopics_doc15_2.json'

            # load data
            with open(fileQueryPermutaion) as self.fileQueryPermutaion:
                self.dictQueryPermutaion = json.load(self.fileQueryPermutaion)

            with open(fileQueryRepresentation) as self.fileQueryRepresentation:
                self.dictQueryRepresentation = json.load(self.fileQueryRepresentation)
            
            with open(fileDocumentRepresentation) as self.fileDocumentRepresentation:
                self.dictDocumentRepresentation = json.load(self.fileDocumentRepresentation)

            with open(fileQueryDocumentSubtopics) as self.fileQueryDocumentSubtopics:
                self.dictQueryDocumentSubtopics = json.load(self.fileQueryDocumentSubtopics)

            self.args.env_args['dictQueryPermutaion'] = self.dictQueryPermutaion
            self.args.env_args['dictQueryRepresentation'] = self.dictQueryRepresentation
            self.args.env_args['dictDocumentRepresentation'] = self.dictDocumentRepresentation
            self.args.env_args['dictQueryDocumentSubtopics'] = self.dictQueryDocumentSubtopics

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, query_id, doc_list, obs_actions, test_mode):
        self.t = 0
        self.batch = self.new_batch()  
        self.env.reset(query_id, doc_list, obs_actions, test_mode, self.t)

    def run(self, test_mode=False, query_id='', doc_list=[]): 

        # initial scores
        obs_actions = [0] * self.env.n_agents
        obs_actions = self.get_onehot(obs_actions)

        self.reset(query_id, doc_list, obs_actions, test_mode=test_mode)

        terminated = False
        episode_return = 0
        episode_return_list = []
        alpha_NDCG_5_list, alpha_NDCG_10_list = [], []
        ERR_IA_5_list, ERR_IA_10_list = [], []
        S_recall_5_list, S_recall_10_list = [], []
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state(obs_actions, self.t)],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs(doc_list, obs_actions, self.t)]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()
            
            if test_mode:
                alpha_NDCG_5, alpha_NDCG_10, ERR_IA_5, ERR_IA_10, S_recall_5, S_recall_10, reward, terminated, env_info, cur_ranks = self.env.step(actions[0], query_id, doc_list, t=self.t, test_mode=test_mode, sparse_reward=self.args.sparse_reward)
            else:
                reward, terminated, env_info, cur_ranks = self.env.step(actions[0], query_id, doc_list, t=self.t, test_mode=test_mode, sparse_reward=self.args.sparse_reward)
            episode_return += reward
            episode_return_list.append(reward)
            if test_mode:
                alpha_NDCG_5_list.append(alpha_NDCG_5)
                alpha_NDCG_10_list.append(alpha_NDCG_10)
                ERR_IA_5_list.append(ERR_IA_5)
                ERR_IA_10_list.append(ERR_IA_10)
                S_recall_5_list.append(S_recall_5)
                S_recall_10_list.append(S_recall_10)

            post_transition_data = {
                "actions": cpu_actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            obs_actions = cpu_actions[0].tolist()
            obs_actions = self.get_onehot(obs_actions)

            self.t += 1

        last_data = {
            "state": [self.env.get_state(obs_actions, self.t)],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs(doc_list, obs_actions, self.t)]
        }

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)
        
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return_list[-1])
        # cur_returns.append(episode_return_list[0])

        # cur_returns.append(episode_return / self.t)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        if test_mode:
            # return np.mean(episode_return_list), np.mean(alpha_NDCG_5_list), np.mean(alpha_NDCG_10_list)
            
            # self.logger.log_stat("test_alpha_ndcg_5_by_step", alpha_NDCG_5_list, self.t_env)
            # self.logger.log_stat("test_alpha_ndcg_10_by_step", alpha_NDCG_10_list, self.t_env)

            # return episode_return_list[-1], alpha_NDCG_5_list[-1], alpha_NDCG_10_list[-1], ERR_IA_5_list[-1], ERR_IA_10_list[-1]
            return episode_return_list[-1], alpha_NDCG_5_list[-1], alpha_NDCG_10_list[-1], ERR_IA_5_list[-1], ERR_IA_10_list[-1], S_recall_5_list[-1], S_recall_10_list[-1]
            # return episode_return_list[0], alpha_NDCG_5_list[0], alpha_NDCG_10_list[0], ERR_IA_5_list[0], ERR_IA_10_list[0]

            # return episode_return_list[0], alpha_NDCG_5_list[0], alpha_NDCG_10_list[0]
        
        else:
            return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def get_onehot(self, input_list):
        original_list = input_list
        one_hot_matrix = np.zeros((self.env.n_agents, self.env.m_score))
        # one-hot
        for i, val in enumerate(original_list):
            one_hot_matrix[i, val] = 1
        one_hot_list = one_hot_matrix.tolist()
        flattened_one_hot_list = [item for sublist in one_hot_list for item in sublist]

        return flattened_one_hot_list
