import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import random
import copy
import yaml
import numpy as np
import json
import math
from tqdm import tqdm

# from smac.env import StarCraft2Env

# def get_agent_own_state_size(env_args):
#     sc_env = StarCraft2Env(**env_args)
#     # qatten parameter setting (only use in qatten)
#     return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def all_doc_evaluate(args, runner, test_mode=True, listTestSet=[]):

    device = runner.args.device
    alpha_NDCG_5_list = []
    alpha_NDCG_10_list = []
    # get query docs in test set
    for query in listTestSet:  # list(listTestSet)
        # print(query)
        # query_repr = runner.env.dictQueryRepresentation[query]
        query_repr = np.squeeze(np.array(runner.env.dictQueryRepresentation[query])).tolist()
        ideal_doc_list = copy.deepcopy(runner.env.dictQueryPermutaion[query]['permutation'])
        random_doc_list = copy.deepcopy(runner.env.dictQueryPermutaion[query]['permutation'])
        # random_doc_list.reverse()
        random.shuffle(random_doc_list)

        flag = False
        for doc in random_doc_list:
            query_doc_list = runner.env.dictQueryDocumentSubtopics[query].keys()
            if doc in query_doc_list:
                subtopic = runner.env.dictQueryDocumentSubtopics[query][doc]
                if len(subtopic) > 0:
                    flag = True
                    break
        
        if not flag:
            # print('abandon abandon abandon abandon', query)
            continue

        docs_repr = []
        for doc in random_doc_list:
            docs_repr.append(np.squeeze(np.array(runner.env.dictDocumentRepresentation[doc])).tolist())
        # print(len(docs_repr), len(docs_repr[-1]))
        # print(np.array(query_repr).shape)
        # print(query_repr)
        
        # obs = query + docs + actions + doc id(30)
        obs_repr = []
        for i in range(len(random_doc_list)):
            obs_actions = [0] * runner.env.n_agents
            obs_actions = get_onehot(obs_actions, runner.env.n_agents, runner.env.m_score)
            last_action_i = get_onehot([0], 1, runner.env.m_score)
            agent_id = get_onehot([0], 1, runner.env.n_agents)
            dot_query_doc = (np.array(query_repr) * np.array(docs_repr[i])).tolist()

            obs_doc = query_repr + docs_repr[i] + dot_query_doc #+ obs_actions # + last_action_i + agent_id
            obs_repr.append(obs_doc)
        
        # put obs tensor to device
        agent_inputs = th.tensor(np.array(obs_repr)).to(device).unsqueeze(0)#.squeeze(-1)
        # rnn hidden state inital zero
        hidden_states = runner.mac.agent.fc1.weight.new(len(random_doc_list), args.rnn_hidden_dim).zero_().unsqueeze(0)
        # get actions by mac.selection_actions()
        # print(agent_inputs.size(), hidden_states.size())
        qvals, hidden_states = runner.mac.agent(agent_inputs.float(), hidden_states.float())
        bs=slice(None)
        # to device
        qvals = qvals.to(device)
        avail_actions = th.ones(qvals.size()).to(device)
        # select actions
        chosen_actions = runner.mac.action_selector.select_action(qvals[bs], avail_actions[bs], 0, test_mode=test_mode)
        # print(runner.mac.action_selector.epsilon)
        # sort by actions
        ranks, sorted_scores = get_rank(chosen_actions[0].tolist(), random_doc_list)
        # print(ranks)
        # print('len(ranks): ', len(ranks))
        # calculate alpha-NDCG according to sorted list.
        alpha_DCG_5 = alphaDCG(runner.env.alpha, query, ranks, k=5, dictQueryDocumentSubtopics=runner.env.dictQueryDocumentSubtopics)
        alpha_DCG_10 = alphaDCG(runner.env.alpha, query, ranks, k=10, dictQueryDocumentSubtopics=runner.env.dictQueryDocumentSubtopics)
        # get all ranking positions
        ideal_ranks = ideal_doc_list
        ideal_alpha_DCG_5 = alphaDCG(runner.env.alpha, query, ideal_ranks, k=5, dictQueryDocumentSubtopics=runner.env.dictQueryDocumentSubtopics)
        ideal_alpha_DCG_10 = alphaDCG(runner.env.alpha, query, ideal_ranks, k=10, dictQueryDocumentSubtopics=runner.env.dictQueryDocumentSubtopics)

        if ideal_alpha_DCG_5 != 0:
            alpha_NDCG_5 = alpha_DCG_5 / ideal_alpha_DCG_5
        else:
            alpha_NDCG_5 = 0

        if ideal_alpha_DCG_10 != 0:
            alpha_NDCG_10 = alpha_DCG_10/ ideal_alpha_DCG_10
        else:
            alpha_NDCG_10 = 0

        alpha_NDCG_5_list.append(alpha_NDCG_5)
        alpha_NDCG_10_list.append(alpha_NDCG_10)

        # print("query: {}, alpha_NDCG_5: {}, alpha_NDCG_10: {}.".format(query, alpha_NDCG_5, alpha_NDCG_10))
    
    print('runner.t_env: {}, alpha_NDCG_5: {}.'.format(runner.t_env, np.mean(alpha_NDCG_5_list)))
    print('runner.t_env: {}, alpha_NDCG_10: {}.'.format(runner.t_env, np.mean(alpha_NDCG_10_list)))
    print('\t')

    runner.logger.log_stat("test_alpha_NDCG_5", np.mean(alpha_NDCG_5_list), runner.t_env)
    runner.logger.log_stat("test_alpha_NDCG_10", np.mean(alpha_NDCG_10_list), runner.t_env)

    return np.mean(alpha_NDCG_5_list), np.mean(alpha_NDCG_10_list)


def alphaDCG(alpha, query, docList, k, dictQueryDocumentSubtopics):
    DCG = 0.0
    subtopics = []
    for i in range(20):
        subtopics.append(0)
    for i in range(k):
        G = 0.0
        if docList[i] not in dictQueryDocumentSubtopics[query]:  
            continue
        listDocSubtopics = dictQueryDocumentSubtopics[query][docList[i]]  # ['1', '2', '3', '4']
        if len(listDocSubtopics) == 0:
                G = 0.0
        else:
            for subtopic in listDocSubtopics:
                G += (1-alpha) ** subtopics[int(subtopic)-1]
                subtopics[int(subtopic)-1] += 1
        DCG += G/math.log(i+2, 2)
    return DCG


def get_onehot(input_list, row_dim, col_dim):
    original_list = input_list
    one_hot_matrix = np.zeros((row_dim, col_dim))
    for i, val in enumerate(original_list):
        one_hot_matrix[i, val] = 1
    one_hot_list = one_hot_matrix.tolist()
    flattened_one_hot_list = [item for sublist in one_hot_list for item in sublist]

    return flattened_one_hot_list


def get_rank(scores_list, doc_ids):
    sorted_pairs = sorted(zip(doc_ids, scores_list), key=lambda x: x[1], reverse=True)
    sorted_doc_ids, sorted_scores = zip(*sorted_pairs)
    sorted_doc_ids, sorted_scores = list(sorted_doc_ids), list(sorted_scores)

    return sorted_doc_ids, sorted_scores


def evaluate_sequential(listTestSet, runner, listKeys):

    alpha_ndcg_5_list, alpha_ndcg_10_list = [], []
    rewards_list = []
    ERR_IA_5_list, ERR_IA_10_list = [], []
    S_recall_5_list, S_recall_10_list = [], []

    for i in tqdm(range(len(listTestSet))):
        query_id = listTestSet[i]

        initital_listPermutation_can = copy.deepcopy(runner.env.dictQueryPermutaion[query_id]['permutation'][:runner.env.n_agents])  

        listPermutation_can = initital_listPermutation_can
        random.shuffle(listPermutation_can)

        # test
        with th.no_grad():
            # reward, alpha_ndcg_5, alpha_ndcg_10, ERR_IA_5, ERR_IA_10 = runner.run(test_mode=True, query_id=query_id, doc_list=listPermutation_can) 
            reward, alpha_ndcg_5, alpha_ndcg_10, ERR_IA_5, ERR_IA_10, S_recall_5, S_recall_10 = runner.run(test_mode=True, query_id=query_id, doc_list=listPermutation_can)  
            alpha_ndcg_5_list.append(alpha_ndcg_5)
            alpha_ndcg_10_list.append(alpha_ndcg_10)
            rewards_list.append(reward)
            ERR_IA_5_list.append(ERR_IA_5)
            ERR_IA_10_list.append(ERR_IA_10)
            S_recall_5_list.append(S_recall_5)
            S_recall_10_list.append(S_recall_10)
            # print('testing, query_id: {}, alpha_ndcg_10: {}'.format(query_id, alpha_ndcg_10))

    runner.logger.log_stat("test_alpha_ndcg_5", np.mean(alpha_ndcg_5_list), runner.t_env)
    runner.logger.log_stat("test_alpha_ndcg_10", np.mean(alpha_ndcg_10_list), runner.t_env)
    runner.logger.log_stat("test_err_ia_5", np.mean(ERR_IA_5_list), runner.t_env)
    runner.logger.log_stat("test_err_ia_10", np.mean(ERR_IA_10_list), runner.t_env)
    runner.logger.log_stat("test_s_recall_5", np.mean(S_recall_5_list), runner.t_env)
    runner.logger.log_stat("test_s_recall_10", np.mean(S_recall_10_list), runner.t_env)
    runner.logger.log_stat("test_rewards_train_dcg_{}".format(runner.env.train_dcg_n), np.mean(rewards_list), runner.t_env)

    print('runner.t_env: {}, test_alpha_ndcg_5: {}'.format(runner.t_env, np.mean(alpha_ndcg_5_list)))
    print('runner.t_env: {}, test_alpha_ndcg_10: {}'.format(runner.t_env, np.mean(alpha_ndcg_10_list)))
    print('runner.t_env: {}, test_err_ia_5: {}'.format(runner.t_env, np.mean(ERR_IA_5_list)))
    print('runner.t_env: {}, test_err_ia_10: {}'.format(runner.t_env, np.mean(ERR_IA_10_list)))
    print('runner.t_env: {}, test_s_recall_5: {}'.format(runner.t_env, np.mean(S_recall_5_list)))
    print('runner.t_env: {}, test_s_recall_10: {}'.format(runner.t_env, np.mean(S_recall_10_list)))
    print('runner.t_env: {}, test_rewards_train_dcg_{}: {}'.format(runner.t_env, runner.env.train_dcg_n, np.mean(rewards_list)))


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # load data dictQueryDocumentSubtopics dictQueryPermutaion
    listKeys = list(runner.env.dictQueryDocumentSubtopics.keys()) 
    
    # train and test data
    sample_num = int(len(listKeys)*0.2)
    random.seed(0)
    listTestSet = random.sample(listKeys, sample_num)
    temp_train_list = []
    for query in listKeys:
        if query not in listTestSet:
            temp_train_list.append(query)
    listKeys = temp_train_list

    train_query_num = len(listKeys)
    
    print('there are {} training query.'.format(len(listKeys)))
    print('there are {} testing query.'.format(len(listTestSet)))

    args.buffer_size = train_query_num * 10
    runner.args.buffer_size = train_query_num * 10
    
    runner.args.test_nepisode = len(listTestSet)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    print('creating runner.')
    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    print('runner is created.')

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # pure test
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        print(args.checkpoint_path)
        print(timesteps)

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        multi_episode_test_alpha_ndcg_5 = []
        multi_episode_test_alpha_ndcg_10 = []
        for _ in range(1):
            if args.evaluate:

                # temp_alpha_ndcg_5, temp_alpha_ndcg_10 = all_doc_evaluate(args, runner, test_mode=True, listTestSet=listTestSet)
                # multi_episode_test_alpha_ndcg_5.append(temp_alpha_ndcg_5)
                # multi_episode_test_alpha_ndcg_10.append(temp_alpha_ndcg_10)

                evaluate_sequential(listTestSet, runner, listKeys)

        # print('aver_test_alpha_ndcg_5: {}, aver_test_alpha_ndcg_10: {}'.format(np.mean(multi_episode_test_alpha_ndcg_5), np.mean(multi_episode_test_alpha_ndcg_10)))
        
        return
        

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_train_T = 0
    total_train_T = 0
    last_log_T = 0

    cur_max_ndcg_10 = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))  


    # test once before train
    if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

        print('test once before train.')

        last_test_T = runner.t_env
        
        alpha_ndcg_5_list, alpha_ndcg_10_list = [], []
        rewards_list = []
        ERR_IA_5_list, ERR_IA_10_list = [], []
        S_recall_5_list, S_recall_10_list = [], []

        for i in tqdm(range(len(listTestSet))):
            query_id = listTestSet[i]

            initital_listPermutation_can = copy.deepcopy(runner.env.dictQueryPermutaion[query_id]['permutation'][:runner.env.n_agents]) 

            listPermutation_can = initital_listPermutation_can
            random.shuffle(listPermutation_can)

            # test
            with th.no_grad():

                reward, alpha_ndcg_5, alpha_ndcg_10, ERR_IA_5, ERR_IA_10, S_recall_5, S_recall_10 = runner.run(test_mode=True, query_id=query_id, doc_list=listPermutation_can) 
                alpha_ndcg_5_list.append(alpha_ndcg_5)
                alpha_ndcg_10_list.append(alpha_ndcg_10)
                rewards_list.append(reward)
                ERR_IA_5_list.append(ERR_IA_5)
                ERR_IA_10_list.append(ERR_IA_10)
                S_recall_5_list.append(S_recall_5)
                S_recall_10_list.append(S_recall_10)

        runner.logger.log_stat("test_alpha_ndcg_5", np.mean(alpha_ndcg_5_list), runner.t_env)
        runner.logger.log_stat("test_alpha_ndcg_10", np.mean(alpha_ndcg_10_list), runner.t_env)
        runner.logger.log_stat("test_err_ia_5", np.mean(ERR_IA_5_list), runner.t_env)
        runner.logger.log_stat("test_err_ia_10", np.mean(ERR_IA_10_list), runner.t_env)
        runner.logger.log_stat("test_s_recall_5", np.mean(S_recall_5_list), runner.t_env)
        runner.logger.log_stat("test_s_recall_10", np.mean(S_recall_10_list), runner.t_env)
        runner.logger.log_stat("test_rewards_train_dcg_{}".format(runner.env.train_dcg_n), np.mean(rewards_list), runner.t_env)

        print('runner.t_env: {}, test_alpha_ndcg_5: {}'.format(runner.t_env, np.mean(alpha_ndcg_5_list)))
        print('runner.t_env: {}, test_alpha_ndcg_10: {}'.format(runner.t_env, np.mean(alpha_ndcg_10_list)))
        print('runner.t_env: {}, test_err_ia_5: {}'.format(runner.t_env, np.mean(ERR_IA_5_list)))
        print('runner.t_env: {}, test_err_ia_10: {}'.format(runner.t_env, np.mean(ERR_IA_10_list)))
        print('runner.t_env: {}, test_s_recall_5: {}'.format(runner.t_env, np.mean(S_recall_5_list)))
        print('runner.t_env: {}, test_s_recall_10: {}'.format(runner.t_env, np.mean(S_recall_10_list)))
        print('runner.t_env: {}, test_rewards_train_dcg_{}: {}'.format(runner.t_env, runner.env.train_dcg_n, np.mean(rewards_list)))


    # train
    print('begin to train.')
    for epoch in range(501):

        print('train epoch: {}'.format(epoch))
        
        count = 0
        for i in tqdm(range(len(listKeys))):
            query_id = listKeys[i]
            
            initital_listPermutation_can = copy.deepcopy(runner.env.dictQueryPermutaion[query_id]['permutation'][:runner.env.n_agents]) 

            listPermutation_can = initital_listPermutation_can
            random.shuffle(listPermutation_can)

            # Run for a episode at a time
            with th.no_grad():
                episode_batch = runner.run(test_mode=False, query_id=query_id, doc_list=listPermutation_can) 
                buffer.insert_episode_batch(episode_batch)
            
            if runner.t_env - last_train_T >= train_query_num * runner.env.episode_limit and buffer.can_sample(args.batch_size): 

                last_train_T = runner.t_env

                print('sample batch size {} from buffer size {}, train at epoch {} for {} times.'.format(args.batch_size, args.buffer_size, epoch, int(train_query_num / args.batch_size)*10))
                for _ in range(int(train_query_num / args.batch_size)*10):

                    # total_train_T += 1
                    
                    next_episode = episode + args.batch_size_run
                    if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                        continue

                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)

                    del episode_sample

                if True:

                    print('testing at epoch: {}'.format(epoch))

                    last_test_T = runner.t_env
                    
                    alpha_ndcg_5_list, alpha_ndcg_10_list = [], []
                    rewards_list = []
                    ERR_IA_5_list, ERR_IA_10_list = [], []
                    S_recall_5_list, S_recall_10_list = [], []

                    for i in tqdm(range(len(listTestSet))):
                        query_id = listTestSet[i]

                        initital_listPermutation_can = copy.deepcopy(runner.env.dictQueryPermutaion[query_id]['permutation'][:runner.env.n_agents])

                        listPermutation_can = initital_listPermutation_can
                        random.shuffle(listPermutation_can)

                        # test
                        with th.no_grad():
                            reward, alpha_ndcg_5, alpha_ndcg_10, ERR_IA_5, ERR_IA_10, S_recall_5, S_recall_10 = runner.run(test_mode=True, query_id=query_id, doc_list=listPermutation_can)
                            alpha_ndcg_5_list.append(alpha_ndcg_5)
                            alpha_ndcg_10_list.append(alpha_ndcg_10)
                            rewards_list.append(reward)
                            ERR_IA_5_list.append(ERR_IA_5)
                            ERR_IA_10_list.append(ERR_IA_10)
                            S_recall_5_list.append(S_recall_5)
                            S_recall_10_list.append(S_recall_10)

                    runner.logger.log_stat("test_alpha_ndcg_5", np.mean(alpha_ndcg_5_list), runner.t_env)
                    runner.logger.log_stat("test_alpha_ndcg_10", np.mean(alpha_ndcg_10_list), runner.t_env)
                    runner.logger.log_stat("test_err_ia_5", np.mean(ERR_IA_5_list), runner.t_env)
                    runner.logger.log_stat("test_err_ia_10", np.mean(ERR_IA_10_list), runner.t_env)
                    runner.logger.log_stat("test_s_recall_5", np.mean(S_recall_5_list), runner.t_env)
                    runner.logger.log_stat("test_s_recall_10", np.mean(S_recall_10_list), runner.t_env)
                    runner.logger.log_stat("test_rewards_train_dcg_{}".format(runner.env.train_dcg_n), np.mean(rewards_list), runner.t_env)

                    print('runner.t_env: {}, test_alpha_ndcg_5: {}'.format(runner.t_env, np.mean(alpha_ndcg_5_list)))
                    print('runner.t_env: {}, test_alpha_ndcg_10: {}'.format(runner.t_env, np.mean(alpha_ndcg_10_list)))
                    print('runner.t_env: {}, test_err_ia_5: {}'.format(runner.t_env, np.mean(ERR_IA_5_list)))
                    print('runner.t_env: {}, test_err_ia_10: {}'.format(runner.t_env, np.mean(ERR_IA_10_list)))
                    print('runner.t_env: {}, test_s_recall_5: {}'.format(runner.t_env, np.mean(S_recall_5_list)))
                    print('runner.t_env: {}, test_s_recall_10: {}'.format(runner.t_env, np.mean(S_recall_10_list)))
                    print('runner.t_env: {}, test_rewards_train_dcg_{}: {}'.format(runner.t_env, runner.env.train_dcg_n, np.mean(rewards_list)))

        # save model
        if epoch % 20 == 0:

            if args.save_model:
                save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
                #"results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env
    
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
