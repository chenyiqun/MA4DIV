from envs.multiagentenv import MultiAgentEnv
import numpy as np
import copy
import math

class Search(MultiAgentEnv):

    def __init__(self, dictQueryPermutaion, dictQueryRepresentation, dictDocumentRepresentation, dictQueryDocumentSubtopics, 
                 N_agent=30, M_score=10, alpha=0.5, map_name='search_engine', seed=0):

        self.dictQueryPermutaion = dictQueryPermutaion
        self.dictQueryRepresentation = dictQueryRepresentation
        self.dictDocumentRepresentation = dictDocumentRepresentation
        self.dictQueryDocumentSubtopics = dictQueryDocumentSubtopics

        self.query_subtopics = {}
        for query_id, v in self.dictQueryDocumentSubtopics.items():
            subtopics_list = []
            for doc_id, sub in v.items():
                subtopics_list.extend(sub)
            subtopics_set = set(subtopics_list)  # 去重
            self.query_subtopics[query_id] = len(subtopics_set)  # {'66': 3, '133': 5, ...}

        # # 创建一个文件
        # with open("subtopics_permutation.txt", "w") as file:
        #     pass
        
        # for key, value in self.dictQueryPermutaion.items():
        #     sub_docs = self.dictQueryDocumentSubtopics[key].keys()
        #     for doc in value['permutation']:
        #         if doc not in sub_docs:
        #             with open("subtopics_permutation.txt", "a") as file:
        #                 file.write(key + ' ' + doc + ' 不存在。' + "\n")
        #             # print(key + ' ' + doc + ' 不存在。')
        #         else:
        #             with open("subtopics_permutation.txt", "a") as file:
        #                 file.write(key + ' ' + doc + ' ' + str(len(self.dictQueryDocumentSubtopics[key][doc])) + "\n")
        #             # print(key, doc, len(self.dictQueryDocumentSubtopics[key][doc]))

        self.dictDCG = copy.deepcopy(self.dictQueryPermutaion)
        for key, value in self.dictDCG.items():
            temp_dic = {}
            ideal_docs = value['permutation']
            all_num = len(ideal_docs)
            for i in range(all_num):
                doc = ideal_docs[i]
                temp_dic[doc] = all_num - i
            self.dictDCG[key] = temp_dic
        #     for doc in ideal_docs:
        #         temp_dic[doc]
        #         self.dictDCG[key]['ideal_rank'] = list(range(len(self.dictDCG[key]['permutation'])))
        # del self.dictDCG['permutation']

        # 智能体相关
        self.n_agents = N_agent

        # 打分跨度，即{0, 1, 2, 3, ..., M_score-1}
        self.m_score = M_score  # 此参数作为超参，可等于2倍N_agent，后期可调参。越细化平分也许越少，排序越精确。

        # observations and state
        self.query_doc_dims = 1024
        # self.obs_size = self.query_dims + self.n_agents * self.doc_dims # + self.n_agents * self.m_score + self.n_agents * self.n_agents  # query + N docs + N actions + N ranks
        self.obs_size = self.query_doc_dims + self.query_doc_dims # + self.n_agents * self.m_score + self.n_agents * self.n_agents  # query + docs + N actions + N ranks
        self.state_size = self.query_doc_dims + self.n_agents * self.query_doc_dims  # query + N docs

        # action
        self.n_actions = self.m_score  # 动作空间维度即打分跨度。

        # 其他
        self.episode_limit = 1
        self.all_step = 0
        self.alpha = alpha
        self.last_doclist = ['999'] * self.n_actions

        self.train_dcg_n = self.n_agents
        self.query_features = []
        self.docs_features = []

        self.last_alpha_NDCG = 0

    def get_rank(self, scores_list, doc_ids):
        # 假设 doc_ids 是你的文档 ID 列表，scores 是 PyTorch tensor 存储的得分
        # 使用 zip 打包 doc_ids 和 scores_list，然后使用 sorted 函数进行排序
        sorted_pairs = sorted(zip(doc_ids, scores_list), key=lambda x: x[1], reverse=True)

        # 解压排序后的 pairs，获取排序后的 doc_ids
        sorted_doc_ids, sorted_scores = zip(*sorted_pairs)

        # 转换为 list
        sorted_doc_ids = list(sorted_doc_ids)

        return sorted_doc_ids
    
    def get_query_docs_features(self, query_id, doclist, test_mode):

        # if test_mode:
        #     self.query_features = np.array(self.dictQueryRepresentation[query_id]).reshape(-1).tolist()
        # else:
        #     self.query_features = (np.array(self.dictQueryRepresentation[query_id]).reshape(-1) + np.random.normal(0, 0.0005, self.query_dims)).tolist()

        # self.query_features = np.array(self.dictQueryRepresentation[query_id]).reshape(-1).tolist()

        self.query_features = self.dictQueryRepresentation[query_id]

        self.docs_features = []
        for doc_id in doclist:
            doc_repr = self.dictDocumentRepresentation[doc_id]  # np.array(self.dictQueryDocumentRepresentation[doc_id]).reshape(-1).tolist()
            self.docs_features.extend(doc_repr)

    def step(self, actions, query, doclist, t, test_mode, sparse_reward):
        """ Returns reward, terminated, info """
        # # 此处还需再加代码，根据给出的信息得到排序列表，以便计算reward。函数的doclist只是暂时占位，后续需修改。
        # docList = self.get_can_doclist(actions)
        ranks = self.get_rank(actions.tolist(), doclist)
        if test_mode:
            alpha_DCG_5 = self.alphaDCG(self.alpha, query, ranks, k=5)
            alpha_DCG_10 = self.alphaDCG(self.alpha, query, ranks, k=10)
            ERR_IA_5 = self.expected_reciprocal_rank(query, ranks, k=5)
            ERR_IA_10 = self.expected_reciprocal_rank(query, ranks, k=10)
        alpha_DCG = self.alphaDCG(self.alpha, query, ranks, k=self.train_dcg_n)
        alpha_DCG_add = self.alphaDCG(self.alpha, query, ranks, k=5)

        # print('------------------')
        # print('t: {}'.format(t))

        # 得到所有ranks在理想排名中的位置（变向的score）
        ideal_scores = []
        for can_doc in doclist:
            ideal_scores.append(self.dictDCG[query][can_doc])
        ideal_ranks = self.get_rank(ideal_scores, doclist)

        if test_mode:
            ideal_alpha_DCG_5 = self.alphaDCG(self.alpha, query, ideal_ranks, k=5)
            ideal_alpha_DCG_10 = self.alphaDCG(self.alpha, query, ideal_ranks, k=10)
        ideal_alpha_DCG = self.alphaDCG(self.alpha, query, ideal_ranks, k=self.train_dcg_n)
        ideal_alpha_DCG_add = self.alphaDCG(self.alpha, query, ideal_ranks, k=5)

        if test_mode:
            if ideal_alpha_DCG_5 != 0:
                alpha_NDCG_5 = alpha_DCG_5 / ideal_alpha_DCG_5
            else:
                alpha_NDCG_5 = 0

            if ideal_alpha_DCG_10 != 0:
                alpha_NDCG_10 = alpha_DCG_10/ ideal_alpha_DCG_10
            else:
                alpha_NDCG_10 = 0
        
        # if ideal_alpha_DCG != 0:
        #     reward = alpha_DCG / ideal_alpha_DCG
        # else:
        #     reward = 0
        if ideal_alpha_DCG != 0:
            cur_alpha_NDCG = alpha_DCG / ideal_alpha_DCG
        else:
            cur_alpha_NDCG = 0
        # if ideal_alpha_DCG_add != 0:
        #     cur_alpha_NDCG_add = alpha_DCG_add / ideal_alpha_DCG_add
        # else:
        #     cur_alpha_NDCG_add = 0


        # reward设置
        # reward = cur_alpha_NDCG
        if t == 0:
            self.last_alpha_NDCG = cur_alpha_NDCG
        
        if t < self.episode_limit - 1:
            reward = cur_alpha_NDCG - self.last_alpha_NDCG
            self.last_alpha_NDCG = cur_alpha_NDCG
        else:
            reward = cur_alpha_NDCG# + cur_alpha_NDCG_add

        if t >= self.episode_limit-1:
            terminated = True 
        else:
            terminated = False

        info = {}

        # if sparse_reward:  # 稀疏奖励
        #     if not terminated:  # 非终结step reward设为0
        #         reward = 0
        # else:
        #     pass

        if test_mode:
            return alpha_NDCG_5, alpha_NDCG_10, ERR_IA_5, ERR_IA_10, reward, terminated, info, ranks
        else:
            return reward, terminated, info, ranks
    
    def reset(self, query, doclist, actions, test_mode, time_step):
        """ Returns initial observations and states"""
        self.steps = 0
        self.get_query_docs_features(query, doclist, test_mode)
        return self.get_obs(doclist, actions, time_step), self.get_state(actions, time_step)
    
    def get_obs(self, doclist, actions, time_step):
        """Returns all agent observations in a list."""
        agents_obs = [self.get_obs_agent(doclist[i], actions, time_step) for i in range(self.n_agents)]
        return agents_obs
    
    def get_obs_agent(self, doc_id, actions, time_step):
        """ Returns observation for agent_id """
        # # # query + N docs + N actions + N ranks
        # query_repr = np.array(self.dictQueryRepresentation[query_id]).reshape(-1)
        # query_repr = np.array(self.query_features)
        # doc_repr = np.array(self.dictDocumentRepresentation[doc_id]).reshape(-1)
        # agent_obs = self.query_features + self.docs_features + actions
        # dot_query_doc = query_repr * doc_repr

        # 以下是加入att的
        doc_repr = self.dictDocumentRepresentation[doc_id]
        return self.query_features + self.docs_features + self.query_features + doc_repr
    
        # # 以下是没有加入att的
        # doc_repr = self.dictDocumentRepresentation[doc_id]
        # return self.query_features + doc_repr

    def get_obs_size(self):
        """ Returns the shape of the observation """
        # # 以下是加入att的
        # return self.state_size + self.obs_size + self.episode_limit  #  + self.n_agents * self.n_actions 

        # 以下是没有加入att的
        return self.state_size + self.obs_size
    
    def get_state(self, actions, time_step):
        # query_repr = np.array(self.dictQueryRepresentation[query_id]).reshape(-1).tolist()
        # doc_repr_list = []
        # for doc_id in can_doclist:
        #     doc_repr = np.array(self.dictDocumentRepresentation[doc_id]).reshape(-1).tolist()
        #     doc_repr_list.extend(doc_repr)
        # state = query_repr + doc_repr_list

        # # 以下是加入att的
        # one_hot_t = [0] * self.episode_limit
        # one_hot_t[time_step-1] = 1

        # return self.query_docs_features + actions + one_hot_t

        # 以下是没有加入att的
        return self.query_features + self.docs_features

    def get_state_size(self):
        """ Returns the shape of the state"""
        # # 以下是加入att的
        # return self.state_size + self.n_agents * self.n_actions + self.episode_limit
        
        # 以下是没有加入att的
        return self.state_size

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
    
    def alphaDCG(self, alpha, query, docList, k):
        DCG = 0.0
        subtopics = []
        for i in range(50):
            subtopics.append(0)
        for i in range(k):
            G = 0.0
            if docList[i] not in self.dictQueryDocumentSubtopics[query]:  # 候选文档（216）中没有文档i，不计算。
                # print('有subtopic标签的文档中，query.{}没有doc {}'.format(query, docList[i]))
                continue
            listDocSubtopics = self.dictQueryDocumentSubtopics[query][docList[i]]  # ['1', '2', '3', '4']
            # print('subtopics: ', docList[i], listDocSubtopics)
            if len(listDocSubtopics) == 0:  # 如果此文档不包含任何子话题，则对于i=k处的rank，G=0。
                    G = 0.0
            else:
                for subtopic in listDocSubtopics:
                    G += (1-alpha) ** subtopics[int(subtopic)-1]
                    subtopics[int(subtopic)-1] += 1
            DCG += G/math.log(i+2, 2)
        return DCG
        
    def expected_reciprocal_rank(self, query, docList, k):
        n = self.query_subtopics[query]
        all_doc = len(self.dictQueryPermutaion[query]['permutation'])
        p_topic = [0.0] * n
        topic_map = {}
        for d in self.dictQueryPermutaion[query]['permutation']:
            if d in self.dictQueryDocumentSubtopics[query]:
            # if self.dictQueryDocumentSubtopics[query].has_key(d):
                for doc_topic in self.dictQueryDocumentSubtopics[query][d]:
                    if doc_topic in topic_map:
                    # if topic_map.has_key(doc_topic):
                        p_topic[topic_map[doc_topic]] += 1
                    else:
                        topic_map[doc_topic] = len(topic_map)
                        p_topic[topic_map[doc_topic]] += 1
        err = 0.0
        for id_n, d in enumerate(docList[:k]):
            all_topic = 0.0
            for topic_name, id_t in topic_map.items():
                score = 1.0
                for selected_doc in docList[:id_n]:
                    r = 0.0
                    if selected_doc in self.dictQueryDocumentSubtopics[query]:
                    # if self.dictQueryDocumentSubtopics[query].has_key(selected_doc):
                        for doc_t in self.dictQueryDocumentSubtopics[query][selected_doc]:
                            if doc_t == topic_name:
                                r = (2.0**1 - 1) / 2**1
                    score *= (1-r)
                r = 0.0
                if docList[id_n] in self.dictQueryDocumentSubtopics[query]:
                # if self.dictQueryDocumentSubtopics[query].has_key(docList[id_n]):
                    for doc_t in self.dictQueryDocumentSubtopics[query][docList[id_n]]:
                        if doc_t == topic_name:
                            r = (2.0**1 - 1) / 2**1
                score *= r
                all_topic += p_topic[id_t] / all_doc * score
            err += 1.0 / (id_n+1) * all_topic

        return err