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
            subtopics_set = set(subtopics_list)
            self.query_subtopics[query_id] = len(subtopics_set)  # {'66': 3, '133': 5, ...}

        self.dictDCG = copy.deepcopy(self.dictQueryPermutaion)
        for key, value in self.dictDCG.items():
            temp_dic = {}
            ideal_docs = value['permutation']
            all_num = len(ideal_docs)
            for i in range(all_num):
                doc = ideal_docs[i]
                temp_dic[doc] = all_num - i
            self.dictDCG[key] = temp_dic
        
        # agent number
        self.n_agents = N_agent

        # {0, 1, 2, 3, ..., M_score-1}
        self.m_score = M_score

        # observations and state
        self.query_doc_dims = 1024
        self.obs_size = self.query_doc_dims + self.query_doc_dims
        self.state_size = self.query_doc_dims + self.n_agents * self.query_doc_dims

        # action
        self.n_actions = self.m_score

        # others
        self.episode_limit = 1
        self.all_step = 0
        self.alpha = alpha
        self.last_doclist = ['999'] * self.n_actions

        self.train_dcg_n = self.n_agents
        self.query_features = []
        self.docs_features = []

        self.last_alpha_NDCG = 0

    def get_rank(self, scores_list, doc_ids):
        sorted_pairs = sorted(zip(doc_ids, scores_list), key=lambda x: x[1], reverse=True)

        sorted_doc_ids, sorted_scores = zip(*sorted_pairs)

        sorted_doc_ids = list(sorted_doc_ids)

        return sorted_doc_ids
    
    def get_query_docs_features(self, query_id, doclist, test_mode):

        self.query_features = self.dictQueryRepresentation[query_id]

        self.docs_features = []
        for doc_id in doclist:
            doc_repr = self.dictDocumentRepresentation[doc_id]  # np.array(self.dictQueryDocumentRepresentation[doc_id]).reshape(-1).tolist()
            self.docs_features.extend(doc_repr)

    def step(self, actions, query, doclist, t, test_mode, sparse_reward):
        """ Returns reward, terminated, info """
        # docList = self.get_can_doclist(actions)
        ranks = self.get_rank(actions.tolist(), doclist)
        if test_mode:
            alpha_DCG_5 = self.alphaDCG(self.alpha, query, ranks, k=5)
            alpha_DCG_10 = self.alphaDCG(self.alpha, query, ranks, k=10)
            ERR_IA_5 = self.expected_reciprocal_rank(query, ranks, k=5)
            ERR_IA_10 = self.expected_reciprocal_rank(query, ranks, k=10)
            S_recall_5 = self.subtopic_recall(query, ranks, k=5)
            S_recall_10 = self.subtopic_recall(query, ranks, k=10)
        alpha_DCG = self.alphaDCG(self.alpha, query, ranks, k=self.train_dcg_n)
        alpha_DCG_add = self.alphaDCG(self.alpha, query, ranks, k=5)

        # print('------------------')
        # print('t: {}'.format(t))

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
        if ideal_alpha_DCG_add != 0:
            cur_alpha_NDCG_add = alpha_DCG_add / ideal_alpha_DCG_add
        else:
            cur_alpha_NDCG_add = 0


        # reward setting
        # reward = cur_alpha_NDCG
        if t == 0:
            self.last_alpha_NDCG = cur_alpha_NDCG
        
        if t < self.episode_limit - 1:
            reward = cur_alpha_NDCG - self.last_alpha_NDCG
            self.last_alpha_NDCG = cur_alpha_NDCG
        else:
            reward = cur_alpha_NDCG + cur_alpha_NDCG_add

        if t >= self.episode_limit-1:
            terminated = True 
        else:
            terminated = False

        info = {}

        # if sparse_reward:  # sparse reward
        #     if not terminated: 
        #         reward = 0
        # else:
        #     pass

        if test_mode:
            return alpha_NDCG_5, alpha_NDCG_10, ERR_IA_5, ERR_IA_10, S_recall_5, S_recall_10, reward, terminated, info, ranks
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
        doc_repr = self.dictDocumentRepresentation[doc_id]
        return self.query_features + self.docs_features + self.query_features + doc_repr
    
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
        doc_repr = self.dictDocumentRepresentation[doc_id]
        return self.query_features + self.docs_features + self.query_features + doc_repr

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.state_size + self.obs_size
    
    def get_state(self, actions, time_step):
        return self.query_features + self.docs_features

    def get_state_size(self):
        """ Returns the shape of the state"""
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
            if docList[i] not in self.dictQueryDocumentSubtopics[query]:  
                continue
            listDocSubtopics = self.dictQueryDocumentSubtopics[query][docList[i]]  # ['1', '2', '3', '4']
            # print('subtopics: ', docList[i], listDocSubtopics)
            if len(listDocSubtopics) == 0:  
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

    def subtopic_recall(self, query, docList, k):
        n = self.query_subtopics[query]
        subtopics_r = []
        for d in docList[:k]:
            if d in self.dictQueryDocumentSubtopics[query]:
            # if self.dictQueryDocumentSubtopics[query].has_key(d):
                subtopics_r.extend(self.dictQueryDocumentSubtopics[query][d])
        return len(set(subtopics_r))*1.0 / n
