# Part of this file is derived from 
# https://github.com/albertyang33/FOREST
"""
Created on Mon Jan 18 22:28:02 2021

@author: Ling Sun
"""
import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import pickle

class Options(object):

    def __init__(self, data_name='douban'):
        self.data = 'data/' + data_name + '/cascades.txt'
        self.u2idx_dict = 'data/' + data_name + '/u2idx.pickle'
        self.idx2u_dict = 'data/' + data_name + '/idx2u.pickle'
        self.save_path = ''
        self.net_data = 'data/' + data_name + '/edges.txt'
        self.embed_dim = 64

        # 资源难度数据用于适应性评价指标计算
        self.difficult_file = 'data/' + data_name + '/difficulty.csv'


def Split_data(data_name, train_rate=0.8, valid_rate=0.1, random_seed=300, load_dict=True, with_EOS=True):
    options = Options(data_name)
    u2idx = {}
    idx2u = []
    if not load_dict:
        resource_size, u2idx, idx2u = buildIndex(options.data)
        with open(options.u2idx_dict, 'wb') as handle:
            pickle.dump(u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options.idx2u_dict, 'wb') as handle:
            pickle.dump(idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(options.u2idx_dict, 'rb') as handle:
            u2idx = pickle.load(handle)
        with open(options.idx2u_dict, 'rb') as handle:
            idx2u = pickle.load(handle)
        resource_size = len(u2idx)

    t_cascades = []
    timestamps = []
    answers = []
    # 新增：存储其他字段
    end_times = []
    answer_opens = []
    retry_statuses = []
    evaluate_counts = []

    for line in open(options.data):
        if len(line.strip()) == 0:
            continue
        timestamplist = []
        userlist = []
        answerlist = []
        end_timelist = []
        answer_openlist = []
        retry_statuslist = []
        evaluate_countlist = []

        chunks = line.strip().split(',')
        for chunk in chunks:
            try:
                parts = chunk.split()
                if len(parts) == 7:
                    # 7个字段: challenge_id, open_time, 是否正确, end_time, answer_open, retry_status, evaluate_count
                    challenge_id, open_time, is_correct, end_time, answer_open, retry_status, evaluate_count = parts
                    user = challenge_id
                    timestamp = open_time
                    answer = is_correct
                elif len(parts) == 3:
                    user, timestamp, answer = parts
                    # 兼容3字段格式，其他字段设默认值
                    end_time = '0'
                    answer_open = '0'
                    retry_status = '0'
                    evaluate_count = '0'
                elif len(parts) == 2:
                    # 兼容只有 user 和 timestamp 的情况（默认 answer 为 1）
                    user, timestamp = parts
                    answer = '1'
                    end_time = '0'
                    answer_open = '0'
                    retry_status = '0'
                    evaluate_count = '0'
                else:
                    # 格式不正确，跳过这个 chunk
                    print(f"Warning: Skipping malformed chunk: {chunk}")
                    continue

                if user in u2idx:
                    userlist.append(u2idx[user])
                    timestamplist.append(float(timestamp))
                    answerlist.append(float(answer))
                    end_timelist.append(float(end_time))
                    answer_openlist.append(float(answer_open))
                    retry_statuslist.append(float(retry_status))
                    evaluate_countlist.append(float(evaluate_count))
            except Exception as e:
                print(f"Error processing chunk '{chunk}': {e}")
                continue

        if len(userlist) > 1 and len(userlist) <= 500:
            if with_EOS:
                userlist.append(Constants.EOS)
                timestamplist.append(Constants.EOS)
                answerlist.append(Constants.EOS)
                end_timelist.append(Constants.EOS)
                answer_openlist.append(Constants.EOS)
                retry_statuslist.append(Constants.EOS)
                evaluate_countlist.append(Constants.EOS)
            t_cascades.append(userlist)
            timestamps.append(timestamplist)
            answers.append(answerlist)
            end_times.append(end_timelist)
            answer_opens.append(answer_openlist)
            retry_statuses.append(retry_statuslist)
            evaluate_counts.append(evaluate_countlist)

    '''ordered by timestamps'''
    order = [i[0] for i in sorted(enumerate(timestamps), key=lambda x: x[1])]
    timestamps = sorted(timestamps)
    t_cascades[:] = [t_cascades[i] for i in order]
    answers[:] = [answers[i] for i in order]
    cas_idx = [i for i in range(len(t_cascades))]

    '''data split'''
    train_idx_ = int(train_rate * len(t_cascades))
    # ======== 【新增：保留打乱前的纯历史序列，专门用于建图】 ========
    train_history_cas = t_cascades[0:train_idx_]
    train_history_t = timestamps[0:train_idx_]
    # ==========================================================
    train = t_cascades[0:train_idx_]
    train_t = timestamps[0:train_idx_]
    train_idx = cas_idx[0:train_idx_]
    train_ans = answers[0:train_idx_]
    train_end_time = end_times[0:train_idx_]
    train_answer_open = answer_opens[0:train_idx_]
    train_retry_status = retry_statuses[0:train_idx_]
    train_evaluate_count = evaluate_counts[0:train_idx_]

    n = len(train)
    indices = list(range(n))
    random.shuffle(indices)  # 生成随机索引序列
    # --- 按索引重新排列所有列表 ---
    train = [train[i] for i in indices]
    train_t = [train_t[i] for i in indices]
    train_idx = [train_idx[i] for i in indices]
    train_ans = [train_ans[i] for i in indices]
    train_end_time = [train_end_time[i] for i in indices]
    train_answer_open = [train_answer_open[i] for i in indices]
    train_retry_status = [train_retry_status[i] for i in indices]
    train_evaluate_count = [train_evaluate_count[i] for i in indices]

    train = [train, train_t, train_idx, train_ans, train_end_time, train_answer_open, train_retry_status, train_evaluate_count]

    valid_idx_ = int((train_rate + valid_rate) * len(t_cascades))
    valid = t_cascades[train_idx_:valid_idx_]
    valid_t = timestamps[train_idx_:valid_idx_]
    valid_idx = cas_idx[train_idx_:valid_idx_]
    valid_ans = answers[train_idx_:valid_idx_]
    valid_end_time = end_times[train_idx_:valid_idx_]
    valid_answer_open = answer_opens[train_idx_:valid_idx_]
    valid_retry_status = retry_statuses[train_idx_:valid_idx_]
    valid_evaluate_count = evaluate_counts[train_idx_:valid_idx_]
    valid = [valid, valid_t, valid_idx, valid_ans, valid_end_time, valid_answer_open, valid_retry_status, valid_evaluate_count]

    test = t_cascades[valid_idx_:]
    test_t = timestamps[valid_idx_:]
    test_idx = cas_idx[valid_idx_:]
    test_ans = answers[valid_idx_:]
    test_end_time = end_times[valid_idx_:]
    test_answer_open = answer_opens[valid_idx_:]
    test_retry_status = retry_statuses[valid_idx_:]
    test_evaluate_count = evaluate_counts[valid_idx_:]
    test = [test, test_t, test_idx, test_ans, test_end_time, test_answer_open, test_retry_status, test_evaluate_count]

    total_len = sum(len(i) - 1 for i in t_cascades)
    train_size = len(train_t)
    valid_size = len(valid_t)
    test_size = len(test_t)
    print("training size:%d\n   valid size:%d\n  testing size:%d" % (train_size, valid_size, test_size))
    print("total size:%d " % (len(t_cascades)))
    print("average length:%f" % (total_len / len(t_cascades)))
    print('maximum length:%f' % (max(len(cas) for cas in t_cascades)))
    print('minimum length:%f' % (min(len(cas) for cas in t_cascades)))
    print("resource size:%d" % (resource_size - 2))

    return resource_size, train_history_cas, train_history_t, train, valid, test


def buildIndex(data):
    user_set = set()
    u2idx = {}
    idx2u = []

    lineid = 0
    for line in open(data):
        lineid += 1
        if len(line.strip()) == 0:
            continue
        chunks = line.strip().split(',')
        for chunk in chunks:
            try:
                parts = chunk.split()
                if len(parts) == 7:
                    # 7个字段: challenge_id, open_time, 是否正确, end_time, answer_open, retry_status, evaluate_count
                    challenge_id = parts[0]
                    user = challenge_id
                elif len(parts) == 3:
                    user = parts[0]
                elif len(parts) == 2:
                    user = parts[0]
            except:
                print(line)
                print(chunk)
                print(lineid)
            user_set.add(user)
    pos = 0
    u2idx['<blank>'] = pos
    idx2u.append('<blank>')
    pos += 1
    u2idx['</s>'] = pos
    idx2u.append('</s>')
    pos += 1

    for user in user_set:
        u2idx[user] = pos
        idx2u.append(user)
        pos += 1
    resource_size = len(user_set) + 2
    print("resource_size : %d" % (resource_size))
    return resource_size, u2idx, idx2u


class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, cas, batch_size=64, load_dict=True, cuda=True, test=False, with_EOS=True):
        self._batch_size = batch_size
        self.cas = cas[0]           # challenge_id 序列
        self.time = cas[1]          # open_time 序列
        self.idx = cas[2]           # 序列索引
        self.ans = cas[3]           # is_correct 序列
        # 新增字段
        self.end_time = cas[4] if len(cas) > 4 else None          # end_time 序列
        self.answer_open = cas[5] if len(cas) > 5 else None       # answer_open 序列
        self.retry_status = cas[6] if len(cas) > 6 else None      # retry_status 序列
        self.evaluate_count = cas[7] if len(cas) > 7 else None    # evaluate_count 序列

        self.test = test
        self.with_EOS = with_EOS
        self.cuda = cuda

        self._n_batch = int(np.ceil(len(self.cas) / self._batch_size))
        self._iter_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = 200

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst)) if len(inst) < max_len else inst[:max_len]
                for inst in insts])

            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            seq_insts = self.cas[start_idx:end_idx]
            seq_timestamp = self.time[start_idx:end_idx]
            seq_answer = self.ans[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)
            seq_data_timestamp = pad_to_longest(seq_timestamp)
            seq_data_answer = pad_to_longest(seq_answer)
            seq_idx = Variable(
                torch.LongTensor(self.idx[start_idx:end_idx]), volatile=self.test)

            # 新增字段的处理
            if self.end_time is not None:
                seq_end_time = self.end_time[start_idx:end_idx]
                seq_data_end_time = pad_to_longest(seq_end_time)
            else:
                seq_data_end_time = None

            if self.answer_open is not None:
                seq_answer_open = self.answer_open[start_idx:end_idx]
                seq_data_answer_open = pad_to_longest(seq_answer_open)
            else:
                seq_data_answer_open = None

            if self.retry_status is not None:
                seq_retry_status = self.retry_status[start_idx:end_idx]
                seq_data_retry_status = pad_to_longest(seq_retry_status)
            else:
                seq_data_retry_status = None

            if self.evaluate_count is not None:
                seq_evaluate_count = self.evaluate_count[start_idx:end_idx]
                seq_data_evaluate_count = pad_to_longest(seq_evaluate_count)
            else:
                seq_data_evaluate_count = None

            return seq_data, seq_data_timestamp, seq_idx, seq_data_answer, seq_data_end_time, seq_data_answer_open, seq_data_retry_status, seq_data_evaluate_count
        else:

            self._iter_count = 0
            raise StopIteration()
