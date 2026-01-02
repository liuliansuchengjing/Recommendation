import numpy as np
import json
import pickle
import csv
import random
from collections import defaultdict
import networkx as nx
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score


def load_idx2u():
    with open('/kaggle/working/GCN/data/r_MOOC10000/idx2u.pickle', 'rb') as f:
        return pickle.load(f)


def load_u2idx():
    with open('/kaggle/working/GCN/data/r_MOOC10000/u2idx.pickle', 'rb') as f:
        return pickle.load(f)


def load_course_video():
    data = {}
    with open('/kaggle/input/riginmooccube/MOOCCube/relations/course-video.json', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) == 2:
                course_id, video_id = row
                if course_id not in data:
                    data[course_id] = []
                data[course_id].append(video_id)
    return data


def load_course():
    courses = []
    with open('/kaggle/input/riginmooccube/MOOCCube/entities/course.json', 'r', encoding='utf-8') as f:
        data = f.read()
        start = 0
        end = 0
        while True:
            start = data.find('{', end)
            if start == - 1:
                break
            end = data.find('}', start) + 1
            json_obj = data[start:end]
            try:
                course = json.loads(json_obj)
                courses.append(course)
            except json.decoder.JSONDecodeError as e:
                print(f"解析错误: {e}")
    return courses


class Metrics(object):

    def __init__(self):
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # if not actual:
        # 	return 0.0
        return score / min(len(actual), k)

    def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
        '''
            y_true: (#samples, )
            y_pred: (#samples, #users)
        '''
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    def get_courses_by_video(self, video_name, course_video_mapping):
        """根据视频名称获取其所属的课程"""
        courses = []
        for course, videos in course_video_mapping.items():
            if video_name in videos:
                courses.append(course)
        return courses

# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)


class KTLoss(nn.Module):
    def __init__(self):
        super(KTLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')  # 不自动求平均

    def forward(self, pred_answers, real_answers, kt_mask):
        real_answers = real_answers[:, 1:].float()  # 截取有效部分并转为浮点
        answer_mask = kt_mask.bool()

        # --- 计算 AUC 和 ACC ---
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_answers[answer_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))  # 直接根据概率阈值计算ACC
        except ValueError:
            auc, acc = -1, -1

        # --- 计算带掩码的 BCE 损失 ---
        loss_per_element = self.bce_loss(pred_answers, real_answers)
        valid_loss = loss_per_element * answer_mask.float()  # 应用掩码
        loss = valid_loss.sum() / answer_mask.float().sum()  # 仅对有效位置求平均

        return loss, auc, acc

