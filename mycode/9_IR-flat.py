import os.path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re
import random

# 设置种子
SEED = 5
random.seed(SEED)        # Python的随机库
np.random.seed(SEED)     # NumPy库
# torch.manual_seed(SEED)  # CPU上的PyTorch操作
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)           # 当前GPU上的PyTorch操作
#     torch.cuda.manual_seed_all(SEED)       # 所有GPU上的PyTorch操作

def calculate_precision_at_k(recommended_variables, k, true_logged_variables):
    relevant = [var for var in recommended_variables[:k] if var in true_logged_variables]
    precision = len(relevant) / k
    return precision

def calculate_average_precision(recommended_variables, true_logged_variables):
    precisions = [calculate_precision_at_k(recommended_variables, k + 1, true_logged_variables)
                  for k in range(len(recommended_variables)) if recommended_variables[k] in true_logged_variables]
    if not precisions:
        return 0.0
    average_precision = sum(precisions) / len(true_logged_variables)
    return average_precision

def calculate_map(recommendations, true_logged_variables_list):
    average_precisions = [calculate_average_precision(recs, true_vars)
                          for recs, true_vars in zip(recommendations, true_logged_variables_list)]
    map_score = sum(average_precisions) / len(average_precisions)
    return map_score

def top_k_acc(ground_truth_list, pred_list, k):
    pred_top_k = pred_list[0:k]
    # 遍历 pred_top_k 查询预测的变量是否在ground_truth_list中
    for pred_var1 in pred_top_k:
        if pred_var1 in ground_truth_list:
            return 1
    return 0

def calculate_mrr(true_labels, predicted_lst):
    """ 计算 MRR """
    reciprocal_rank_list = []
    # 查询 true_labels 中每个变量在 predicted_lst 中的位置
    for var_name in true_labels:
        # 如果变量在预测的列表里
        if var_name in predicted_lst:
            # 查询 var_name 在 predicted_lst 列表中的第一个位置
            first_occurrence_index = predicted_lst.index(var_name)
            # 求该变量索引的倒数值
            var_result = 1 / (first_occurrence_index + 1)
            # print("var_result:", var_result)
            reciprocal_rank_list.append(var_result)
        # 如果变量不在预测的列表中
        elif var_name not in predicted_lst:
            var_result = 0
            reciprocal_rank_list.append(var_result)
    return max(reciprocal_rank_list)

def get_data_name_list(lst):
    data_name_list = []
    for data_one in lst:
        # print(data_one["train_name"])
        data_name_list.append(data_one["train_name"])
    return data_name_list

def split_compound_word(word):
    # Split snake_case and camelCase
    tokens = re.split(r'[_]+', word)
    simple_tokens = []
    for token in tokens:
        simple_tokens.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', token))
    return simple_tokens

def creat_new_bytecode_tokens(lst):
    new_bytecode_tokens = []
    # 遍历每个单词把是复合词的单词进一步划分
    for token in lst:
        simple_tokens = split_compound_word(token)
        # print("simple_tokens:", simple_tokens)
        new_bytecode_tokens.extend(simple_tokens)
    return new_bytecode_tokens

def IR_process(train_path,test_path):
    # 打开并加载pkl文件
    with open(train_path, 'rb') as file:
        train_data_list = pickle.load(file)

    # 使用 all_data
    # print(type(train_data_list))
    print("训练集样本数量为:", len(train_data_list))
    train_name_list = get_data_name_list(train_data_list)
    print("len(train_name_list):", len(train_name_list))

    train_token_list = []
    # 读取对应名字的 csv 数据
    for bytecode_file_name in train_name_list:
        # print("=============="+csv_name+"==============")
        train_dir = "./all_data/all_train3"
        bytecode_file_path = os.path.join(train_dir, bytecode_file_name)
        with open(bytecode_file_path, 'r', encoding='utf-8') as file:
            bytecode_str = file.read()
            # print("bytecode_content:")
            # print(bytecode_str)
        bytecode_token_list = re.split(r'(\W+)', bytecode_str)  # 按照非字母,数字,下划线进行分割单词
        # 对复合词进行进一步划分
        bytecode_token_list = creat_new_bytecode_tokens(bytecode_token_list)
        bytecode_tokens = " ".join(bytecode_token_list)
        tuple1 = (bytecode_file_name, bytecode_tokens)
        train_token_list.append(tuple1)
    print("train_token_list:")
    # print(train_token_list[0:10])
    print(len(train_token_list))


    # 打开并加载 pkl 文件
    with open(test_path, 'rb') as test_file:
        test_data_list = pickle.load(test_file)
    # 使用data
    # print(type(test_data_list))
    print("测试集样本数量为:", len(test_data_list))
    test_name_list = get_data_name_list(test_data_list)
    # print("len(test_data_list):", len(test_data_list))

    test_token_list = []
    # 读取对应名字的 csv 数据
    for bytecode_file_name in test_name_list:
        train_dir = "./all_data/all_train3"
        bytecode_file_path = os.path.join(train_dir, bytecode_file_name)
        with open(bytecode_file_path, 'r', encoding='utf-8') as file:
            bytecode_str = file.read()
            # print("bytecode_content:")
            # print(bytecode_str)
        bytecode_token_list = re.split(r'(\W+)', bytecode_str)  # 按照非字母数字下划线进行分割单词
        bytecode_token_list = creat_new_bytecode_tokens(bytecode_token_list)
        # print("bytecode_tokens:", bytecode_tokens)
        bytecode_tokens = " ".join(bytecode_token_list)
        tuple2 = (bytecode_file_name, bytecode_tokens)
        test_token_list.append(tuple2)

    # 基于 IR 表示的模型
    x_train = []
    for train_sample in train_token_list:
        # print(train_sample[1])
        x_train.append(train_sample[1])

    x_test = []
    for test_sample in test_token_list:
        # 对于每个测试集样本
        # print(test_sample[0])
        x_test.append(test_sample[1])
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(x_train)
    X_test = vectorizer.transform(x_test)

    # 计算余弦相似度
    similarity = cosine_similarity(X_test, X_train)
    print("similarity:")
    print(similarity.shape)
    # 计算准确率
    # 取每一行最大值的索引
    # 获取每行最大值的索引
    max_indices = np.argmax(similarity, axis=1)
    print(max_indices)
    acc_list = []
    mrr_list = []
    top1_list = []
    top2_list = []
    true_logged_variables_list = []
    recommendations_list = []
    # 根据测试集的 index 取出对应的label
    for test_index, test_name in enumerate(test_name_list):
        train_label_dir = "./all_data/all_label"
        label_path = os.path.join(train_label_dir, test_name.split(".txt")[0]+".csv")
        df3 = pd.read_csv(label_path)
        label_list = df3["label"].tolist()
        var_name_list = df3["name"]
        # 取出对应label为1的索引
        ground_truth = []
        for idx, label_item in enumerate(label_list):
            if label_item == 1:
                ground_truth.append(var_name_list[idx])
        true_logged_variables_list.append(ground_truth)

        # 读取相似样本的label
        train_index = max_indices[test_index]
        pred_csv_name = train_name_list[train_index]
        pred_path = os.path.join(train_label_dir, pred_csv_name.split(".txt")[0]+".csv")
        df4 = pd.read_csv(pred_path)
        pred_label_list = df4["label"].tolist()
        pred_var_name_list = df4["name"]

        pred_list = []
        for idx, label_item in enumerate(pred_label_list):
            if label_item == 1:
                pred_list.append(pred_var_name_list[idx])
        pred_list.sort()
        recommendations_list.append(pred_list)

        # 取跟 ground_truth 等长的预测
        limit_pred_list = pred_list[0:len(ground_truth)]
        acc_count = 0
        # 计算准确率
        for one in ground_truth:
            if one in limit_pred_list:
                acc_count += 1
        acc_result = acc_count / len(ground_truth)
        acc_list.append(acc_result)

        # 计算 mrr
        mrr_result = calculate_mrr(ground_truth, pred_list)
        mrr_list.append(mrr_result)

        # 计算top1_acc
        top1_result = top_k_acc(ground_truth, pred_list, k=1)
        top1_list.append(top1_result)

        # 计算top2_acc
        top2_result = top_k_acc(ground_truth, pred_list, k=2)
        top2_list.append(top2_result)

    # 计算整体mrr
    test_mrr = sum(mrr_list) / len(mrr_list)
    print("test_mrr:", test_mrr)

    # 计算top1_acc
    top1_acc = sum(top1_list) / len(top1_list)
    print("test_top1_acc:", top1_acc)

    # 计算top2_acc
    top2_acc = sum(top2_list) / len(top2_list)
    print("test_top2_acc:", top2_acc)

    # 计算map
    map_result = calculate_map(recommendations_list, true_logged_variables_list)
    print("test_map:", map_result)

if __name__ == '__main__':
    train_path = './all_data/split_data/train_pkl.pkl'
    test_path = './all_data/split_data/test_pkl.pkl'
    IR_process(train_path, test_path)