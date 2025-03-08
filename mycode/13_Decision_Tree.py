# 根据变量索引, 变量作用范围, 变量类型, 变量加载次数, 变量读取次数来构建决策树判断哪个变量导致了bug
import os.path
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

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
    # top_k_result = 0
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

def decision_tree(train_pkl, test_pkl):
    # 加载 train_pkl 获取加载的 csv 文件名
    with open(train_pkl, 'rb') as file:
        data_list = pickle.load(file)
    # print(type(data_list))
    # 获取每个训练集方法名
    train_csv_name_list = []
    for train_data in data_list:
        # 获取训练集每个文件名
        train_csv_name_list.append(train_data["train_name"])
    # print("训练集样本数量为:", len(train_csv_name_list))

    # 创建一个空的 DataFrame 用于存储合并结果
    final_train_df = pd.DataFrame()
    # 读取每个训练方法的 tree_data
    for train_csv_name in train_csv_name_list:
        # print("train_csv_name:", train_csv_name)
        train_csv_name = train_csv_name.split('.txt')[0]+'.csv'
        # print("train_csv_name:", train_csv_name)
        tree_data_dir ="./all_data/tree_data2"
        tree_data_path = os.path.join(tree_data_dir, train_csv_name)
        # 读取 csv 文件
        train_data2 = pd.read_csv(tree_data_path)
        # print(train_data2)
        final_train_df = pd.concat([final_train_df, train_data2], axis=0, ignore_index=True)
    # print("final_train_df:")
    # print(final_train_df)

    # 选择特征和标签
    train_X = final_train_df[['var_index', 'var_scope', 'var_sort', 'var_load_count', 'var_store_count']]
    train_y = final_train_df['label']
    # print(train_X)
    # print(train_y)
    # 使用决策树训练模型
    # 常用参数配置
    clf = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=5,
                                 random_state=1, min_samples_split=2, min_samples_leaf=2)
    clf.fit(train_X, train_y)

    # 加载 test_pkl 获取加载的 csv 文件名
    with open(test_pkl, 'rb') as test_file:
        test_data_list = pickle.load(test_file)
    # 获取每个测试集方法名
    test_csv_name_list = []
    for test_data in test_data_list:
        # 获取训练集每个文件名
        test_csv_name_list.append(test_data["train_name"])
    print("测试集样本数量为:", len(test_csv_name_list))

    acc_list = []
    mrr_list = []
    top1_list = []
    top2_list = []
    true_logged_variables_list = []
    recommendations_list = []

    # 读取每个测试方法的 tree_data
    for test_csv_name in test_csv_name_list:
        test_csv_name = test_csv_name.split('.txt')[0]+'.csv'
        print()
        print("test_csv_name:", test_csv_name)
        tree_data_dir ="./all_data/tree_data2"
        tree_data_path = os.path.join(tree_data_dir, test_csv_name)
        # 读取 csv 文件
        test_data2 = pd.read_csv(tree_data_path)

        # 选择特征和标签
        test_X = test_data2[['var_index', 'var_scope', 'var_sort', 'var_load_count', 'var_store_count']]
        test_y = test_data2['label']

        # 输出所有变量预测的概率
        y_pred = clf.predict_proba(test_X)
        # print("测试样本每个变量预测概率:")
        # print(y_pred)
        var_pred_list = []
        for var_pred in y_pred:
            # print("var_bug_pred:", var_pred[1])
            # 获取变量是bug变量的概率
            var_pred_list.append(var_pred[1])
        # print("var_pred_list:", var_pred_list)

        # 获取变量名
        var_name_list = test_data2['var_name']

        # 组合成子列表
        combined_list = [[var, prob] for var, prob in zip(var_name_list, var_pred_list)]
        # print("combined_list:", combined_list)

        # 按第二个元素排序
        sorted_data = sorted(combined_list, key=lambda x: x[1], reverse=True)  # reverse=True 表示降序排序

        # 打印结果
        # print("sorted_combined_list:", sorted_data)
        pred_var_name = []
        for one in sorted_data:
            pred_var_name.append(one[0])
        # print(pred_var_name)
        recommendations_list.append(pred_var_name)

        # 获取 label 为 1 的变量名, 构建 ground_truth
        test_label = test_data2[['var_name', 'label']]

        # 取出 label=1 的 var_name
        test_label_list = test_label.values.tolist()
        ground_truth_list = []
        for var_info2 in test_label_list:
            # print("var_info2:", var_info2)
            if var_info2[1] == 1:
                ground_truth_list.append(var_info2[0])
        true_logged_variables_list.append(ground_truth_list)
        print("ground_truth_list:", ground_truth_list)

        pred_list = pred_var_name[0:len(ground_truth_list)]
        print("pred_var_name:", pred_list)

        # 计算 mrr
        mrr_result = calculate_mrr(ground_truth_list, pred_list)
        mrr_list.append(mrr_result)

        # 计算 top_1_acc
        top1_result = top_k_acc(ground_truth_list, pred_list, k=1)
        top1_list.append(top1_result)

        # 计算 top_2_acc
        top2_result = top_k_acc(ground_truth_list, pred_list, k=2)
        top2_list.append(top2_result)

    print()
    print("模型测试:")
    # 计算整体预测 mrr
    test_mrr = sum(mrr_list) / len(mrr_list)
    print("test_mrr:", test_mrr)

    # 计算 top1_acc
    top1_acc = sum(top1_list) / len(top1_list)
    print("test_top1_acc:", top1_acc)

    # 计算 top2_acc
    top2_acc = sum(top2_list) / len(top2_list)
    print("test_top2_acc:", top2_acc)

    # 计算 map
    map_result = calculate_map(recommendations_list, true_logged_variables_list)
    print("test_map:", map_result)


if __name__ == '__main__':
    decision_tree("./all_data/split_data/train_pkl.pkl", "./all_data/split_data/test_pkl.pkl")