import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 设置种子
SEED = 1
random.seed(SEED)        # Python的随机库
np.random.seed(SEED)     # NumPy库
torch.manual_seed(SEED)  # CPU上的PyTorch操作

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)           # 当前GPU上的PyTorch操作
    torch.cuda.manual_seed_all(SEED)       # 所有GPU上的PyTorch操作

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

def find_top_three_indices(lst):
    # 创建列表,其中列表的元素是元组，每个元组包含预测值和对应的索引
    indexed_lst = [(index1, value) for index1, value in enumerate(lst)]
    # print("排序前:", indexed_lst)
    # 对indexed_lst 按每个元组的第1个子元素从大到小顺序排序
    indexed_lst.sort(reverse=True, key=lambda x: x[1])
    # print("排序后:", indexed_lst)
    return indexed_lst

class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        # 应用 Sigmoid 激活函数
        logits = torch.sigmoid(logits)

        # 计算二进制交叉熵损失
        loss = - (targets * torch.log(logits + 1e-7) + (1 - targets) * torch.log(1 - logits + 1e-7))

        # 将损失张量与掩码相乘以过滤掉不想考虑的样本的损失
        loss = loss * mask

        if self.reduction == 'mean':
            # 计算平均损失
            loss = torch.sum(loss) / torch.sum(mask)
        elif self.reduction == 'sum':
            # 计算总损失
            loss = torch.sum(loss)
        return loss

class CodeBertDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        # 加载pkl文件
        with open(pkl_path, 'rb') as file:
            pkl_data_list = pickle.load(file)
        self.pkl_data_list = pkl_data_list
        self.pkl_data_len = len(self.pkl_data_list)
        print("pkl_path:", pkl_path)
        print("pkl_data_len:", self.pkl_data_len)

    def __len__(self):
        return self.pkl_data_len

    def __getitem__(self, idx):
        # 获取第 index 个样本
        train_data = self.pkl_data_list[idx]
        return (train_data['bytecode_tensor'], train_data['label_tensor'],
                train_data['var_mask_tensor'], train_data['train_name'])

# 实例化数据集对象
train_dataset = CodeBertDataset("./all_data/split_data/train_pkl.pkl")
val_dataset = CodeBertDataset("./all_data/split_data/val_pkl.pkl")
test_dataset = CodeBertDataset("./all_data/split_data/test_pkl.pkl")
print("train_dataset_size:", len(train_dataset))
print("val_dataset_size:", len(val_dataset))
print("test_dataset_size:", len(test_dataset))

# 加载数据集,构造dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 构建网络模型
class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(768, 128)
        self.classifier = nn.Linear(128, 20)

    def forward(self, x):
        # 输入 x 的形状: [64, 65, 768]
        # print("x.shape (input):", x.shape)

        # 对维度 65（序列维度）进行池化（这里用均值池化）
        x = x.mean(dim=1)  # 结果为 [64, 768]
        # print("x.shape (after pooling):", x.shape)

        # 线性变换 1
        x = self.linear1(x)  # 结果为 [64, 128]
        # print("x.shape (after linear1):", x.shape)

        # 线性变换 2（分类器）
        x = self.classifier(x)  # 结果为 [64, 20]
        # print("x.shape (final output):", x.shape)

        return x

# 实例化模型
model1 = myModel()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model1.to(device)

# # 定义损失和优化器
criterion = MaskedBCEWithLogitsLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-4)

# 模型训练和测试
if __name__ == '__main__':
    epoch_list = []
    epoch_train_loss_sum_list = []
    epoch_val_acc_list = []  # 统计所有 epoch 下 val 集的平均acc
    best_accuracy = 0

    # 早停的参数
    min_val_loss_sum = float('inf')
    no_improvement_count = 0
    patience = 20  # 设置早停等待的 epoch 数

    for epoch in range(200):
        # 模型训练
        model1.train()
        train_loss_sum = 0
        print(f"============ 第 {epoch} 轮训练开始 ============")
        # 记录当前的epoch
        epoch_list.append(epoch)
        for batch_index, data in enumerate(train_loader):
            # print("batch_index:", batch_index)
            # print(data)
            inputs, targets, targets_mask, train_csv_names = data
            inputs, targets, targets_mask = inputs.to(device), targets.to(device), targets_mask.to(device)

            # print("   inputs.shape:", inputs.shape)  # [16, 550, 768]
            # print("   targets.shape:", targets.shape)  # [16, 20]
            # print("   targets_mask:", targets_mask.shape)  # [16, 20]
            # print("   train_names:", train_csv_names)  # 16

            optimizer.zero_grad()
            # forward
            outputs = model1(inputs)
            # print("   outputs.shape:", outputs.shape)
            loss = criterion(outputs, targets, targets_mask)
            # backward
            loss.backward()
            # update
            optimizer.step()

            # loss 累加
            train_loss_sum += loss.item()
        print("train_loss_sum:", train_loss_sum)
        epoch_train_loss_sum_list.append(train_loss_sum)

        # 模型验证(评估当前训练模型的性能)
        with torch.no_grad():
            model1.eval()

            val_acc_list = []  # 统计当前 epoch 中每个样本的准确率
            val_mrr_list = []
            val_top1_list = []
            val_top2_list = []
            true_logged_variables_list = []
            recommendations_list = []
            val_loss_sum = 0

            for batch_index, data in enumerate(val_loader):
                # print()
                # print(f"batch{batch_index}:")
                inputs, targets, targets_mask, val_csv_names = data
                inputs, targets, targets_mask = inputs.to(device), targets.to(device), targets_mask.to(device)
                # print("inputs.shape:", inputs.shape)
                # print("targets:", targets)
                # print("targets_mask:", targets_mask)
                # print("val_csv_names:", val_csv_names)
                # 获取测试的方法的 has_log_method_csv 名字
                csv_name = val_csv_names[0].split(".txt")[0]+".csv"

                outputs = model1(inputs)
                loss = criterion(outputs, targets, targets_mask)
                val_loss_sum += loss.item()
                # print("outputs:", outputs)
                # print("outputs.shape:", outputs.shape)

                # 根据 targets_mask 取 output 有效的预测结果
                mask_list = targets_mask.squeeze().tolist()
                one_count = mask_list.count(1)  # 统计 mask_list 有效局部变量的个数
                # print("one_count:", one_count)
                # 取 output 里的前 one_count 有效的预测结果
                outputs_list1 = outputs.squeeze().tolist()[0: one_count]
                # print("未截取前outputs:", outputs.squeeze().tolist())
                # print("截取后的outputs:", outputs_list1)

                # 把截取后的 outputs_list1 按照预测概率大小排序组成 sorted_list
                sorted_list = find_top_three_indices(outputs_list1)
                # print("sorted_list:", sorted_list)

                # 读取 train_label 获取预测值对应 index 位置的变量名
                label_dir = "./all_data/all_label"
                label_path = os.path.join(label_dir, csv_name)
                label_csv = pd.read_csv(label_path)

                # var_name_list 列表
                var_name_list = label_csv['name'].tolist()
                # print("var_name_list:", var_name_list)

                # label 列表
                label_list = label_csv['label'].tolist()
                # print("label_list:", label_list)
                # 根据 var_name_list 和 label_list 获得日志中记录的变量名
                ground_truth_list = []
                for index, label in enumerate(label_list):
                    if label == 1:
                        ground_truth_list.append(var_name_list[index])
                # print("ground_truth_list:", ground_truth_list)
                true_logged_variables_list.append(ground_truth_list)

                # 根据模型预测结果, 取出对应的变量名
                pred_var_list = []
                # print("概率从大到小排序后的局部变量推荐表:")
                for ele in sorted_list:
                    var_index = ele[0]
                    pred_var_list.append(var_name_list[var_index])
                # print("pred_var_list:", pred_var_list)
                recommendations_list.append(pred_var_list)

                # 从 pred_var_list 中取跟日志中记录变量个数一样的前几个预测变量
                limit_pred_var_list = pred_var_list[0:len(ground_truth_list)]
                # print("pred_var_list:", pred_var_list)

                # 计算该样本预测准确率
                correct_count = 0
                total_count = len(ground_truth_list)  # 日志中记录的变量个数
                for pred_var in limit_pred_var_list:  # 遍历pred_var_list 中每个变量
                    if pred_var in ground_truth_list:
                        correct_count += 1
                sample_acc = correct_count / total_count

                # 添加模型预测的 acc
                val_acc_list.append(sample_acc)

                # 计算该样本预测 mrr
                mrr_result = calculate_mrr(ground_truth_list, pred_var_list)
                val_mrr_list.append(mrr_result)

                # 计算该样本的 top1_acc
                top1_result = top_k_acc(ground_truth_list, pred_var_list, k=1)
                val_top1_list.append(top1_result)

                # 计算该样本的 top2_acc
                top2_result = top_k_acc(ground_truth_list, pred_var_list, k=2)
                val_top2_list.append(top2_result)

        # 计算当前epoch下, 模型在验证集下loss_sum
        print("val_loss_sum:", val_loss_sum)

        # 计算模型的在验证集上整体的 mrr
        val_mrr = sum(val_mrr_list) / len(val_mrr_list)
        print("val_mrr:", val_mrr)

        # 计算当前epoch下,模型的top1_acc
        val_top1_acc = sum(val_top1_list) / len(val_top1_list)
        print("val_top1_acc:", val_top1_acc)

        # 计算当前epoch下,模型的top2_acc
        val_top2_acc = sum(val_top2_list) / len(val_top2_list)
        print("val_top2_acc:", val_top2_acc)

        # 计算当前epoch下,模型map
        val_map = calculate_map(recommendations_list, true_logged_variables_list)
        print("val_map:", val_map)

        # 检查是否有改善
        if val_loss_sum < min_val_loss_sum:
            min_val_loss_sum = val_loss_sum
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # 检查是否需要早停
        if no_improvement_count >= patience:
            print(f"早停: 在连续 {patience} 个 epoch 中验证集损失loss没有改善.")
            break

        # Save the model if the accuracy is the best
        if val_map > best_accuracy:
            # 模型保存
            model_dir = "./all_data/model_save"
            model_name = "CodeBert_model.pth"
            model_path = os.path.join(model_dir, model_name)
            torch.save(model1, model_path)
            best_accuracy = val_map
            print("模型已经保存")

    # 模型测试
    print(f"============ 模型测试开始 ============")
    test_model = torch.load("./all_data/model_save/CodeBert_model.pth")
    # 模型验证(评估当前训练模型的性能)
    test_model.eval()
    test_acc_list = []  # 统计每个样本的准确率
    test_mrr_list = []
    test_top1_list = []
    test_top2_list = []
    true_logged_variables_list = []
    recommendations_list = []
    with torch.no_grad():
        for batch_index, data in enumerate(test_loader):
            inputs, targets, targets_mask, test_csv_names = data
            inputs, targets, targets_mask = inputs.to(device), targets.to(device), targets_mask.to(device)
            csv_name = test_csv_names[0].split(".txt")[0]+".csv"

            outputs = test_model(inputs)
            loss = criterion(outputs, targets, targets_mask)

            # 根据 targets_mask 取 outputs 中有效的预测结果
            mask_list = targets_mask.squeeze().tolist()
            one_count = mask_list.count(1)  # 统计mask_list有效局部变量的个数

            # 取 outputs 里的前 one_count 有效的预测结果
            outputs_list1 = outputs.squeeze().tolist()[0: one_count]

            # 把截取后的 outputs_list1 按照预测概率大小排序组成 sorted_list
            sorted_list = find_top_three_indices(outputs_list1)

            # 读取 train_label 获取预测值对应 index 位置的变量名
            label_dir = "./all_data/all_label"
            label_path = os.path.join(label_dir, csv_name)
            label_csv = pd.read_csv(label_path)

            # var_name_list 列表
            var_name_list = label_csv['name'].tolist()

            # all_label 列表
            label_list = label_csv['label'].tolist()

            # 根据 var_name_list 和 label_list 获得日志中记录的变量名
            ground_truth_list = []
            for index, label in enumerate(label_list):
                if label == 1:
                    ground_truth_list.append(var_name_list[index])
            true_logged_variables_list.append(ground_truth_list)

            # 根据模型预测结果, 取出对应的变量名
            pred_var_list = []
            for ele in sorted_list:
                var_index = ele[0]
                pred_var_list.append(var_name_list[var_index])
            recommendations_list.append(pred_var_list)

            # 从 pred_var_list 中取跟日志中记录变量个数一样的前几个预测变量
            limit_pred_var_list = pred_var_list[0:len(ground_truth_list)]

            # 计算模型预测的正确率
            correct_count = 0
            total_count = len(ground_truth_list)  # 日志中记录的变量个数
            for pred_var in limit_pred_var_list:  # 遍历pred_var_list 中每个变量
                if pred_var in ground_truth_list:
                    correct_count += 1
            sample_acc = correct_count / total_count
            # 添加模型预测的准确率
            test_acc_list.append(sample_acc)

            # 计算该样本预测 mrr
            mrr_result = calculate_mrr(ground_truth_list, pred_var_list)
            test_mrr_list.append(mrr_result)

            # 计算当前样本的top1_acc
            top1_result = top_k_acc(ground_truth_list, pred_var_list, k=1)
            test_top1_list.append(top1_result)

            # 计算当前样本的top2_acc
            top2_result = top_k_acc(ground_truth_list, pred_var_list, k=2)
            test_top2_list.append(top2_result)

    # # 计算模型在测试集下的整体平均acc
    # test_acc_result = sum(test_acc_list) / len(test_acc_list)
    # print("test_acc_result:", test_acc_result)

    # 计算模型的在测试集上整体的 mrr
    test_mrr = sum(test_mrr_list) / len(test_mrr_list)
    print("test_mrr:", test_mrr)

    # 计算top1_acc
    test_top1_acc = sum(test_top1_list) / len(test_top1_list)
    print("test_top1_acc:", test_top1_acc)

    # 计算top2_acc
    test_top2_acc = sum(test_top2_list) / len(test_top2_list)
    print("test_top2_acc:", test_top2_acc)

    # 计算 map
    test_map = calculate_map(recommendations_list, true_logged_variables_list)
    print("test_map:", test_map)

