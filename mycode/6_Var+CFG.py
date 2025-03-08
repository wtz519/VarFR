import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os.path
import pickle
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader as PYG_DataLoader
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN
from torch_geometric.nn import global_mean_pool

# 设置种子
SEED = 1
random.seed(SEED)        # Python的随机库
np.random.seed(SEED)     # NumPy库
torch.manual_seed(SEED)  # CPU上的PyTorch操作
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)           # 当前GPU上的PyTorch操作
    torch.cuda.manual_seed_all(SEED)       # 所有GPU上的PyTorch操作

def top_k_acc(ground_truth_list, pred_list, k):
    pred_top_k = pred_list[0:k]
    # top_k_result = 0
    # 遍历 pred_top_k 查询pred_var_index是否在ground_truth_list中
    for pred_var_index in pred_top_k:
        if pred_var_index in ground_truth_list:
            return 1
    return 0

def calculate_mrr(true_labels, predicted_lst):
    # print("true_labels:", true_labels)
    # print("predicted_lst:", predicted_lst)
    """ 计算 MRR """
    reciprocal_rank_list = []
    # 查询 true_labels 中每个变量索引在 predicted_lst 中的位置
    for var_index2 in true_labels:
        # 如果变量索引在预测的列表里
        if var_index2 in predicted_lst:
            # 查询 var_index2 在 predicted_lst 列表中的位置
            first_occurrence_index = predicted_lst.index(var_index2)
            # 求该变量索引的倒数值
            var_result = 1 / (first_occurrence_index + 1)
            # print("var_result:", var_result)
            reciprocal_rank_list.append(var_result)
        # 如果变量不在预测的列表中
        elif var_index2 not in predicted_lst:
            var_result = 0
            reciprocal_rank_list.append(var_result)
    return max(reciprocal_rank_list)
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
        loss1 = - (targets * torch.log(logits + 1e-7) + (1 - targets) * torch.log(1 - logits + 1e-7))

        # 将损失张量与掩码相乘以过滤掉不想考虑的样本的损失
        loss1 = loss1 * mask

        if self.reduction == 'mean':
            # 计算平均损失
            loss1 = torch.sum(loss1) / torch.sum(mask)
        elif self.reduction == 'sum':
            # 计算总损失
            loss1 = torch.sum(loss1)
        return loss1

# 方法控制流图数据集
class CFG_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(CFG_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        # print("--- init 开始执行 ----")
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 在 process 方法中，需要将原始数据集处理成图数据结构，其中每个图用一个 Data 对象表示
    # 所有处理好的 Data 对象应该可以被索引，因此通常需要将 Data 存储在一个列表中。
    def process(self):
        print("--- process 开始执行 ----")
        print("self.raw_paths[0]:", self.raw_paths[0])
        print("self.process_paths[0]:", self.processed_paths[0])
        pyg_data_list = []
        # 读取 raw 文件夹下的 pickle 文件
        with open(self.raw_paths[0], mode="rb") as pkl_file:
            pkl_data_list = pickle.load(pkl_file)
            print("数据集长度为:", len(pkl_data_list))
            # print("pkl_data_list[0]:")
            # print(pkl_data_list[0])
            for data_index, pkl_data in enumerate(pkl_data_list):
                pyg_data = Data(x=pkl_data["node_tensor"],
                                edge_index=pkl_data["edge_tensor"],
                                y=pkl_data["label_tensor"],
                                var_mask=pkl_data["var_mask_tensor"],
                                train_name=pkl_data["train_name"])
                pyg_data_list.append(pyg_data)
        data, slices = self.collate(pyg_data_list)
        torch.save((data, slices), self.processed_paths[0])

    # 返回原始文件地址, 调用 self.raw_paths[0]获取原始数据文件地址
    @property
    def raw_file_names(self):
        # print("--- processed_file_names 开始执行 ---")
        return ['graph_data.pkl']

    # 返回处理后的文件地址, 调用self.processed_paths[0]获取处理后的文件地址
    @property
    def processed_file_names(self):
        # print("--- processed_file_names 开始执行 ---")
        return ["processed_data.pt"]

# 初始化 cfg 数据集
cfg_train_dataset = CFG_Dataset(root='./all_data/graph_train')
cfg_val_dataset = CFG_Dataset(root='./all_data/graph_val')
cfg_test_dataset = CFG_Dataset(root='./all_data/graph_test')

# DataLoader实例化
cfg_train_loader = PYG_DataLoader(cfg_train_dataset, batch_size=64, shuffle=False)
cfg_val_loader = PYG_DataLoader(cfg_val_dataset, batch_size=1, shuffle=False)
cfg_test_loader = PYG_DataLoader(cfg_test_dataset, batch_size=1, shuffle=False)

# 数据集样本数
print("cfg_train_data_size:", len(cfg_train_dataset))
print("cfg_val_data_size:", len(cfg_val_dataset))
print("cfg_test_data_size:", len(cfg_test_dataset))

# 局部变量表数据集
class LocalVarDataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        # 加载 pkl 文件
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
        return (train_data['var_node_tensor'], train_data['label_tensor'],
                train_data['var_mask_tensor'], train_data['train_name'])

# 实例化数据集对象
train_dataset = LocalVarDataset("./all_data/split_data/train_pkl.pkl")
val_dataset = LocalVarDataset("./all_data/split_data/val_pkl.pkl")
test_dataset = LocalVarDataset("./all_data/split_data/test_pkl.pkl")

# 样本数
print("train_dataset_size:", len(train_dataset))
print("val_dataset_size:", len(val_dataset))
print("test_dataset_size:", len(test_dataset))

# 加载数据集,构造 dataloader
var_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
var_val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
var_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 创建处理 CFG 数据的网络模型
class CFG_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(768, 128)
        self.conv2 = GCNConv(128, 128)

    def forward(self, x, edge_index, batch):
        # 第一层
        x = self.conv1(x, edge_index)
        x = x.relu()

        # 第二层
        x = self.conv2(x, edge_index)

        # 全局池化层
        x = global_mean_pool(x, batch)
        return x

# 实例化 CFG 模型
model1 = CFG_model()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model1.to(device)

# CFG 网络模型测试
gcn_x = torch.randn((4, 768), dtype=torch.float, device=device)
edge_index = torch.tensor([[0, 1, 2, 3, 2], [1, 2, 3, 0, 0]], dtype=torch.long, device=device)
batch = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)
output = model1(gcn_x, edge_index, batch)
print("test Cfg_model output.shape:", output.shape)


# 创建处理局部变量表数据的网络模型
# 构建网络模型
class Var_model(nn.Module):
    def __init__(self, input_dim=768, sentence_length=20, out_channels=128, kernel_size=3):
        super(Var_model, self).__init__()

        # 1D卷积，卷积核的大小为3，步长为1，仅在句子长度上滑动
        self.conv = nn.Conv1d(in_channels=input_dim,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=1)  # 保持长度

        # 全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层用于输出
        # self.fc = nn.Linear(out_channels, 20)  # 输出大小可根据需要修改

    def forward(self, x):
        # 输入维度为 (batch_size, sentence_length, input_dim)
        # 需要转换为 (batch_size, input_dim, sentence_length) 以便于 Conv1d 操作
        x = x.permute(0, 2, 1)  # 变为 (batch_size, input_dim, sentence_length)

        # 经过一维卷积层
        x = self.conv(x)  # 维度变为 (batch_size, out_channels, sentence_length)

        # 全局平均池化
        x = self.global_pool(x)  # 变为 (batch_size, out_channels, 1)
        x = x.squeeze(-1)  # 去掉最后一个维度，变为 (batch_size, out_channels)

        # 全连接层
        # output = self.fc(x)  # 输出维度为 (batch_size, 1)

        return x


# 实例化 LSTM 模型
model2 = Var_model()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model2.to(device)

# LSTM 模型测试
test_inputs = torch.ones((4, 550, 768), dtype=torch.float32, device=device)
output = model2(test_inputs)
print("test LSTM output.shape:", output.shape)

# print("============= 以下是fusion_model模型测试 =============")
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM_cell = Var_model()  # [batch,128]
        self.GCN_cell = CFG_model()  # [batch,128]

        # 融合层
        self.fusion_linear = nn.Linear(256, 128)  # 融合LSTM和GCN拼接的输出向量
        self.classifier = nn.Linear(128, 20)  # 最终分类类别数预测(20分类)

    def forward(self, lstm_input1, gcn_input1, gcn_edge_index1, gcn_batch1):
        # LSTM
        lstm_output = self.LSTM_cell(lstm_input1)
        # print("lstm.output.shape:", lstm_output.shape)

        # GCN
        gcn_output = self.GCN_cell(gcn_input1, gcn_edge_index1, gcn_batch1)
        # print("gcn_output.shape:", gcn_output.shape)

        # 融合 LSTM 和 GCN 输出
        fusion_output = torch.cat([lstm_output, gcn_output], dim=1)

        # 融合层
        fusion_output = F.relu(self.fusion_linear(fusion_output))

        # 分类层
        fusion_output = self.classifier(fusion_output)

        return fusion_output


# 实例化网络模型
fusion_model = FusionModel()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
fusion_model.to(device)

# 定义损失和优化器
criterion = MaskedBCEWithLogitsLoss()
optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-4)

if __name__ == '__main__':
    epoch_list = []
    train_loss_sum_list = []
    total_val_average_acc_list = []
    best_accuracy = 0

    # 早停的参数
    min_val_loss_sum = float('inf')
    no_improvement_count = 0
    patience = 20  # 设置早停等待的 epoch 数

    for epoch in range(300):
        # 模型训练
        fusion_model.train()
        train_loss_sum = 0
        batch_size = 0
        print(f"============ 第 {epoch} 轮训练开始 ============")
        for cfg_data, var_data in zip(cfg_train_loader, var_train_loader):

            # 获取 CFG 数据
            cfg_inputs = cfg_data.x.to(device)
            cfg_edge_index = cfg_data.edge_index.to(device)
            cfg_batch = cfg_data.batch.to(device)

            # 获取 Var 数据
            var_inputs, targets, targets_mask, train_csv_names = var_data
            var_inputs, targets, targets_mask = var_inputs.to(device), targets.to(device), targets_mask.to(device)
            csv_name = train_csv_names[0].split(".txt")[0]+".csv"

            # 梯度清零
            optimizer.zero_grad()

            # forward
            outputs = fusion_model(var_inputs, cfg_inputs, cfg_edge_index, cfg_batch)
            loss = criterion(outputs, targets, targets_mask)

            # backward
            loss.backward()

            # update
            optimizer.step()

            # loss累加
            train_loss_sum += loss.item()

        print("train_loss_sum:", train_loss_sum)
        train_loss_sum_list.append(train_loss_sum)

        # 模型验证
        with torch.no_grad():
            fusion_model.eval()

            acc_list = []  # 统计当前epoch中每个样本预测准确率
            mrr_list = []
            val_top1_list = []
            val_top2_list = []
            true_logged_variables_list = []
            recommendations_list = []
            val_loss_sum = 0

            for val_cfg_data, val_var_data in zip(cfg_val_loader, var_val_loader):
                # 获取 CFG 数据
                cfg_inputs = val_cfg_data.x.to(device)
                cfg_edge_index = val_cfg_data.edge_index.to(device)
                cfg_batch = val_cfg_data.batch.to(device)

                # 获取 Var 数据
                var_inputs, targets, targets_mask, train_csv_names = val_var_data
                var_inputs, targets, targets_mask = var_inputs.to(device), targets.to(device), targets_mask.to(device)
                csv_name = train_csv_names[0].split(".txt")[0] + ".csv"

                # 验证
                outputs = fusion_model(var_inputs, cfg_inputs, cfg_edge_index, cfg_batch)
                val_loss = criterion(outputs, targets, targets_mask)
                val_loss_sum += val_loss.item()

                # 根据 targets_mask 为 1 的索引, 取 output 中前几位有效的预测结果
                mask_list = targets_mask.squeeze().tolist()
                one_count = mask_list.count(1)  # 统计 mask_list 有效局部变量的个数
                # print("one_count:", one_count)
                outputs_list1 = outputs.squeeze().tolist()[0: one_count]
                # print("未截取前outputs:", outputs.squeeze().tolist())
                # print("截取后的outputs:", outputs_list1)

                # 把截取后的 outputs_list1 按照预测概率大小排序
                sorted_list = find_top_three_indices(outputs_list1)
                # print("sorted_list:", sorted_list)

                # 读取 train_label 文件获取预测值对应 index 位置的变量名
                label_dir = "./all_data/all_label"
                label_path = os.path.join(label_dir, csv_name)
                label_csv = pd.read_csv(label_path)

                # var_name_list 列表
                var_name_list = label_csv['name'].tolist()
                # print("var_name_list:", var_name_list)

                # all_label 列表
                label_list = label_csv['label'].tolist()
                # print("label_list:", label_list)

                # 根据 var_name_list 和 label_list 获得日志中变量名
                ground_truth_list = []
                for index1, label in enumerate(label_list):
                    if label == 1:
                        ground_truth_list.append(var_name_list[index1])
                # print("ground_truth_list:", ground_truth_list)
                true_logged_variables_list.append(ground_truth_list)

                # 根据排序后模型的预测结果,取出对应的变量索引
                pred_var_list = []
                for ele in sorted_list:
                    var_index = ele[0]
                    pred_var_list.append(var_name_list[var_index])
                # print("pred_var_list:", pred_var_list)
                recommendations_list.append(pred_var_list)

                # 从 pred_var_list 中取跟ground_truth 中记录变量个数一样的变量
                limit_pred_var_list = pred_var_list[0:len(ground_truth_list)]
                # print("limit_pred_var_list:", limit_pred_var_list)

                # 计算当前样本预测的 acc
                correct_count = 0
                # 日志中记录的变量个数
                total_count = len(ground_truth_list)
                for pred_var in limit_pred_var_list:  # 遍历pred_var_list 中每个变量
                    if pred_var in ground_truth_list:
                        correct_count += 1
                pred_acc = correct_count / total_count

                # 添加模型预测的准确率
                acc_list.append(pred_acc)

                # 计算当前样本的 mrr
                mrr_result = calculate_mrr(ground_truth_list, pred_var_list)
                mrr_list.append(mrr_result)

                # 计算当前样本的top1_acc
                top1_result = top_k_acc(ground_truth_list, pred_var_list, k=1)
                val_top1_list.append(top1_result)

                # 计算当前样本的top2_acc
                top2_result = top_k_acc(ground_truth_list, pred_var_list, k=2)
                val_top2_list.append(top2_result)

            # # 计算当前epoch下, 模型在验证集下loss_sum
            # print("val_loss_sum:", val_loss_sum)

            # 计算当前epoch下, 模型的 mrr
            val_mrr = sum(mrr_list) / len(mrr_list)
            print("val_mrr:", val_mrr)

            # 计算当前epoch下,模型的 top1_acc
            val_top1_acc = sum(val_top1_list) / len(val_top1_list)
            print("val_top1_acc:", val_top1_acc)

            # 计算当前epoch下,模型的 top2_acc
            val_top2_acc = sum(val_top2_list) / len(val_top2_list)
            print("val_top2_acc:", val_top2_acc)

            # 计算当前epoch下,模型的 map
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
                model_name = "Var+CFG.pth"
                model_path = os.path.join(model_dir, model_name)
                torch.save(fusion_model, model_path)
                best_accuracy = val_map
                print("模型已经保存")

    # 模型测试
    print(f"============ 模型测试开始 ============")
    test_model = torch.load("./all_data/model_save/Var+CFG.pth")
    test_model.eval()
    test_acc_list = []  # 统计每个样本的准确率
    test_mrr_list = []
    test_top1_list = []
    test_top2_list = []
    true_logged_variables_list = []
    recommendations_list = []
    with torch.no_grad():
        for test_cfg_data, test_var_data in zip(cfg_val_loader, var_val_loader):

            # 获取 cfg 数据
            cfg_data_x = test_cfg_data.x.to(device)
            cfg_data_edge_index = test_cfg_data.edge_index.to(device)
            cfg_data_batch = test_cfg_data.batch.to(device)

            # 获取 var 数据
            var_inputs, targets, targets_mask, train_csv_names = test_var_data
            var_inputs, targets, targets_mask = var_inputs.to(device), targets.to(device), targets_mask.to(device)

            # 获取测试的样本的文件名
            test_csv_names = test_cfg_data.train_name
            csv_name = test_csv_names[0].split(".txt")[0] + ".csv"

            outputs = fusion_model(var_inputs, cfg_data_x, cfg_data_edge_index, cfg_data_batch)
            # loss = criterion(outputs, targets, targets_mask)

            # 根据 targets_mask 取 outputs 中有效的预测结果
            mask_list = targets_mask.squeeze().tolist()
            one_count = mask_list.count(1)  # 统计mask_list有效局部变量的个数
            outputs_list1 = outputs.squeeze().tolist()[0: one_count]

            # 把截取后的 outputs_list1 按照预测概率大小排序组成 sorted_list
            sorted_list = find_top_three_indices(outputs_list1)

            # 读取 train_label 获取预测值对应 index 位置的变量名
            label_dir = "./all_data/all_label"
            label_path = os.path.join(label_dir, csv_name)
            label_csv = pd.read_csv(label_path)

            # var_name_list 列表
            var_name_list = label_csv['name'].tolist()

            # label 列表
            label_list = label_csv['label'].tolist()

            # 根据 label_list 获得label为 1 位置的变量名
            ground_truth_list = []
            for index1, label in enumerate(label_list):
                if label == 1:
                    ground_truth_list.append(var_name_list[index1])
            true_logged_variables_list.append(ground_truth_list)

            # 根据模型预测结果 sorted_list, 取出对应的索引
            pred_var_list = []
            for ele in sorted_list:
                var_index = ele[0]
                pred_var_list.append(var_name_list[var_index])
            recommendations_list.append(pred_var_list)

            # 从 pred_var_list 中取跟ground_truth_list变量个数一样的前几个预测变量
            limit_pred_var_list = pred_var_list[0:len(ground_truth_list)]

            # 计算当前样本的 acc
            correct_count = 0
            total_count = len(ground_truth_list)  # 日志中记录的变量个数
            for pred_var in limit_pred_var_list:
                if pred_var in ground_truth_list:
                    correct_count += 1
            pred_acc = correct_count / total_count

            # 添加模型预测的acc
            test_acc_list.append(pred_acc)

            # 计算当前样本的 mrr
            mrr_result = calculate_mrr(ground_truth_list, pred_var_list)
            test_mrr_list.append(mrr_result)

            # 计算当前样本的top1_acc
            top1_result = top_k_acc(ground_truth_list, pred_var_list, k=1)
            test_top1_list.append(top1_result)

            # 计算当前样本的top2_acc
            top2_result = top_k_acc(ground_truth_list, pred_var_list, k=2)
            test_top2_list.append(top2_result)

    # 计算模型的在测试集上整体的 mrr
    test_mrr = sum(test_mrr_list) / len(test_mrr_list)
    print("test_mrr:", test_mrr)

    # 计算top1_acc
    test_top1_acc = sum(test_top1_list) / len(test_top1_list)
    print("test_top1_acc:", test_top1_acc)

    # 计算top2_acc
    test_top2_acc = sum(test_top2_list) / len(test_top2_list)
    print("test_top2_acc:", test_top2_acc)

    # 计算map
    test_map = calculate_map(recommendations_list, true_logged_variables_list)
    print("test_map:", test_map)