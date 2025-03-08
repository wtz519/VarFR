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

# 计算公式
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

print("=================以下是 Bytecode 模型的代码=================")
class Bytecode_Dataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        # 加载pkl文件
        with open(pkl_path, 'rb') as file:
            pkl_data_list = pickle.load(file)
        self.pkl_data_list = pkl_data_list
        self.pkl_data_len = len(self.pkl_data_list)
        # print("pkl_path:", pkl_path)
        # print("pkl_data_len:", self.pkl_data_len)

    def __len__(self):
        return self.pkl_data_len

    def __getitem__(self, idx):
        # 获取第 index 个样本的数据
        train_data = self.pkl_data_list[idx]
        # print("train_data:",train_data)
        return (train_data['bytecode_tensor'], train_data['label_tensor'],
                train_data['var_mask_tensor'], train_data['train_name'])

# 实例化 Bytecode 数据集
Bytecode_train_dataset = Bytecode_Dataset("./all_data/split_data/train_pkl.pkl")
Bytecode_val_dataset = Bytecode_Dataset("./all_data/split_data/val_pkl.pkl")
Bytecode_test_dataset = Bytecode_Dataset("./all_data/split_data/test_pkl.pkl")

# 实例化 Bytecode 数据集 Dataloader
Bytecode_train_loader = DataLoader(Bytecode_train_dataset, batch_size=64, shuffle=False)
Bytecode_val_loader = DataLoader(Bytecode_val_dataset, batch_size=1, shuffle=False)
Bytecode_test_loader = DataLoader(Bytecode_test_dataset, batch_size=1, shuffle=False)

# 获取 Bytecode 数据集长度
print("Bytecode_train_dataset_size:", len(Bytecode_train_dataset))
print("Bytecode_val_dataset_size:", len(Bytecode_val_dataset))
print("Bytecode_test_dataset_size:", len(Bytecode_test_dataset))

# 创建 LSTM 网络模型
class LSTM_model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.linear1 = nn.Linear(2 * 128, 128)

    def forward(self, inputs):
        # LSTM层
        lstm_output, (h, c) = self.lstm(inputs)

        # cat层
        cat_hidden = torch.cat((h[-2], h[-1]), -1)

        # 线性层1
        cat_hidden = self.linear1(cat_hidden)

        return cat_hidden

# 实例化 LSTM 模型
model1 = LSTM_model1()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model1.to(device)

# LSTM 模型测试
test_inputs = torch.ones((4, 550, 768), dtype=torch.float32, device=device)
output = model1(test_inputs)
print("test LSTM output.shape:", output.shape)


print("================= 以下是 Var 模型的代码 =================")
class Var_Dataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        # 加载pkl文件
        with open(pkl_path, 'rb') as file:
            pkl_data_list = pickle.load(file)
        self.pkl_data_list = pkl_data_list
        self.pkl_data_len = len(self.pkl_data_list)

    def __len__(self):
        return self.pkl_data_len

    def __getitem__(self, idx):
        # 获取第 index 个样本的数据
        train_data = self.pkl_data_list[idx]
        # print("train_data:",train_data)
        return (train_data['var_node_tensor'], train_data['label_tensor'],
                train_data['var_mask_tensor'], train_data['train_name'])

# 实例化 Bytecode 数据集
Var_train_dataset = Var_Dataset("./all_data/split_data/train_pkl.pkl")
Var_val_dataset = Var_Dataset("./all_data/split_data/val_pkl.pkl")
Var_test_dataset = Var_Dataset("./all_data/split_data/test_pkl.pkl")

# 实例化 Bytecode 数据集 Dataloader
Var_train_loader = DataLoader(Var_train_dataset, batch_size=64, shuffle=False)
Var_val_loader = DataLoader(Var_val_dataset, batch_size=1, shuffle=False)
Var_test_loader = DataLoader(Var_test_dataset, batch_size=1, shuffle=False)

# 获取 Bytecode 数据集长度
print("Var_train_dataset_size:", len(Var_train_dataset))
print("Var_val_dataset_size:", len(Var_val_dataset))
print("Var_test_dataset_size:", len(Var_test_dataset))

# 创建 Var 网络模型
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


print("=================以下是 CFG 模型的代码=================")
# 定义GCN_Dataset
class CFG_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(CFG_Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        # print("--- init 开始执行 ----")
        self.data, self.slices = torch.load(self.processed_paths[0])

    # 在 process 方法中，需要将原始数据集处理成图数据结构，其中每个图用一个 Data 对象表示
    # 所有处理好的 Data 对象应该可以被索引，因此通常需要将 Data 存储在一个列表中。
    def process(self):
        # print("--- process 开始执行 ----")
        print("self.raw_paths[0]:", self.raw_paths[0])  # 原始文件存放的路径
        print("self.process_paths[0]:", self.processed_paths[0])  # 处理后的文件存放的路径
        pyg_data_list = []
        # 读取 raw 文件夹下的 pickle 文件
        with open(self.raw_paths[0], mode="rb") as pkl_file:
            pkl_data_list = pickle.load(pkl_file)
            print("数据集长度为:", len(pkl_data_list))
            for data_index, pkl_data in enumerate(pkl_data_list):
                print("===================== data_index:", data_index, "=====================")
                # print("all_data:", pkl_data)
                pyg_data = Data(x=pkl_data["node_tensor"],
                                edge_index=pkl_data["edge_tensor"],
                                y=pkl_data["label_tensor"],
                                label_mask=pkl_data["var_mask_tensor"],
                                train_name=pkl_data["train_name"])
                print("pyg_data:", pyg_data)
                pyg_data_list.append(pyg_data)
        data, slices = self.collate(pyg_data_list)
        torch.save((data, slices), self.processed_paths[0])

    # 返回原始文件地址, 调用 self.raw_paths[0]获取原始数据文件地址
    @property
    def raw_file_names(self):
        # print("--- processed_file_names 开始执行 ---")
        return ["graph_data.pkl"]

    # 返回处理后的文件地址, 调用self.processed_paths[0]获取处理后的文件地址
    @property
    def processed_file_names(self):
        # print("--- processed_file_names 开始执行 ---")
        return ["processed_data.pt"]

# 初始化GCN数据集
CFG_train_dataset = CFG_Dataset(root='./all_data/graph_train')
CFG_val_dataset = CFG_Dataset(root='./all_data/graph_val')
CFG_test_dataset = CFG_Dataset(root='./all_data/graph_test')

# GCN_DataLoader 实例化
CFG_train_loader = PYG_DataLoader(CFG_train_dataset, batch_size=64, shuffle=False)
CFG_val_loader = PYG_DataLoader(CFG_val_dataset, batch_size=1, shuffle=False)
CFG_test_loader = PYG_DataLoader(CFG_test_dataset, batch_size=1, shuffle=False)

# 获取数据集样本数
print("GCN_train_data_size:", len(CFG_train_dataset))
print("GCN_val_data_size:", len(CFG_val_dataset))
print("GCN_test_data_size:", len(CFG_test_dataset))

# 创建 CFG 网络模型
class GCN_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(768, 128)
        self.conv2 = GCNConv(128, 128)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()

        x = self.conv2(x, edge_index)

        # 全局池化层
        x = global_mean_pool(x, batch)

        return x

# 实例化GCN模型
model3 = GCN_model()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model3.to(device)


print("============= 以下是 fusion_model 模型测试 =============")
# 构建融合模型(LSTM+GCN)
class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Bytecode_cell = LSTM_model1()  # [batch,128]
        self.Var_cell = Var_model()
        self.CFG_cell = GCN_model()  # [batch,128]

        # 融合层
        self.fusion_linear = nn.Linear(384, 128)  # 融合LSTM和GCN拼接的输出向量
        self.classifier = nn.Linear(128, 20)  # 最终分类类别数预测(20分类)

    def forward(self, lstm_input1, lstm_input2,
                gcn_input1, gcn_edge_index1, gcn_batch1):
        # Bytecode
        lstm_output1 = self.Bytecode_cell(lstm_input1)

        # Var
        lstm_output2 = self.Var_cell(lstm_input2)

        # CFG
        CFG_output = self.CFG_cell(gcn_input1, gcn_edge_index1, gcn_batch1)

        # 融合层
        fusion_output = torch.cat([lstm_output1, lstm_output2, CFG_output], dim=1)

        # 通过融合层
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
        print(f"============ 第 {epoch} 轮训练开始 ============")
        # 模型训练
        fusion_model.train()
        train_loss_sum = 0
        # batch_size = 0
        for Bytecode_data, Var_data, CFG_data in zip(Bytecode_train_loader, Var_train_loader, CFG_train_loader):
            # 获取 Bytecode 数据和标签
            Bytecode_inputs, targets, targets_mask, train_csv_names = Bytecode_data
            Bytecode_inputs, targets, targets_mask = Bytecode_inputs.to(device), targets.to(device), targets_mask.to(device)
            csv_name = train_csv_names[0].split(".txt")[0]+".csv"

            # 获取 Var 数据
            Var_inputs, var_targets, var_targets_mask, var_train_csv_names = Var_data
            Var_inputs = Var_inputs.to(device)

            # 获取 CFG 数据
            CFG_inputs = CFG_data.x.to(device)
            CFG_edge_index = CFG_data.edge_index.to(device)
            CFG_batch = CFG_data.batch.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward
            outputs = fusion_model(Bytecode_inputs, Var_inputs, CFG_inputs, CFG_edge_index, CFG_batch)
            loss = criterion(outputs, targets, targets_mask)

            # backward
            loss.backward()

            # update
            optimizer.step()

            # 累加loss
            train_loss_sum += loss.item()

        print("train_loss_sum:", train_loss_sum)
        train_loss_sum_list.append(train_loss_sum)

        # 模型验证
        with torch.no_grad():
            fusion_model.eval()
            val_acc_list = []  # 记录当前 epoch 下每个测试样本的准确率 sample_acc
            val_mrr_list = []
            val_top1_list = []
            val_top2_list = []
            true_logged_variables_list = []
            recommendations_list = []
            val_loss_sum = 0

            for Bytecode_data, Var_data, CFG_data in zip(Bytecode_val_loader, Var_val_loader, CFG_val_loader):
                # 获取Bytecode数据
                Bytecode_inputs, targets, targets_mask, train_csv_names = Bytecode_data
                Bytecode_inputs, targets, targets_mask = Bytecode_inputs.to(device), targets.to(device), targets_mask.to(device)
                csv_name = train_csv_names[0].split('.txt')[0]+".csv"

                # 获取 Var 数据
                Var_inputs, Var_targets, Var_targets_mask, Var_train_csv_names = Var_data
                Var_inputs = Var_inputs.to(device)

                # 获取 CFG 数据
                CFG_inputs = CFG_data.x.to(device)
                CFG_edge_index = CFG_data.edge_index.to(device)
                CFG_batch = CFG_data.batch.to(device)

                # 模型验证
                outputs = fusion_model(Bytecode_inputs, Var_inputs, CFG_inputs, CFG_edge_index, CFG_batch)
                loss = criterion(outputs, targets, targets_mask)
                val_loss_sum += loss.item()
                # print("outputs.shape:", outputs.shape)

                # 根据 targets_mask 为1的索引,取 output 中有效的预测结果
                mask_list = targets_mask.squeeze().tolist()
                # print("mask_list:", mask_list)
                one_count = mask_list.count(1)  # 统计 mask_list 有效局部变量的个数
                outputs_list1 = outputs.squeeze().tolist()[0: one_count]
                # print("outputs_list1:", outputs_list1)

                # 把截取后的 outputs_list1 按照预测概率大小排序组成 sorted_list
                sorted_list = find_top_three_indices(outputs_list1)
                # print("sorted_list:", sorted_list)

                # 读取 train_label 文件获取预测值对应 index 位置的变量名
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
                true_logged_variables_list.append(ground_truth_list)
                # print("ground_truth_list:", ground_truth_list)

                # 根据排序后模型的预测结果, 从 var_name_list 中取出对应的变量名
                pred_var_list = []
                for ele in sorted_list:
                    var_index = ele[0]
                    pred_var_list.append(var_name_list[var_index])
                # print("pred_var_list:", pred_var_list)
                recommendations_list.append(pred_var_list)

                # 从 pred_var_list 中取跟 ground_truth 中记录变量个数一样的变量
                limit_pred_var_list = pred_var_list[0:len(ground_truth_list)]
                # print("pred_var_list:", pred_var_list)

                # 计算模型预测的acc
                correct_count = 0
                # 日志中记录的变量个数
                total_count = len(ground_truth_list)
                for pred_var in limit_pred_var_list:  # 遍历pred_var_list 中每个变量
                    if pred_var in ground_truth_list:
                        correct_count += 1
                pred_acc = correct_count / total_count
                # 添加模型预测的准确率
                val_acc_list.append(pred_acc)

                # 计算当前样本的mrr
                mrr_result = calculate_mrr(ground_truth_list, pred_var_list)
                val_mrr_list.append(mrr_result)

                # 计算该样本的 top1_acc
                top1_result = top_k_acc(ground_truth_list, pred_var_list, k=1)
                val_top1_list.append(top1_result)

                # 计算该样本的 top2_acc
                top2_result = top_k_acc(ground_truth_list, pred_var_list, k=2)
                val_top2_list.append(top2_result)

            # 计算当前epoch下,模型在验证集下的平均mrr
            val_mrr = sum(val_mrr_list) / len(val_mrr_list)
            print("val_mrr:", val_mrr)

            # 计算当前epoch下,模型的top1_acc
            val_top1_acc = sum(val_top1_list) / len(val_top1_list)
            print("val_top1_acc:", val_top1_acc)

            # 计算当前epoch下,模型的top2_acc
            val_top2_acc = sum(val_top2_list) / len(val_top2_list)
            print("val_top2_acc:", val_top2_acc)

            # 计算当前epoch下,模型的map
            val_map = calculate_map(recommendations=recommendations_list, true_logged_variables_list=true_logged_variables_list)
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
                model_name = "fusion_model_cat.pth"
                model_path = os.path.join(model_dir, model_name)
                torch.save(fusion_model, model_path)
                best_accuracy = val_map
                print("模型已经保存")

    print(f"============ 模型测试开始 ============")
    # 模型测试
    test_model = torch.load("./all_data/model_save/fusion_model_cat.pth")
    # 模型验证(评估当前训练模型的性能)
    test_model.eval()
    test_acc_list = []  # 统计每个样本的准确率
    test_mrr_list = []
    test_top1_list = []
    test_top2_list = []
    true_logged_variables_list = []
    recommendations_list = []
    with torch.no_grad():
        for Bytecode_data, Var_data, CFG_data in zip(Bytecode_test_loader, Var_test_loader, CFG_test_loader):

            # 获取 Bytecode 数据
            Bytecode_inputs, targets, targets_mask, train_csv_names = Bytecode_data
            Bytecode_inputs, targets, targets_mask = Bytecode_inputs.to(device), targets.to(device), targets_mask.to(device)
            csv_name = train_csv_names[0].split(".txt")[0]+".csv"

            # 获取 Var 数据
            Var_inputs, Var_targets, Var_targets_mask, Var_train_csv_names = Var_data
            Var_inputs = Var_inputs.to(device)

            # 获取 CFG 数据
            CFG_inputs = CFG_data.x.to(device)
            CFG_edge_index = CFG_data.edge_index.to(device)
            CFG_batch = CFG_data.batch.to(device)

            # 模型测试
            outputs = test_model(Bytecode_inputs,
                                 Var_inputs,
                                 CFG_inputs, CFG_edge_index, CFG_batch)

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

            # label 列表
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
            # 计算模型预测的acc
            correct_count = 0
            total_count = len(ground_truth_list)  # 日志中记录的变量个数
            for pred_var in limit_pred_var_list:  # 遍历 pred_var_list 中每个变量
                if pred_var in ground_truth_list:
                    correct_count += 1
            sample_acc = correct_count / total_count
            # 添加模型预测的准确率
            test_acc_list.append(sample_acc)

            # 计算当前样本下的mrr
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
