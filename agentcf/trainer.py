# ========== 导入模块部分 ==========
import os, numpy as np                            # 基础库
from tqdm import tqdm                             # 进度条
import torch                                      # PyTorch深度学习框架
from recbole.trainer import Trainer               # RecBole基础训练器
from recbole.utils import EvaluatorType, set_color, early_stopping, dict2str, get_gpu_usage
from recbole.data.interaction import Interaction  # RecBole交互数据类
from time import time                             # 时间功能

class LanguageLossTrainer(Trainer):
    """
    语言损失训练器：继承自RecBole的Trainer类
    💡语法解释：class ClassName(ParentClass)表示类继承
    专门用于处理基于语言模型的推荐系统训练
    """
    
    def __init__(self, config, model, dataset):
        """
        初始化函数
        💡语法解释：__init__是Python的构造函数，创建对象时自动调用
        """
        super().__init__(config, model)           # 调用父类构造函数
        # 💡语法解释：super()调用父类方法，保持继承链的完整性
        
        # 从配置中读取参数
        self.sampled_user_suffix = config['sampled_user_suffix']    # 候选生成模型后缀
        self.recall_budget = config['recall_budget']                # 候选集大小，默认20
        self.fix_pos = config['fix_pos']                            # 正样本位置，默认-1
        
        # 加载已选择的物品
        self.user2sampled_item = self.load_selected_items(config, dataset)

    def load_selected_items(self, config, dataset):
        """
        加载用户的预选物品列表
        这个函数读取文件，建立用户到候选物品的映射
        """
        # 构造采样物品文件路径
        sampled_item_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.sampled_user_suffix}')
        
        # 获取用户和物品的token映射
        user_token2id = dataset.field2token_id['user_id']     # 用户token到ID的映射
        item_token2id = dataset.field2token_id['item_id']     # 物品token到ID的映射
        
        user2sampled_item = {}                    # 用户到采样物品的字典
        
        # 读取文件并解析
        with open(sampled_item_file, 'r', encoding='utf-8') as file:
            for line in file:                     # 逐行读取
                uid, iid_list = line.strip().split('\t')              # 按制表符分割
                # 💡语法解释：split('\t')按制表符分割字符串，strip()去除首尾空白
                user2sampled_item[user_token2id[uid]] = [item_token2id[_] for _ in iid_list.split(' ')]
                # 💡语法解释：列表推导式[expression for item in iterable]
        
        return user2sampled_item

    @torch.no_grad()                             # 装饰器：禁用梯度计算
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        """
        评估函数：计算模型在测试集上的性能
        💡语法解释：@torch.no_grad()是装饰器，表示函数内不计算梯度（节省内存）
        """
        eval_func = self._full_sort_batch_eval    # 选择评估函数
        
        # 如果需要加载最佳模型
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file  # 使用指定文件或默认文件
            checkpoint = torch.load(checkpoint_file, map_location=self.device)  # 加载检查点
            # 💡语法解释：map_location指定加载到的设备，避免GPU/CPU冲突
            
            self.model.load_state_dict(checkpoint["state_dict"])     # 加载模型参数
            self.model.load_other_parameter(checkpoint.get("other_parameter"))  # 加载其他参数
            
            message_output = "Loading model structure and parameters from {}".format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()                         # 设置模型为评估模式
        # 💡语法解释：model.eval()关闭dropout和batch normalization的训练行为
        
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        # 创建进度条（如果需要显示）
        iter_data = (
            tqdm(eval_data, total=len(eval_data), ncols=100, desc=set_color(f"Evaluate   ", "pink"))
            if show_progress else eval_data
        )
        # 💡语法解释：三元运算符 A if condition else B

        # 开始评估循环
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            
            sampled_items = []                    # 存储采样物品
            for i in range(len(interaction)):     # 遍历批次中的每个用户
                user_id = int(interaction['user_id'][i].item())  # 获取用户ID
                # 💡语法解释：.item()将tensor转换为Python标量
                sampled_item = self.user2sampled_item[user_id]   # 获取用户的候选物品
                sampled_items.append(sampled_item)
            
            sampled_items = torch.LongTensor(sampled_items)      # 转换为tensor

            # 处理ground truth（如果有的话）
            if self.config['has_gt']:
                self.logger.info('Has ground truth')
                idxs = torch.LongTensor(sampled_items)
                
                # 从候选中移除正样本（避免数据泄露）
                for i in range(idxs.shape[0]):
                    if positive_i[i] in idxs[i]:               # 如果正样本在候选中
                        pr = idxs[i].cpu().numpy().tolist().index(positive_i[i].item())
                        # 💡语法解释：链式调用 tensor.cpu().numpy().tolist()
                        idxs[i][pr:-1] = torch.clone(idxs[i][pr+1:])  # 移除正样本
                
                # 截取到预算大小
                idxs = idxs[:,:self.recall_budget - 1]
                
                # 根据固定位置设置重新插入正样本
                if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                    idxs = torch.cat([idxs, positive_i.unsqueeze(-1)], dim=-1).numpy()
                    # 💡语法解释：torch.cat()拼接tensor，unsqueeze(-1)增加最后一个维度
                elif self.fix_pos == 0:
                    idxs = torch.cat([positive_i.unsqueeze(-1), idxs], dim=-1).numpy()
                else:
                    idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                    # 💡语法解释：torch.split()按指定大小分割tensor
                    idxs = torch.cat([idxs_a, positive_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
            else:
                self.logger.info('Does not have ground truth.')
                idxs = torch.LongTensor(self.sampled_items)
                idxs = idxs[:,:self.recall_budget].numpy()

            # 如果需要随机打乱ground truth位置
            if self.fix_pos == -1:
                self.logger.info('Shuffle ground truth')
                for i in range(idxs.shape[0]):
                    np.random.shuffle(idxs[i])    # 随机打乱数组

            idxs = torch.LongTensor(idxs)
            
            # 执行批次评估
            interaction, scores, positive_u, positive_i = eval_func(batched_data, idxs)
            
            # 收集评估结果
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        
        # 模型收集和最终评估
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        
        return result

    def _full_sort_batch_eval(self, batched_data, sampled_items):
        """
        全排序批次评估：对候选物品进行打分排序
        """
        interaction, history_index, positive_u, positive_i = batched_data
        
        try:
            # 尝试使用模型的全排序预测功能
            scores = self.model.full_sort_predict(interaction.to(self.device), sampled_items.to(self.device))
        except NotImplementedError:
            # 如果模型没有实现全排序，则使用传统方法
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            # 💡语法解释：repeat_interleave()重复张量元素
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))

            # 根据批次大小选择预测方式
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        # 重塑分数矩阵
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf                    # 将padding项设为负无穷
        
        # 屏蔽历史交互物品
        if history_index is not None:
            scores[history_index] = -np.inf
        
        return interaction, scores, positive_u, positive_i

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None, resume_epoch=None):
        """
        训练函数：执行模型训练过程
        💡这是简化版的训练函数，主要用于AgentCF的特殊训练需求
        Args:
            resume_epoch: 从指定epoch开始训练，如果为None则从头开始
        """
        # 设置起始epoch
        start_epoch = resume_epoch if resume_epoch is not None else self.start_epoch

        # 如果已经训练完成且需要保存
        if saved and start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        # 数据收集
        self.eval_collector.data_collect(train_data)

        # 动态负采样设置
        if self.config["train_neg_sample_args"].get("dynamic", False):
            train_data.get_model(self.model)

        valid_step = 0

        # 训练循环
        for epoch_idx in range(start_epoch, self.epochs):
            # 训练一个epoch
            train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)

            # 保存检查点（如果有save_checkpoint方法）
            if hasattr(self, 'save_checkpoint'):
                self.save_checkpoint(epoch_idx, verbose=verbose)
            elif saved:
                # 如果没有save_checkpoint方法，手动保存
                self._manual_save_checkpoint(epoch_idx, verbose=verbose)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """
        训练一个epoch
        💡epoch是深度学习中的概念，表示对整个训练集进行一次完整的遍历
        """
        self.model.train()                        # 设置模型为训练模式
        loss_func = loss_func or self.model.calculate_loss  # 选择损失函数
        total_loss = None

        # 创建数据迭代器（带或不带进度条）
        iter_data = (
            tqdm(train_data, total=len(train_data), ncols=100, 
                 desc=set_color(f"Train {epoch_idx:>5}", "pink"))
            if show_progress else train_data
        )

        # 设置分布式训练的epoch
        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        # 训练循环
        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)  # 将数据移到指定设备
            
            # 显示GPU使用情况
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            
            # 计算损失（这里调用模型的calculate_loss方法）
            loss_func(interaction)

        return total_loss

    def _manual_save_checkpoint(self, epoch_idx, verbose=True):
        """
        手动保存检查点
        """
        import os
        import torch

        # 创建checkpoint目录
        checkpoint_dir = os.path.join(self.config['checkpoint_dir'], self.config['model'])
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型状态
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch_idx}.pth')
        checkpoint = {
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': getattr(self, 'optimizer', {}).state_dict() if hasattr(self, 'optimizer') else None,
            'config': self.config,
        }

        torch.save(checkpoint, checkpoint_path)

        if verbose:
            self.logger.info(f'Checkpoint saved to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path, resume_epoch=None):
        """
        加载检查点
        Args:
            checkpoint_path: 检查点文件路径
            resume_epoch: 指定要加载的epoch，如果为None则从最新checkpoint加载
        """
        import torch

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            if hasattr(self, 'optimizer') and checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            loaded_epoch = checkpoint.get('epoch', 0)
            self.logger.info(f'Loaded checkpoint from epoch {loaded_epoch}')
            return loaded_epoch
        else:
            self.logger.info(f'Checkpoint {checkpoint_path} not found, starting from scratch')
            return 0
