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
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)  # 加载检查点
            # 💡语法解释：map_location指定加载到的设备，避免GPU/CPU冲突
            
            # 加载模型参数
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            elif "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.logger.warning("⚠️ Checkpoint中没有找到模型状态字典")
            
            # 加载智能体状态
            if 'user_agents' in checkpoint and hasattr(self.model, 'user_agents'):
                try:
                    self.model.user_agents = checkpoint['user_agents']
                    self.logger.info(f'✅ 评估时加载用户智能体状态: {len(self.model.user_agents)} 个智能体')
                except Exception as agent_error:
                    self.logger.warning(f"⚠️ 评估时加载用户智能体状态失败: {str(agent_error)}")
            
            if 'item_agents' in checkpoint and hasattr(self.model, 'item_agents'):
                try:
                    self.model.item_agents = checkpoint['item_agents']
                    self.logger.info(f'✅ 评估时加载物品智能体状态: {len(self.model.item_agents)} 个智能体')
                except Exception as agent_error:
                    self.logger.warning(f"⚠️ 评估时加载物品智能体状态失败: {str(agent_error)}")
            
            if 'embedding_agent' in checkpoint and hasattr(self.model, 'embedding_agent'):
                try:
                    self.model.embedding_agent = checkpoint['embedding_agent']
                    self.logger.info(f'✅ 评估时加载嵌入智能体状态')
                except Exception as agent_error:
                    self.logger.warning(f"⚠️ 评估时加载嵌入智能体状态失败: {str(agent_error)}")
            
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
            try:
                # 训练一个epoch
                train_loss = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)

                # 保存检查点（如果有save_checkpoint方法）
                if hasattr(self, 'save_checkpoint'):
                    self.save_checkpoint(epoch_idx, verbose=verbose)
                elif saved:
                    # 如果没有save_checkpoint方法，手动保存
                    self._manual_save_checkpoint(epoch_idx, verbose=verbose)

            except Exception as e:
                # 打印错误信息但继续训练
                self.logger.error(f"❌ Epoch {epoch_idx} 训练失败: {str(e)}")
                self.logger.error(f"错误类型: {type(e).__name__}")
                import traceback
                self.logger.error(f"详细错误信息:\n{traceback.format_exc()}")

                # 即使出现错误也尝试保存检查点
                try:
                    if saved:
                        self._manual_save_checkpoint(epoch_idx, verbose=False)
                        self.logger.info(f"✅ 已保存epoch {epoch_idx}的检查点")
                except Exception as save_error:
                    self.logger.error(f"❌ 保存检查点失败: {str(save_error)}")

                # 继续下一个epoch
                self.logger.info(f"🔄 继续训练下一个epoch...")
                continue

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
            try:
                interaction = interaction.to(self.device)  # 将数据移到指定设备

                # 显示GPU使用情况
                if self.gpu_available and show_progress:
                    iter_data.set_postfix_str(
                        set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                    )

                # 计算损失（这里调用模型的calculate_loss方法）
                loss_func(interaction)

            except Exception as e:
                # 打印batch级别的错误信息但继续训练
                self.logger.error(f"❌ Batch {batch_idx} (Epoch {epoch_idx}) 处理失败: {str(e)}")
                self.logger.error(f"错误类型: {type(e).__name__}")

                # 记录交互数据的基本信息（用于调试）
                try:
                    if hasattr(interaction, 'shape'):
                        self.logger.error(f"交互数据shape: {interaction.shape}")
                    elif hasattr(interaction, '__len__'):
                        self.logger.error(f"交互数据长度: {len(interaction)}")
                    else:
                        self.logger.error(f"交互数据类型: {type(interaction)}")
                except Exception as info_error:
                    self.logger.error(f"无法获取交互数据信息: {str(info_error)}")

                # 继续下一个batch
                self.logger.info(f"🔄 跳过有问题的batch {batch_idx}，继续训练...")
                continue

        return total_loss

    def _manual_save_checkpoint(self, epoch_idx, verbose=True):
        """
        手动保存检查点，包含智能体状态和时间戳
        """
        import os
        import torch
        from datetime import datetime

        try:
            # 创建带时间戳的checkpoint目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(self.config['checkpoint_dir'], self.config['model'], f"run_{timestamp}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # 保存模型状态
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch_idx}.pth')
            
            # 构建checkpoint字典，包含智能体状态
            checkpoint = {
                'epoch': epoch_idx,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': getattr(self, 'optimizer', {}).state_dict() if hasattr(self, 'optimizer') else None,
                'config': self.config,
                'timestamp': timestamp,
            }
            
            # 保存智能体状态（如果模型有这些属性）
            if hasattr(self.model, 'user_agents'):
                checkpoint['user_agents'] = self.model.user_agents
                if verbose:
                    self.logger.info(f"💾 保存用户智能体状态: {len(self.model.user_agents)} 个智能体")
            
            if hasattr(self.model, 'item_agents'):
                checkpoint['item_agents'] = self.model.item_agents
                if verbose:
                    self.logger.info(f"💾 保存物品智能体状态: {len(self.model.item_agents)} 个智能体")
            
            if hasattr(self.model, 'embedding_agent'):
                checkpoint['embedding_agent'] = self.model.embedding_agent
                if verbose:
                    self.logger.info("💾 保存嵌入智能体状态")

            torch.save(checkpoint, checkpoint_path)

            if verbose:
                self.logger.info(f'✅ Checkpoint saved to {checkpoint_path}')
                self.logger.info(f'🕒 时间戳: {timestamp}')

        except Exception as e:
            self.logger.error(f"❌ 保存checkpoint失败 (epoch {epoch_idx}): {str(e)}")
            # 尝试保存到备用位置
            try:
                backup_path = os.path.join('./saved_backup', f'checkpoint_epoch_{epoch_idx}_backup_{timestamp}.pth')
                os.makedirs('./saved_backup', exist_ok=True)
                torch.save(checkpoint, backup_path)
                self.logger.info(f'✅ Checkpoint saved to backup location: {backup_path}')
            except Exception as backup_error:
                self.logger.error(f"❌ 备份checkpoint也失败: {str(backup_error)}")

    def load_checkpoint(self, checkpoint_path, resume_epoch=None):
        """
        加载检查点
        Args:
            checkpoint_path: 检查点文件路径
            resume_epoch: 指定要加载的epoch，如果为None则从最新checkpoint加载
        """
        import torch

        try:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                # 加载模型状态
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f'✅ 模型状态已加载')
                except Exception as model_error:
                    self.logger.error(f"❌ 加载模型状态失败: {str(model_error)}")
                    return 0

                # 加载优化器状态
                if hasattr(self, 'optimizer') and checkpoint.get('optimizer_state_dict'):
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.logger.info(f'✅ 优化器状态已加载')
                    except Exception as opt_error:
                        self.logger.warning(f"⚠️ 加载优化器状态失败，将重新初始化: {str(opt_error)}")

                # 加载智能体状态
                if 'user_agents' in checkpoint and hasattr(self.model, 'user_agents'):
                    try:
                        self.model.user_agents = checkpoint['user_agents']
                        self.logger.info(f'✅ 用户智能体状态已加载: {len(self.model.user_agents)} 个智能体')
                    except Exception as agent_error:
                        self.logger.warning(f"⚠️ 加载用户智能体状态失败: {str(agent_error)}")
                
                if 'item_agents' in checkpoint and hasattr(self.model, 'item_agents'):
                    try:
                        self.model.item_agents = checkpoint['item_agents']
                        self.logger.info(f'✅ 物品智能体状态已加载: {len(self.model.item_agents)} 个智能体')
                    except Exception as agent_error:
                        self.logger.warning(f"⚠️ 加载物品智能体状态失败: {str(agent_error)}")
                
                if 'embedding_agent' in checkpoint and hasattr(self.model, 'embedding_agent'):
                    try:
                        self.model.embedding_agent = checkpoint['embedding_agent']
                        self.logger.info(f'✅ 嵌入智能体状态已加载')
                    except Exception as agent_error:
                        self.logger.warning(f"⚠️ 加载嵌入智能体状态失败: {str(agent_error)}")

                loaded_epoch = checkpoint.get('epoch', 0)
                timestamp = checkpoint.get('timestamp', 'unknown')
                self.logger.info(f'✅ Checkpoint loaded from epoch {loaded_epoch} (timestamp: {timestamp})')
                return loaded_epoch
            else:
                self.logger.info(f'📝 Checkpoint {checkpoint_path} not found, starting from scratch')
                return 0

        except Exception as e:
            self.logger.error(f"❌ 加载checkpoint失败: {str(e)}")
            self.logger.info(f'📝 将从头开始训练')
            return 0
