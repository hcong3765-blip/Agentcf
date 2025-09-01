# ========== 导入模块部分 ==========
import sys                                          # 系统相关功能（如命令行参数）
from logging import getLogger                       # 日志记录器，用于输出程序运行信息
import argparse                                     # 命令行参数解析器，处理用户输入的参数
from recbole.config import Config                   # RecBole框架的配置管理器
from recbole.data import create_dataset, data_preparation  # RecBole的数据集创建和预处理
from recbole.data.transform import construct_transform     # RecBole的数据变换构造器
from recbole.utils import init_logger, get_trainer, init_seed, set_color, get_flops  # RecBole工具函数
from trainer import LanguageLossTrainer  # 自定义训练器
from utils import get_model                         # 自定义工具函数：动态模型加载
from dataset import BPRDataset, ITEMBPRDataset     # 自定义数据集类

def run_baseline(model_name, dataset_name, **kwargs):
    """
    核心函数：运行基线模型的完整流程
    参数：
    - model_name: 模型名称（如'AgentCF'）
    - dataset_name: 数据集名称（如'ml-1m'）  
    - **kwargs: 其他可选参数
    """
    
    # ========== 第一步：配置文件准备 ==========
    props = ['props/overall.yaml', f'props/{model_name}.yaml', f'props/{dataset_name}.yaml']
    print(props)                                    # 🐛调试点：查看配置文件路径列表
    # 💡解释：配置文件按优先级加载，后面的会覆盖前面的同名配置
    
    # ========== 第二步：动态模型加载 ==========  
    model_class = get_model(model_name)             # 调用utils.py中的get_model函数
    # 💡解释：根据模型名动态导入对应的模型类
    
    # ========== 第三步：配置对象初始化 ==========
    config = Config(
        model=model_class,                          # 模型类对象
        dataset=dataset_name,                       # 数据集名称
        config_file_list=props,                     # 配置文件列表
        config_dict=kwargs,                         # 额外配置参数
    )
    init_seed(config["seed"], config["reproducibility"])  # 设置随机种子，确保结果可重现
    
    # ========== 第四步：日志系统初始化 ==========
    init_logger(config)                             # 初始化日志系统
    logger = getLogger()                            # 获取日志记录器
    logger.info(sys.argv)                           # 记录命令行参数
    logger.info(config)                             # 记录配置信息
    
    # ========== 第五步：数据集选择和创建 ==========
    # 💡解释：根据不同模型选择不同的数据集处理方式
    if model_name == 'BPR' or model_name == 'UUPretrain' or model_name == 'ReRec' or model_name == 'AllReRec' or model_name == 'IITest' or model_name == 'TestGames' or model_name == 'SparseReRec'\
            or model_name in ['TestPantry', 'TestOffice', 'IITestDiag', 'IITestDiagNew', 'TestOfficeBPR', 'TestOfficeUUPretrain','UserReRec','AgentCF']:
        dataset = BPRDataset(config)                # 使用自定义BPR数据集
    elif model_name in ['UUTest','UUTestDiag','TestOfficeUUTest']:
        dataset = ITEMBPRDataset(config)            # 使用物品BPR数据集
    else:
        dataset = create_dataset(config)            # 使用RecBole默认数据集创建

    logger.info(dataset)                            # 记录数据集信息

    # ========== 第六步：数据分割 ==========
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # 💡解释：将数据集分成训练集、验证集、测试集，通常按时间顺序分割

    # ========== 第七步：模型实例化 ==========
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])  # 再次设置种子
    model = model_class(config, train_data._dataset).to(config["device"])       # 创建模型并移到指定设备
    logger.info(model)                              # 记录模型结构信息

    # ========== 第八步：数据变换构造 ==========
    transform = construct_transform(config)         # 构造数据变换（如归一化等）
    
    # ========== 第九步：训练器选择 ==========
    # 💡解释：不同模型需要不同的训练策略
    if model_name in ['SASRec','BPRMF']:
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)  # RecBole默认训练器
    elif model_name in ['UUTest', 'UUTestDiag','TestOfficeUUTest','TestOfficeUUTestDiag']:
        trainer = LanguageLossTrainer(config,model,dataset)                          # 语言损失训练器
    else:
        trainer = LanguageLossTrainer(config,model,dataset)                          # 语言损失训练器

    # ========== 第十步：模型训练 ==========
    if not config['test_only']:                     # 如果不是仅测试模式
        # 检查是否有resume_epoch参数
        resume_epoch = kwargs.get('resume_epoch', None)
        if resume_epoch is not None:
            # 加载checkpoint
            checkpoint_path = kwargs.get('checkpoint_path', f'./saved/{model_name}/checkpoint_epoch_{resume_epoch-1}.pth')
            loaded_epoch = trainer.load_checkpoint(checkpoint_path)
            if loaded_epoch != resume_epoch - 1:
                logger.warning(f"Loaded epoch {loaded_epoch}, but expected {resume_epoch-1}")
            resume_epoch = loaded_epoch + 1

        trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"], resume_epoch=resume_epoch)
        # 💡解释：开始训练过程，saved=True表示保存最佳模型

    # ========== 第十一步：模型评估 ==========
    test_result = trainer.evaluate(test_data, model_file='./AgentCF-Sep-07-2024_16-09-29.pth', 
                                  load_best_model=False, show_progress=config["show_progress"])
    print(test_result)                              # 🐛调试点：打印测试结果

    logger.info(set_color("test result", "yellow") + f": {test_result}")  # 彩色日志输出
    
    # ========== 第十二步：返回结果 ==========
    return model_name, dataset_name, {
        "test_result": test_result,
    }

# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="SASRec", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="ml-1m", help="name of datasets")
    parser.add_argument("--resume_epoch", "-r", type=int, default=None, help="resume training from specific epoch")
    parser.add_argument("--checkpoint_path", "-c", type=str, default=None, help="path to checkpoint file")

    args, _ = parser.parse_known_args()             # 解析命令行参数

    # 🚀 启动核心流程
    kwargs = {}
    if args.resume_epoch is not None:
        kwargs['resume_epoch'] = args.resume_epoch
    if args.checkpoint_path is not None:
        kwargs['checkpoint_path'] = args.checkpoint_path

    run_baseline(args.model, args.dataset, **kwargs)