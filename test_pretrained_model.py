"""
测试加载预训练MSHGAT模型的脚本
用于验证模型是否能正确加载已训练好的参数
"""
import torch
import argparse
import os
from HGAT import MSHGAT

def test_pretrained_model_loading():
    """测试预训练模型加载"""
    print("开始测试预训练模型加载...")
    
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_name', default='MOO')
    parser.add_argument('-batch_size', type=int, default=64)  # 与run.py保持一致
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-initialFeatureSize', type=int, default=64)
    parser.add_argument('-train_rate', type=float, default=0.8)
    parser.add_argument('-valid_rate', type=float, default=0.1)
    parser.add_argument('-n_warmup_steps', type=int, default=1000)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-save_path', default="./checkpoint/DiffusionPrediction.pt")
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-pos_emb', type=bool, default=True)
    opt = parser.parse_args([])
    opt.d_word_vec = opt.d_model

    # 尝试加载预训练模型
    print(f"尝试从 {opt.save_path} 加载预训练模型...")
    
    # 初始化MSHGAT模型
    mshgat_model = MSHGAT(opt, dropout=opt.dropout)
    
    # 检查预训练模型文件是否存在
    if os.path.exists(opt.save_path):
        print(f"发现预训练模型文件: {opt.save_path}")
        try:
            # 检查CUDA是否可用
            if torch.cuda.is_available() and not opt.no_cuda:
                print("CUDA可用，将模型加载到GPU...")
                checkpoint = torch.load(opt.save_path)
                mshgat_model.load_state_dict(checkpoint)
                mshgat_model = mshgat_model.cuda()
                print("模型已加载到GPU!")
            else:
                print("CUDA不可用或已禁用，将模型加载到CPU...")
                checkpoint = torch.load(opt.save_path, map_location=torch.device('cpu'))
                mshgat_model.load_state_dict(checkpoint)
                print("模型已加载到CPU!")
                
            print("预训练模型权重加载成功!")
        except Exception as e:
            print(f"加载预训练模型权重时出错: {e}")
            print("将使用随机初始化的模型进行测试")
    else:
        print(f"未找到预训练模型文件: {opt.save_path}")
        print("将使用随机初始化的模型进行测试")
    
    # 将模型设置为评估模式
    mshgat_model.eval()
    
    print(f"模型参数数量: {sum(p.numel() for p in mshgat_model.parameters())}")
    print(f"可训练参数数量: {sum(p.numel() for p in mshgat_model.parameters() if p.requires_grad)}")
    
    # 测试模型前向传播
    print("\n测试模型前向传播...")
    try:
        # 创建测试输入
        batch_size = 2
        seq_len = 10
        
        # 获取实际的用户/知识点数量
        user_size, _, _, _, _, _ = Split_data(opt.data_name, opt.train_rate, opt.valid_rate, load_dict=True)
        opt.user_size = user_size
        mshgat_model.n_node = opt.user_size  # 更新模型中的节点数量
        
        input_tensor = torch.randint(1, opt.user_size, (batch_size, seq_len))  # 输入序列
        input_timestamp = torch.randint(0, 100, (batch_size, seq_len))  # 时间戳
        input_idx = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # 索引
        ans = torch.randint(0, 2, (batch_size, seq_len))  # 答案
        
        # 将数据移到GPU（如果可用）
        if torch.cuda.is_available() and not opt.no_cuda:
            input_tensor = input_tensor.cuda()
            input_timestamp = input_timestamp.cuda()
            input_idx = input_idx.cuda()
            ans = ans.cuda()
        
        # 创建简单的图和超图数据用于测试
        class MockGraph:
            def __init__(self):
                self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                if torch.cuda.is_available() and not opt.no_cuda:
                    self.edge_index = self.edge_index.cuda()
        
        graph = MockGraph()
        
        # 创建超图列表
        hypergraph_list = [
            {0: torch.tensor([[0, 1], [1, 2]], dtype=torch.long).cuda() if torch.cuda.is_available() and not opt.no_cuda 
             else torch.tensor([[0, 1], [1, 2]], dtype=torch.long)},
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        ]
        if torch.cuda.is_available() and not opt.no_cuda:
            hypergraph_list[1] = hypergraph_list[1].cuda()
        
        # 执行前向传播
        with torch.no_grad():
            pred, pred_res, kt_mask, yt, hidden = mshgat_model(
                input_tensor, input_timestamp, input_idx, ans, graph, hypergraph_list
            )
        
        print(f"前向传播成功!")
        print(f"pred shape: {pred.shape}")
        print(f"pred_res shape: {pred_res.shape}")
        print(f"kt_mask shape: {kt_mask.shape}")
        print(f"yt shape: {yt.shape}")
        print(f"hidden shape: {hidden.shape}")
        
        if torch.cuda.is_available() and not opt.no_cuda:
            print(f"所有张量都在GPU上: pred on cuda = {pred.is_cuda}, yt on cuda = {yt.is_cuda}")
        
        print("\n预训练模型测试完成!")
        return mshgat_model
        
    except Exception as e:
        print(f"测试前向传播时出错: {e}")
        import traceback
        traceback.print_exc()
        return mshgat_model

if __name__ == "__main__":
    model = test_pretrained_model_loading()