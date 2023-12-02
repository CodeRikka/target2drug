import pandas as pd
import torch

def read_data(file_path):
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 将 'Protein Embedding' 列的字符串转换为 [1, n] 形状的张量
        df['Protein Embedding'] = df['Protein Embedding'].apply(
            lambda x: torch.tensor([float(num) for num in x.split(', ')])
        )

        # 丢弃 'Flag' 列
        df.drop(columns='Flag', inplace=True)

        # 展示转换后的DataFrame的前几行
        print(df.head())

        smiles = df['SMILES'].tolist()
        # 如果你想要得到 'Protein Embedding' 列的张量列表
        protein_embeddings = df['Protein Embedding'].tolist()
        # 现在，protein_embeddings 是一个包含所有蛋白质嵌入的 [1, n] 形状的PyTorch张量列表
        # print(smiles[0], protein_embeddings[0])
        return smiles, protein_embeddings
    except Exception as e:
        print(f"读取文件或转换数据时发生错误: {e}")

# # 定义文件路径
# file_path = 'data/processed_data_clean.csv'  # 更新为实际文件路径

# try:
#     # 读取CSV文件
#     df = pd.read_csv(file_path)

#     # 将 'Protein Embedding' 列的字符串转换为 [1, n] 形状的张量
#     df['Protein Embedding'] = df['Protein Embedding'].apply(
#         lambda x: torch.tensor([float(num) for num in x.split(', ')])
#     )

#     # 丢弃 'Flag' 列
#     df.drop(columns='Flag', inplace=True)

#     # 展示转换后的DataFrame的前几行
#     print(df.head())

#     smiles = df['SMILES'].tolist()
#     # 如果你想要得到 'Protein Embedding' 列的张量列表
#     protein_embeddings = df['Protein Embedding'].tolist()
#     # 现在，protein_embeddings 是一个包含所有蛋白质嵌入的 [1, n] 形状的PyTorch张量列表
#     print(smiles[0], protein_embeddings[0])
# except Exception as e:
#     print(f"读取文件或转换数据时发生错误: {e}")
