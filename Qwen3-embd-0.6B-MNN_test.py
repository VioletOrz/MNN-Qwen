import MNN.llm as llm
import MNN
import numpy as np

# 创建并加载模型
model_path = r"models\ckpt\Qwen3-Embedding-0.6B-MNN\config.json"
model = llm.create(model_path, embedding_model=True)
model.load()
print("模型加载完成")

texts = [
    "中国的首都是北京。",
    "北京是中国的首都。",
    "苹果是一种水果。"
]

embeddings = []
for text in texts:
    emb_var = model.txt_embedding(text)
    emb = np.array(MNN.expr.clone(emb_var).read()).reshape(-1) 
    embeddings.append(emb)
    print(f"文本: {text}\n向量维度: {emb.shape}\n")

# 余弦相似度
def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

sim_12 = cosine_similarity(embeddings[0], embeddings[1])
sim_13 = cosine_similarity(embeddings[0], embeddings[2])

print(f"相似度（句1 vs 句2）: {sim_12:.6f}")
print(f"相似度（句1 vs 句3）: {sim_13:.6f}")
