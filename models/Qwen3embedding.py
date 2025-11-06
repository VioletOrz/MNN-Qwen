import MNN.llm as llm
import MNN
import numpy as np
import os
from tools.Violet_base import *
import time
import datetime
from typing import Literal

# 创建并加载模型

def text_to_text_list(text):
    texts = text.split('\n\n')
    text_list = [t.strip() for t in texts if len(t.strip()) > 0]
    return text_list
    

class Qwen3Embedding:
    def __init__(self, model_path, db_path = 'data', index_path = 'index.yaml'):
        self.model = llm.create(model_path, embedding_model=True)
        self.model.load()
        self.db_path = db_path
        self.index_path = index_path
        print("模型加载完成")

    def embedding_text(self, text):
        emb_var = self.model.txt_embedding(text)
        emb = np.array(MNN.expr.clone(emb_var).read()).reshape(-1) 
        return emb
    
    def cosine_similarity(self, a, b):
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    def text_list_embeddings(self, texts):
        embeddings = []
        for text in texts:
            emb = self.embedding_text(text)
            embeddings.append(emb)
            print(f"文本: {text}\n向量维度: {emb.shape}\n")
        return embeddings

    def embedding_text_file_and_save_to_json_and_pkl(self, text_path, sector_name, embed_data_name):
        text_list = text_to_text_list(read_txt_file(text_path))
        embeddings = self.text_list_embeddings(text_list)

        pkl_data = {}
        json_data = {}
        for idx, emb in enumerate(embeddings):
            pkl_data[f'{idx:06d}'] = {
                'text': text_list[idx],
                'embedding': emb
            }
            json_data[f'{idx:06d}'] = {
                'text': text_list[idx],
                'embedding': "Plase load pkl file to get embedding data."
            }
        if os.path.exists(f'{self.db_path}/{sector_name}') == False:
            os.makedirs(f'{self.db_path}/{sector_name}')

        write_pkl_file(f'{self.db_path}/{sector_name}/{embed_data_name}.pkl', pkl_data)
        write_json_file(f'{self.db_path}/{sector_name}/{embed_data_name}.json', json_data)
    
        index = read_yaml_file(os.path.join(self.db_path, self.index_path))
        if index == None: index = {}
        if index.get(sector_name) == None:
            index[sector_name] = {} 
        readable_time = datetime.datetime.fromtimestamp(time.time())

        index[sector_name][embed_data_name] = {
            'embed_data_name': embed_data_name,
            'data_lenth': len(text_list),
            'time': readable_time,
            'embedding_model_type': 'Qwen3-Embedding-0.6B-MNN',
            'embed_json_path': f'{self.db_path}/{sector_name}/{embed_data_name}.json',
            'embed_pkl_path': f'{self.db_path}/{sector_name}/{embed_data_name}.pkl',
        }

        write_yaml_file(os.path.join(self.db_path, self.index_path), index)

    def search_similar_texts(self, query_text, pkl_data, mode: Literal["topk", "threshold", "all"] = "all", top_k=5, threshold=0.6):
        query_emb = self.embedding_text(query_text)
        similarity_list = []
        for idx in pkl_data:
            data_emb = pkl_data[idx]['embedding']
            sim = self.cosine_similarity(query_emb, data_emb)
            similarity_list.append((sim, pkl_data[idx]['text']))
        similarity_list.sort(key=lambda x: x[0], reverse=True)

        if mode == "topk":
            return similarity_list[:top_k]
        elif mode == "threshold":
            return [s for s in similarity_list if s[0] > threshold]
        elif mode == "all":
            return [s for s in similarity_list[:top_k] if s[0] > threshold]



