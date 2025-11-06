import MNN.llm as llm
import MNN
import numpy as np
import os
from tools.Violet_base import *
import time
import datetime

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
            'embed_json_path': f'{self.db_path}/{sector_name}/{embed_data_name}.json',
            'embed_pkl_path': f'{self.db_path}/{sector_name}/{embed_data_name}.pkl',
        }

        write_yaml_file(os.path.join(self.db_path, self.index_path), index)



