from models.Qwen3embedding import Qwen3Embedding
from models.Qwen3vl import Qwen3VL
from tools.Violet_base import *
from data.data_manager import DataManager

if __name__ == '__main__':
    qwen3_embedding = Qwen3Embedding('models/ckpt/Qwen3-Embedding-0.6B-MNN/config.json')
    # qwen3_embedding.embedding_text_file_and_save_to_json_and_pkl('001.txt', 'test', '001')

    data_manager = DataManager()
    # r1 = data_manager.list_and_print_sectors()
    # print("========================================")
    # r2 = data_manager.list_and_print_sector_and_data_names()
    # print("========================================")
    # r3 = data_manager.list_and_print_data_name_in_sector('text_sector')
    # print("========================================")
    # r4 = data_manager.print_json_data('text_sector', 'test_txt')
    # print("========================================")
    # data = data_manager.get_merge_pkl_data([('text_sector', 'test_txt'), ('text_sector', 'test_01')])
    # pass

    qwen3vl = Qwen3VL('models/ckpt/Qwen3-VL-4B-Instruct-MNN/config.json', history_window=4)
    scenes = read_json_file('scene_desc_results_srhw.json')

    
    json_data = []
    pkl_data = data_manager.list_pkl_data('test', '001')
    system = "你是一个弹幕生成助手,请结合画面内容借鉴知识库中的网络热梗及其介绍给当前画面生成一条有趣的弹幕。"

    for scene in scenes:
        desc = scene['response']
        # qwen3vl.chat(prompt)
        print(qwen3_embedding.embedding_text(desc))
        prompt = "图像描述：" + desc + "\n请为该图像生成一条有趣的弹幕。仅输出一条简短的弹幕内容，不要包含其他说明。"
        knb = qwen3_embedding.search_similar_texts(desc, pkl_data, top_k=1, threshold=0.6)
        print(knb)
        if knb != []:
            knb_text = knb[0][1]
        else:
            knb_text = "无关联内容"
        knb_prompt = f"根据以下知识库内容回答问题：\n{knb_text}\n{prompt}"
        res = qwen3vl.chat(user_prompt=knb_prompt, system_prompt=system, max_len=256)
        json_data.append({
            "prompt": knb_prompt,
            "response": res
        })
        write_json_file('danmu_results_srhw.json', json_data)