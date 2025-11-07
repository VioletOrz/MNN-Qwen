import easyocr
import difflib
from models.Qwen3vl import Qwen3VL, clean_llm_output
import os

def write_json_file(json_path:str, data):
    """
    将数据写入json文件
    json_path: json文件的路径
    data: 需要写入的数据
    """
    import json

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json_file(json_path:str):
    """
    读取json文件并返回数据
    json_path: json文件的路径
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 解析 JSON 文件内容为 Python 数据结构
    return data


def ocr_image(image_path, threshold=0.6):
    # 初始化中英文 OCR（会自动检测 GPU）
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    # 识别一张图片
    results = reader.readtext(image_path)

    # 输出结果
    ocr_list = []
    ocr_text = ""
    for (bbox, text, prob) in results:
        print(f"{text} ({prob:.3f})")
        if prob > 0:

            max_similarity = 0.0
            for ocr in ocr_list:
                similarity = difflib.SequenceMatcher(None, ocr, text).ratio()
                if similarity > max_similarity:
                    max_similarity = similarity
            # print(f"max_similarity: ({max_similarity:.3f})")   
            if max_similarity < threshold:    
                ocr_list.append(text)
                ocr_text += text + "\n"  

    return ocr_list, ocr_text          

def scene_desc(image_path):
    # ocr_list, ocr_text = ocr_image(image_path)
    prompt_text = '请结合图像内容，简洁的介绍画面中的内容。' #\nocr结果：\n' + ocr_text
    config_path = 'models/ckpt/Qwen3-VL-4B-Instruct-MNN/config.json'
    model = Qwen3VL(config_path, history_window=0)
    # image = model.preprocess_image(image_path, limit_size=720)
    # prompt = {
    #     "text": prompt_text,
    #     "image": [
    #         {
    #             "data": image,
    #             "height": image.shape[0],
    #             "width": image.shape[1]
    #         }
    #     ]
    # }
    
    response = model.chat(user_prompt=prompt_text, system_prompt="你是一个画面分析助手，请结合图像中的文字对画面中的场景进行简洁的介绍。", image_path=image_path, max_len=128, image_max_size=720, mode="max")                            
    print("clean_llm_output########################################################")
    print(clean_llm_output(response))
    return clean_llm_output(response)
if __name__ == '__main__':
    image_dir = r"Hello_World_cn_Frames"
    path_list = os.listdir(image_dir)
    json_path = "scene_desc_results_srhw.json"
    if os.path.exists(json_path):
        json_data = read_json_file(json_path)
    else:
        json_data = []
    for path in path_list:
        res = scene_desc(os.path.join(image_dir, path))          
        json_data.append({
            "image_path": path,
            # "ocr": ocr,
            "response": res
        })  
        write_json_file(json_path, json_data)                                                                                                                                                                                                                                                                                                          