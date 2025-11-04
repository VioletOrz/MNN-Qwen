
import MNN.llm as llm
import MNN.cv as cv
import MNN.numpy as np
import sys
from typing import Literal
import re


class Qwen3VL:

    def __init__(self, model_path, history_window = 4):
        self.model = llm.create(model_path)
        self.model.load()
        self.history = {
            'text': [],
            'image': []
        }
        self.history_window = history_window


    def generate(self, prompt, max_len = 512):

        # print('user: ' + prompt['text'])
        prompt['text'] = self.model.apply_chat_template(prompt['text'])
        ids = self.model.tokenizer_encode(prompt)
        self.model.generate_init()
        logits = self.model.forward(ids)
        token = np.argmax(logits)
        self.model.context.current_token = token
        word = self.model.tokenizer_decode(token)
        print('assistant: ' + word, end='', flush=True)

        response = "" + word
        for i in range(max_len):
            logits = self.model.forward(token)
            token = np.argmax(logits)
            self.model.context.current_token = token
            if self.model.stoped():
                break
            try:
                word = self.model.tokenizer_decode(token)
            except Exception as e:
                # print(f"解码token {token} {type(token)} 时出错: {str(e)}")
                word = ""
            response += word
            print(word, end='', flush=True)
        print("\n")
        return response

    def preprocess_image(self, image_path, limit_size=720, mode: Literal["max", "min"] = "max"):
        image = cv.imread(image_path)
        if mode == "max":
            new_w, new_h, scale = self.resize_to_max_edge(image.shape[1], image.shape[0], limit_size)
        else:
            new_w, new_h, scale = self.resize_to_min_edge(image.shape[1], image.shape[0], limit_size)
        image = cv.resize(image, (new_w, new_h))
        return image

    def chat(self, user_prompt, system_prompt = None, image_path = None, max_len = 512, image_max_size=512, mode: Literal["max", "min"] = "max"):
        
        if system_prompt != None:
            print('system: ' + system_prompt)
        print('user: ' + user_prompt)

        image_id = len(self.history['image'])
        if system_prompt != None:
            context = [
                {
                    'role': 'system',
                    'content': system_prompt
                }
            ]
        else:
            context = []

        for text in self.history['text']:
            if text['role'] == 'system' and system_prompt != None:
                continue
            context.append(text)

        prompt_text = "历史对话:{"
        for c in context:
            role = c['role']
            content = c['content']
            prompt_text += f'\"role\": \"{role}\", \"content\": \"{content}\"\n'
        
        prompt_text += "}\n"
        prompt_text += f'{user_prompt if image_path is None else f"<img>image_{image_id}</img> " + user_prompt}\n'

        prompt_image = []
        for i, image in enumerate(self.history['image']):
            prompt_image.append(
                {
                    'data': image['data'],
                    'height': image['height'],
                    'width': image['width']
                }
            )
        
        if image_path is not None:
            image = self.preprocess_image(image_path, image_max_size, mode)
            prompt_image.append(
                {
                    'data': image,
                    'height': image.shape[0],
                    'width': image.shape[1],
                }
            )
        else:
            image = None
        

        prompt_full = {
            'text': prompt_text,
            'images': prompt_image
        }

        response = self.generate(prompt_full, max_len)
        self.history_updata(user_prompt, system_prompt, image, response)
        return response


    def history_updata(self, user_prompt, system_prompt, image, response):
        self.history['text'].append(
            {
                'role': 'user',
                'content': user_prompt if image is None else f"<img>image_{len(self.history['image'])}</img> " + user_prompt
            }
        )
        self.history['text'].append(
            {
                'role': 'assistant',
                'content': response
            }
        )
        if image is not None:
            self.history['image'].append(
                {
                    'data': image,
                    'height': image.shape[0],
                    'width': image.shape[1],
                }
            )
        if system_prompt is not None:
            if self.history['text'][0]['role'] == 'system':
                self.history['text'][0]['content'] = system_prompt
            else:
                self.history['text'].insert(
                    0,
                    {
                        'role': 'system',
                        'content': system_prompt
                    }
                )

        if self.history['text'][0]['role'] == 'system': windows_size = self.history_window + 1

        if len(self.history['text']) > windows_size:
            if self.history['text'][0]['role'] == 'system':
                new_history = [self.history['text'][0]] + self.history['text'][-self.history_window:]
                self.history['text'] = new_history
            else:
                self.history['text'] = self.history['text'][-self.history_window:]
        
        def reindex_image_tags_in_history(history_list):
            """
            在形如 [{'role':..., 'content':...}, ...] 的列表中，
            重新编号所有 <img>image_N</img> 标签，使它们从 0 开始连续递增。

            Args:
                history_list (list[dict]): 包含 'content' 字段的字典列表

            Returns:
                new_history (list[dict]): 替换后的新列表（浅拷贝）
                total_count (int): 替换的标签总数
            """
            pattern = r"<img>image_(\d+)</img>"
            counter = 0
            new_history = []

            def repl(match):
                nonlocal counter
                new_tag = f"<img>image_{counter}</img>"
                counter += 1
                return new_tag

            for entry in history_list:
                if not isinstance(entry, dict) or "content" not in entry:
                    new_history.append(entry)
                    continue
                text = entry["content"]
                new_text, _ = re.subn(pattern, repl, text)
                new_entry = entry.copy()
                new_entry["content"] = new_text
                new_history.append(new_entry)

            return new_history, counter

        new_history, total = reindex_image_tags_in_history(self.history['text'])
        self.history['text'] = new_history
        self.history['image'] = self.history['image'][-total:]

        print('history:##############################################')   
        for item in new_history:
            print(item)

        print("总共重编号:", total)
        print('######################################################') 
        pass
        
        
    def resize_to_max_edge(self, width, height, max_edge):

        max_dim = max(width, height)
        if max_dim <= max_edge:
            return int(width), int(height), 1.0

        scale = max_edge / max_dim
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        return new_w, new_h, scale

    def resize_to_min_edge(self, width, height, min_edge):

        max_dim = min(width, height)
        if max_dim <= min_edge:
            return int(width), int(height), 1.0

        scale = min_edge / max_dim
        new_w = int(round(width * scale))
        new_h = int(round(height * scale))
        return new_w, new_h, scale
    
def generate_example(model, prompt, max_len = 512):

    prompt['text'] = model.apply_chat_template(prompt['text'])
    ids = model.tokenizer_encode(prompt)
    model.generate_init()
    logits = model.forward(ids)
    token = np.argmax(logits)
    model.context.current_token = token
    word = model.tokenizer_decode(token)
    print(word, end='', flush=True)

    response = "" + word
    for i in range(max_len):
        logits = model.forward(token)
        token = np.argmax(logits)
        model.context.current_token = token
        if model.stoped():
            break
        try:
            word = model.tokenizer_decode(token)
        except Exception as e:
            # print(f"解码token {token} {type(token)} 时出错: {str(e)}")
            word = ""
        response += word
        print(word, end='', flush=True)

    return response


# def response_example(model, prompt):
#     # response stream
#     model.response(prompt, True)
#     vision_us = model.context.vision_us
#     prefill_us = model.context.prefill_us
#     decode_us = model.context.decode_us
#     prompt_len = model.context.prompt_len
#     decode_len = model.context.gen_seq_len
#     pixels_mp = model.context.pixels_mp
#     print('pixels : {}'.format(pixels_mp))
#     print('vision time : {} ms'.format(vision_us / 1000.0))
#     print('prefill speed : {} token/s'.format(prompt_len / (prefill_us / 1000000.0)))
#     print('decode speed : {} token/s'.format(decode_len / (decode_us / 1000000.0)))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python vllm_example.py <path_to_model_config>')
        exit(1)

    config_path = sys.argv[1]
    # create model
    model = llm.create(config_path)
    # load model
    model.load()


    img_path = '../../../resource/images/cat.jpg'
    img = cv.imread(img_path)

    prompt = {
        'text': '<img>image_0</img>介绍一下这张图',
        'images': [
            {
                'data': img,
                'height': 420,
                'width': 420
            }
        ]
    }

    # response_example(model, prompt)
    generate_example(model, prompt)