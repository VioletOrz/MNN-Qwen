
import MNN.llm as llm
import MNN.cv as cv
import MNN.numpy as np
import sys


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


def response_example(model, prompt):
    # response stream
    model.response(prompt, True)
    vision_us = model.context.vision_us
    prefill_us = model.context.prefill_us
    decode_us = model.context.decode_us
    prompt_len = model.context.prompt_len
    decode_len = model.context.gen_seq_len
    pixels_mp = model.context.pixels_mp
    print('pixels : {}'.format(pixels_mp))
    print('vision time : {} ms'.format(vision_us / 1000.0))
    print('prefill speed : {} token/s'.format(prompt_len / (prefill_us / 1000000.0)))
    print('decode speed : {} token/s'.format(decode_len / (decode_us / 1000000.0)))

def resize_to_max_edge(width, height, max_edge):

    max_dim = max(width, height)
    if max_dim <= max_edge:
        return int(width), int(height), 1.0

    scale = max_edge / max_dim
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    return new_w, new_h, scale

def resize_to_min_edge(width, height, min_edge):

    max_dim = min(width, height)
    if max_dim <= min_edge:
        return int(width), int(height), 1.0

    scale = min_edge / max_dim
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    return new_w, new_h, scale




if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('usage: python vllm_example.py <path_to_model_config>')
    #     exit(1)

    # config_path = sys.argv[1]

    config_path = 'Qwen3-VL-4B-Instruct-MNN/config.json'
    # create model
    model = llm.create(config_path)
    # load model
    model.load()

    img_path = '1.png'
    img = cv.imread(img_path)
    img = cv.resize(img, (512, 512))

    prompt = {
        'text': '历史对话:{\"role\": \"system\", \"content\": \"你叫NEWQ,是一个AI对话助手。\"} 你的名字是？',
        'images': [
            {
                'data': img,
                'height': 512,
                'width': 512
            }
        ]
    }

    response_example(model, prompt)
    generate_example(model, prompt)
