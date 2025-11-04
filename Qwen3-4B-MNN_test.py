import MNN.llm as llm
import MNN.numpy as np

model = llm.create(r"F:\code\Qwen3\Qwen3-4B-Instruct-2507-MNN\llm.mnn")
model.load()

prompt = "请简要解释什么是量子力学。"
prompt = model.apply_chat_template(prompt)
ids = model.tokenizer_encode(prompt)
model.generate_init()
logits = model.forward(ids)
token = np.argmax(logits)
model.context.current_token = token
first_token = model.tokenizer_decode(token)
print("Token:", token)
print("First generated token:", first_token)
print(first_token, end="", flush=True)

while True:
    logits = model.forward(token)
    token = np.argmax(logits)
    model.context.current_token = token
    if model.stoped(): break
    try:
        word = model.tokenizer_decode(token)
    except Exception as e:
        print(f"解码{token}时出错:")
        word = ""
    print(word, end="", flush=True)

# import MNN.llm as llm
# import MNN.cv as cv, MNN.numpy as np
# import argparse, os 

# current_path = os.path.dirname(os.path.abspath(__file__))

# def generate_example(model, prompt):
#     prompt['text'] = model.apply_chat_template(prompt['text'])
#     ids = model.tokenizer_encode(prompt)
#     model.generate_init()
#     logits = model.forward(ids)
#     token = np.argmax(logits)
#     model.context.current_token = token
#     first_token = model.tokenizer_decode(token)
#     print(f"token: {token}, first_token: {first_token}")
#     print(first_token, end='', flush=True) # first token
#     while True:
#         logits = model.forward(token)
#         token = np.argmax(logits)
#         model.context.current_token = token
#         if model.stoped(): break
#         try:
#             word = model.tokenizer_decode(token)
#         except Exception as e:
#             print(f"解码token {token} {type(token)} 时出错: {str(e)}")
#             word = ""
#         print(word, end='', flush=True)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str, default=os.path.join(current_path, 'Qwen3-VL-4B-Instruct-MNN_Q8'))
#     parser.add_argument('--img_path', type=str, default=os.path.join(current_path, 'cat.jpg'))
#     args = parser.parse_args()
#     img_path = args.img_path; 
#     if not os.path.exists(img_path): print('image not exists'); exit(1)
#     config_path = os.path.join(args.model_path, 'config.json')
#     model = llm.create(config_path) # create model
#     model.load() # load model
#     img = cv.imread(img_path)
#     if img is None: print('read image failed'); exit(1)

#     prompt_list = [
#         "简要描述图片内容",
#         "详细描述图片内容",
#         "用自然语言简单描述图片",
#         "用自然语言详细描述图片",
#         "简单描述图片，减少形容词使用",
#     ]
#     for prompt_text in prompt_list:
#         prompt = {
#             'text': '<img>image_0</img>' + prompt_text,
#             'images': [
#                 { 'data': img, 'height': 420, 'width': 420 }
#             ]
#         }
#         generate_example(model, prompt)
#         print()
