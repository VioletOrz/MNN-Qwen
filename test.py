from models.Qwen3VL import Qwen3VL
import re
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


if __name__ == '__main__':
 
    config_path = 'models/ckpt/Qwen3-VL-4B-Instruct-MNN/config.json'
    model = Qwen3VL(config_path)

    # model.chat(user_prompt="你的名字是什么。", system_prompt="你叫NEWQ是一个有帮助的AI助手。", max_len=512, image_max_size=720, mode="max")
    # model.chat(user_prompt="这个图片里的角色是男是女。", system_prompt="你是一个有帮助的AI助手。", image_path="1.png", max_len=512, image_max_size=400, mode="max")
    # model.chat(user_prompt="我的上个问题是什么。", max_len=512, image_max_size=720, mode="max")
    # model.chat(user_prompt="这个图片里的角色是男是女。", system_prompt="你是一个有帮助的AI助手。", image_path="1.jpg", max_len=512, image_max_size=400, mode="max")

    model.chat(user_prompt="你的名字是什么。", system_prompt="你叫NEWQ是一个有帮助的AI助手。", max_len=512, image_max_size=720, mode="max")
    model.chat(user_prompt="我的上个问题是什么。", max_len=512, image_max_size=720, mode="max")
    model.chat(user_prompt="你为什么会有一个这么奇怪的名字？。", max_len=512, image_max_size=720, mode="max")

    # history = [
    #     {"role": "user", "content": "图一 <img>image_9</img>"},
    #     {"role": "assistant", "content": "返回 <img>image_12</img> <img>image_13</img>"},
    #     {"role": "user", "content": "图三 <img>image_1</img>"},
    # ]

    # new_history, total = reindex_image_tags_in_history(history)

    # for item in new_history:
    #     print(item)

    # print("总共重编号:", total)