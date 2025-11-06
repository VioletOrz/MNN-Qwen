##########################################################################
#1.设置可用的GPU
#2.基础yaml/json读写
##########################################################################

#========================================================================#
#1.1 设置可用的GPU

def set_cuda_visible_gpu(dev:list):
    """
    设置可用的GPU
    dev:可用GPU列表 内部成员为gpu编号
    """
    import os

    dev_str = ''
    for i in range(dev):
        dev_str = dev_str + str(i) + ','
    dev_str = dev_str[:-1]

    os.environ["CUDA_VISIBLE_DEVICES"] = dev_str

#========================================================================#
#2.1 读取json文件
def read_json_file(json_path:str):
    """
    读取json文件并返回数据
    json_path: json文件的路径
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 解析 JSON 文件内容为 Python 数据结构
    return data

#2.2 读取yaml文件
def read_yaml_file(yaml_path:str):
    """
    读取yaml文件并返回数据
    yaml_path: yaml文件的路径
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

#2.3 写入json文件
def write_json_file(json_path:str, data):
    """
    将数据写入json文件
    json_path: json文件的路径
    data: 需要写入的数据
    """
    import json

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

#2.4 写入yaml文件
def write_yaml_file(yaml_path:str, data):
    """
    将数据写入yaml文件
    yaml_path: yaml文件的路径
    data: 需要写入的数据
    """

    import yaml

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, allow_unicode=True, default_flow_style=False)

#2.5 读取txt文件
def read_txt_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

#2.6 写入txt文件
def write_txt_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)


def read_pkl_file(path):
    import pickle
    with open(path, "rb") as file:  # "rb" 表示以二进制读取
        data = pickle.load(file)
    return data

def write_pkl_file(path, data):
    import pickle
    with open(path, "wb") as file:  # "wb" 表示以二进制写入
        pickle.dump(data, file)

#========================================================================#
import os

def list_all_files(folder_path):
    """
    列出给定文件夹下所有文件的文件名（包括子文件夹内的文件）。

    :param folder_path: 文件夹路径
    :return: 文件名列表
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def list_immediate_files_and_folders(folder_path):
    """
    列出给定文件夹下一级的文件和文件夹名。

    :param folder_path: 文件夹路径
    :return: 文件和文件夹名列表
    """
    if not os.path.exists(folder_path):
        return []
    return [os.path.join(folder_path, item) for item in os.listdir(folder_path)]


def list_immediate_subfolders(folder_path):
    """
    列出给定文件夹下一级的子文件夹名。

    :param folder_path: 文件夹路径
    :return: 子文件夹名列表
    """
    if not os.path.exists(folder_path):
        return []
    return [os.path.join(folder_path, item) for item in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, item))]

def get_parent_and_grandparent_dir(file_path):
    # 获取父文件夹路径
    parent_dir = os.path.dirname(file_path)
    # 获取父文件夹名称
    parent_name = os.path.basename(parent_dir)
    # 获取祖父文件夹路径
    grandparent_dir = os.path.dirname(parent_dir)
        # 获取祖父文件夹名称
    grandparent_name = os.path.basename(grandparent_dir)

    base_name = os.path.basename(file_path)
    return base_name, parent_name, grandparent_name


#========================================================================#

def calculate_mean_and_variance(data):
    import numpy as np
    """
    计算 Python list 或 NumPy ndarray 的均值和方差。

    Args:
        data (list or numpy.ndarray): 输入数据。 可以是一维列表，或者任意维度的 NumPy 数组。

    Returns:
        tuple or None:  包含 (mean, variance) 的元组， 如果输入数据为空或无效， 则返回 None
    """

    if not data:
        print("Error: Input data cannot be empty.")
        return None
    try:
      # 1. 转换为 NumPy 数组
      data = np.array(data, dtype=float) # 使用 float 类型避免整数运算误差

      # 2. 检查数据是否为空
      if data.size == 0:
          print("Error: Input data cannot be empty.")
          return None

      # 3. 计算均值和方差
      mean = np.mean(data)
      variance = np.var(data)

      return mean, variance
    except Exception as e:
       print(f"Error: Input data invalid. {e}")
       return None