class models\Qwen3vl.py Qwen3VL

初始化参数，config_path为模型配置文件路径, history_window为滑动窗口大小，system_prompt被设置为全局共享，不占用窗口。一次QA对话占用2个窗口。

有部分tokens无法decode, 异常捕捉的提示目前被注释掉了。输入图像和某些prompt组合后，会反复输出相同的tokens，建议修改图像大小或者prompt。多个尺寸过大的图像输入也可能导致上述问题。图像大小建议在512*512以内，一轮对话窗口的图像数量在3以内。

.chat为聊天模式，基于self.history作为历史记录进行对话，输入user_prompt、system_prompt、image_path，其中user_prompt为必要参数。max_len，单次对话的输出tokens限制。image_max_size，限制图像的最大边长，超出会等比缩小。mode，限制模式，max为限制长边的最大边长，min为限制短边的最大边长。

.generate为生成模式，无历史记录。参考自官方文档，https://github.com/alibaba/MNN/blob/master/pymnn/examples/MNNLlm/vllm_exmaple.py

test.py为测试文件，测试聊天模式。

模型使用models/download.py下载到models/ckpt路径下。自行安装huggingface_hub/modelscope。

class models\Qwen3embedding.py Qwen3Embedding

用于编码文本到向量，并保存到数据库

初始化参数，config_path为模型配置文件路径, db_path为数据库根目录路径，index_path为数据库索引文件路径。

.embedding_text获取单条文本embedding。

.cosine_similarity计算余弦距离。

.text_list_embeddings批量编码文本，输入是文本列表。

.embedding_text_file_and_save_to_json_and_pkl，编码文本文件保存到json和pkl文件。输入文本文件路径，要保存的扇区名字，保存的文件名字。json文件和pkl文件会保存在db_path/sector_name/file_name.json和db_path/sector_name/file_name.pkl。其中json文件仅保存words和embedding，pkl文件保存文本和embed向量。

.search_similar_texts 搜索相似文本。输入为短文本，如果是长文本需要先分割再逐段匹配，返回相似文本列表。参数 pkl_data 为 data_manager 返回的pkl数据文件。mode 为匹配模式，限制topk，限制阈值，或者all。

class data/data_manager.py DataManager

数据库管理，查、删。

初始化参数，db_path为数据库根目录路径，index_path为数据库索引文件路径。

.list_and_print_sectors 列出数据库中的扇区。返回扇区名列表。

.list_and_print_sector_and_data_names 列出数据库中所有的扇区下的数据文件。返回字典，key为扇区名，value为其下数据文件名列表。

.list_and_print_data_name_in_sector 列出数据库中指定扇区下的数据文件。返回文件名列表。

.list_json_data 列出数据库中指定扇区下的json数据文件。但不会打印。

.list_pkl_data 同上。

.print_json_data 打印数据库中指定扇区下的json数据文件。输入为扇区名称、数据文件名、页码和每页大小。

.delete_data 删除数据库中指定扇区下的数据文件。输入为扇区名称、数据文件名。

.delete_sector 删除数据库中指定扇区。输入为扇区名称。

.get_merge_pkl_data 获取数据库中指定扇区下的数据文件并合并返回。输入为扇区名称、数据文件名。
