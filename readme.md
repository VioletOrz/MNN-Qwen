class models\Qwen3VL.py Qwen3VL

初始化参数，config_path为模型配置文件路径, history_window为滑动窗口大小，system_prompt被设置为全局共享，不占用窗口。一次QA对话占用2个窗口。

有部分tokens无法decode, 异常捕捉的提示目前被注释掉了。输入图像和某些prompt组合后，会反复输出相同的tokens，建议修改图像大小或者prompt。多个尺寸过大的图像输入也可能导致上述问题。图像大小建议在512*512以内，一轮对话窗口的图像数量在3以内。

.chat为聊天模式，基于self.history作为历史记录进行对话，输入user_prompt、system_prompt、image_path，其中user_prompt为必要参数。max_len，单次对话的输出tokens限制。image_max_size，限制图像的最大边长，超出会等比缩小。mode，限制模式，max为限制长边的最大边长，min为限制短边的最大边长。

.generate为生成模式，无历史记录。参考自官方文档，https://github.com/alibaba/MNN/blob/master/pymnn/examples/MNNLlm/vllm_exmaple.py

test.py为测试文件，测试聊天模式。

模型使用models/download.py下载到models/ckpt路径下。自行安装huggingface_hub/modelscope。

