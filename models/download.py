from huggingface_hub import snapshot_download
# from modelscope import snapshot_download

model_dir = snapshot_download("taobao-mnn/Qwen3-VL-4B-Instruct-MNN", local_dir="ckpt/Qwen3-VL-4B-Instruct-MNN")
print("模型已下载到：", model_dir)

model_dir = snapshot_download("taobao-mnn/Qwen3-Embedding-0.6B-MNN", local_dir="ckpt/Qwen3-Embedding-0.6B-MNN")
print("模型已下载到：", model_dir)