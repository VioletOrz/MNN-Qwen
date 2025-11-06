from models.Qwen3embedding import Qwen3Embedding
from tools.Violet_base import *
from data.data_manager import DataManager

if __name__ == '__main__':
    # qwen3_embedding = Qwen3Embedding('models/ckpt/Qwen3-Embedding-0.6B-MNN/config.json')
    # qwen3_embedding.embedding_text_file_and_save_to_json_and_pkl('test_01.txt', 'text_sector', 'test_01')
    # qwen3_embedding.embedding_text_file_and_save_to_json_and_pkl('test_txt.txt', 'text_sector', 'test_txt')
    data_manager = DataManager()
    data_manager.list_and_print_sectors()
    print("========================================")
    data_manager.list_and_print_sector_and_data_names()
    print("========================================")
    data_manager.list_and_print_data_name_in_sector('text_sector')
    print("========================================")
    data_manager.print_json_data('text_sector', 'test_txt')
