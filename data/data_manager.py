from tools.Violet_base import *
import os

class DataManager:
    def __init__(self, db_path = 'data', index_path = 'index.yaml'):
        self.db_path = db_path

        try:
            self.index = read_yaml_file(os.path.join(db_path, index_path))
        except Exception as e:
            print(e)
            print("未找到索引文件，正在创建空的索引文件...")
            self.index = {}
            write_yaml_file(os.path.join(db_path, index_path), self.index)
            print("已创建索引文件")

        if self.index == None:
            self.index = {}
        
    def list_and_print_sectors(self):
        keys = self.index.keys()
        for key in keys:
            print(f"Sector: {key}")
        return keys
    
    def list_and_print_sector_and_data_names(self):
        sector_data = {}
        for sector_name in self.index.keys():
            data_names = list(self.index[sector_name].keys())
            sector_data[sector_name] = data_names
            print(f"Sector: {sector_name}")
            for data_name in data_names:
                print(f"    Data Name: {data_name}, Data Time: {self.index[sector_name][data_name]['time']}, Data Length: {self.index[sector_name][data_name]['data_lenth']}")
        return sector_data
    
    def list_and_print_data_name_in_sector(self, sector_name):
        if sector_name not in self.index:
            print(f"Sector {sector_name} not found.")
            return []
        data_names = list(self.index[sector_name].keys())
        for data_name in data_names:
            print(f"Data Name: {data_name}, Data Time: {self.index[sector_name][data_name]['time']}, Data Length: {self.index[sector_name][data_name]['data_lenth']}")
        return data_names
    
    def list_json_data(self, sector_name, data_name):
        if sector_name not in self.index:
            print(f"Sector {sector_name} not found.")
            return None
        if data_name not in self.index[sector_name]:
            print(f"Data Name {data_name} not found in Sector {sector_name}.")
            return None
        json_path = self.index[sector_name][data_name]['embed_json_path']
        json_data = read_json_file(json_path)

        return json_data

    def print_json_data(self, sector_name, data_name, page_number=1, page_size=10):
        json_data = self.list_json_data(sector_name, data_name)
        if json_data is None:
            return
        if page_number < 1 or page_size < 1:
            print("Page number or page size must be positive integers.")
            return

        start_idx = (page_number - 1) * page_size
        end_idx = start_idx + page_size
        keys = list(json_data.keys())

        print(f"Showing page {page_number} of {len(keys) // page_size + 1}, page size: {page_size}")
        for idx in range(start_idx, min(end_idx, len(keys))):
            key = keys[idx]
            print(f"Index: {key}\nText: {json_data[key]['text']}")

    def delete_data(self, sector_name, data_name):
        json_path = self.index[sector_name][data_name]['embed_json_path']
        pkl_path = self.index[sector_name][data_name]['embed_pkl_path']
        os.remove(json_path)
        print(f"Deleted JSON file: {json_path}")
        os.remove(pkl_path)
        print(f"Deleted PKL file: {pkl_path}")
        del self.index[sector_name][data_name]
        write_yaml_file(os.path.join(self.db_path, 'index.yaml'), self.index)
        print(f"Deleted data: {data_name}, in sector: {sector_name}")
        
    def delete_sector(self, sector_name):
        for data_name in self.index[sector_name]:
            self.delete_data(sector_name, data_name)
        del self.index[sector_name]
        write_yaml_file(os.path.join(self.db_path, 'index.yaml'), self.index)
        print(f"Deleted sector: {sector_name}")

