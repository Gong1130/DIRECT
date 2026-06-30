import copy
import yaml
from torch.utils.data import Dataset, ConcatDataset

from .mvimgnet import MVImgNetDataset_Trellis, MVImgNetDataset_Trellis_MultiRes
from .sa1b import SA1BDataset_Trellis, SA1BDataset_Trellis_MultiRes


DATASET_CLASS_MAP = {
    "MVImgNetDataset_Trellis": MVImgNetDataset_Trellis,
    "MVImgNetDataset_Trellis_MultiRes": MVImgNetDataset_Trellis_MultiRes,
    "SA1BDataset_Trellis": SA1BDataset_Trellis,
    "SA1BDataset_Trellis_MultiRes": SA1BDataset_Trellis_MultiRes,
}

class ComposedDataset(Dataset):
    def __init__(self, config_path, dataset_class_map=DATASET_CLASS_MAP):
        
        self.config_path = config_path
        self.dataset_class_map = dataset_class_map
        self.datasets = []
        self.global_params = {}
        
        self._load_and_combine_datasets()
        
        if not hasattr(self, 'combined_dataset') or self.combined_dataset is None:
            raise RuntimeError("ComposedDataset initialization failed: no valid Dataset instances were created.")

    def _load_and_combine_datasets(self):
        """
        Loads configuration, merges global parameters with dataset-specific parameters,
        instantiates all datasets, and combines them using ConcatDataset.
        """
        if isinstance(self.config_path, str):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f) 
        elif isinstance(self.config_path, dict):
            config_dict = self.config_path
        else:
            raise ValueError("Config must be a file path (str) or a dictionary.")
        
        self.global_params = config_dict.get("global_params", {})
        config_list = config_dict.get("datasets", [])
        
        all_datasets = []
        
        print("--- Initializing Composed Dataset ---")
        print(f"Global Params loaded: {self.global_params}")
        
        for item_config in config_list:
            class_name = item_config.get("class")
            
            DatasetClass = self.dataset_class_map.get(class_name)
            if not DatasetClass:
                raise ValueError(f"Unknown Dataset class '{class_name}'. Available classes: {sorted(self.dataset_class_map)}")
            final_params = copy.deepcopy(self.global_params)
            final_params.update(item_config.get("params", {}))

            instance = DatasetClass(**final_params)
            all_datasets.append(instance)
            print(f"Loaded sub-dataset: {class_name}, final size: {len(instance)}")
                
        if not all_datasets:
            self.combined_dataset = None
            return

        self.combined_dataset = ConcatDataset(all_datasets)
        self.datasets = all_datasets 
        
        print("\n--- ConcatDataset successfully combined ---")
        print(f"Total samples across all datasets: {len(self.combined_dataset)}")

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, idx):
        return self.combined_dataset[idx]
