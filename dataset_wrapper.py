import torch
from torch.utils.data import Dataset

class DatasetWrapper(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.original_dataset = original_dataset
        
        if not hasattr(self.original_dataset, 'collate_fn'):
            # Fallback to the original dataset's module level collate_fn if we pass it later
            pass

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        item_dict = self.original_dataset[index]
        if not isinstance(item_dict, dict):
            raise TypeError(f"Wrapped dataset {type(self.original_dataset)} __getitem__ must return a dict.")
            
        item_dict['index'] = index
        return item_dict

def collate_keep_boxes_with_index(batch):
    from dataset import collate_keep_boxes
    
    valid_batch = [d for d in batch if d is not None]
    if not valid_batch:
        return {} 

    indices = [d.pop('index') for d in valid_batch]
    
    # Delegate to the original collate function
    collated_data = collate_keep_boxes(valid_batch)

    # Attach the indices back
    collated_data['indices'] = torch.tensor(indices, dtype=torch.long)
    return collated_data
