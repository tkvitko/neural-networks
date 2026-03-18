import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        """
        Args:
            images: numpy array shape (N, H, W, C)
            masks: numpy array shape (N, H, W, num_classes)
            transform: transforms для аугментации (опционально)
        """
        self.images = images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Получаем изображение и маску
        image = self.images[idx]  # (572, 572, 1)
        mask = self.masks[idx]    # (572, 572, 2)
        
        # Преобразуем в тензоры и меняем размерность
        # из (H, W, C) в (C, H, W) для PyTorch
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # (1, 572, 572)
        
        mask = torch.from_numpy(mask).float()
        mask = mask.permute(2, 0, 1)    # (2, 572, 572)
        
        # Если маска должна быть классовыми индексами, а не one-hot
        mask = torch.argmax(mask, dim=0)  # (572, 572)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask