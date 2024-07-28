import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

class HCPT1wDataset(Dataset):
    def __init__(self, file_list, conditioning_file=None, with_conditioning=False, transform=None, compute_dtype=torch.float32):
        self.file_list = file_list
        self.transform = transform
        self.with_conditioning = with_conditioning
        self.compute_dtype = compute_dtype
        
        if self.with_conditioning and conditioning_file:
            self.conditioning_data = self.load_conditioning_data(conditioning_file)
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.fit_encoder()
        else:
            self.conditioning_data = None

    def load_conditioning_data(self, conditioning_file):
        conditioning_df = pd.read_csv(conditioning_file, sep='\t', quotechar='"')
        # Normalize 'subjectkey' by removing underscores for matching
        conditioning_df['subjectkey'] = conditioning_df['subjectkey'].str.replace('_', '')
        # Check for duplicates in the 'subjectkey' column
        if conditioning_df['subjectkey'].duplicated().any():
            print("Warning: Duplicate subjectkey values found. Handling duplicates by keeping the first occurrence.")
            conditioning_df = conditioning_df.drop_duplicates(subset='subjectkey', keep='first')
        
        # Create a dictionary with 'subjectkey' as the key for fast lookup
        conditioning_dict = conditioning_df.set_index('subjectkey').to_dict(orient='index')
        return conditioning_dict
    
    def fit_encoder(self):
        # We only want to encode the 'sex' column
        self.string_cols = ['sex']
        sample_df = pd.DataFrame([data for data in self.conditioning_data.values()])
        self.encoder.fit(sample_df[self.string_cols])

    def encode_condition(self, condition):
        # Extract numeric and string data
        interview_age = torch.tensor([float(condition['interview_age'])], dtype=self.compute_dtype)
        string_data = {'sex': condition['sex']}
        
        string_df = pd.DataFrame([string_data])
        one_hot_tensor = torch.tensor(self.encoder.transform(string_df[self.string_cols]).astype(float)).squeeze(0).to(self.compute_dtype)
        
        return torch.cat((interview_age, one_hot_tensor), dim=0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        data = {'image': img_path}
        
        if self.transform:
            data = self.transform(data)
        
        if self.with_conditioning and self.conditioning_data:
            subject_key = self.extract_subject_key(img_path)
            if subject_key in self.conditioning_data:
                condition = self.conditioning_data[subject_key]
                condition_tensor = self.encode_condition(condition)
                condition_tensor = condition_tensor.unsqueeze(0)  # Shape (1, 1, ContextDim)
                data['condition'] = condition_tensor
            else:
                print(f"Conditioning data not found for subject: {subject_key}")
        
        return data
    
    def extract_subject_key(self, img_path):
        # Convert Path object to string
        img_path_str = str(img_path)
        # Assuming the subject key is embedded in the file path as 'sub-<subjectkey>'
        parts = img_path_str.split('/')
        for part in parts:
            if part.startswith('sub-'):
                return part.split('-')[1]
        return None

def main(directory, conditioning_file):
    pattern = os.path.join(directory, 'sub-NDARINV*/ses-baselineYear1Arm1/anat/*baselineYear1Arm1_run-01_T1w.nii')
    file_paths = glob.glob(pattern)
    file_paths = [Path(fp) for fp in file_paths]  # Convert file paths to Path objects
    
    if not file_paths:
        print(f"No files found matching the pattern: {pattern}")
        return
    
    dataset = HCPT1wDataset(file_paths, conditioning_file=conditioning_file, with_conditioning=True, compute_dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Adjust batch size as needed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for batch in dataloader:
        images = [item['image'] for item in batch]
        
        try:
            conditions = [item['condition'] for item in batch if 'condition' in item]
            # Concatenate all conditions along the batch dimension
            if conditions:
                condition_tensor = torch.cat(conditions, dim=0).to(device)
                condition_tensor = condition_tensor.to(torch.float32)  # Ensure the condition tensor is float32
                # print("Condition tensor shape:", condition_tensor.shape)
            else:
                print("No conditioning data in the batch.")
        except Exception as e:
            print("Error processing batch conditions:", e)
        
        # Perform your training steps with images and condition_tensor
        # For debugging, we just print the condition tensor shape
        # Replace this with your actual training code

# # Example usage
# directory = "/path/to/your/dataset"
# conditioning_file = "/path/to/your/conditioning_file.csv"
# main(directory, conditioning_file)

        # Perform your training steps with images and condition_tensor
        # For debugging, we just print the condition tensor shape
        # Replace this with your actual training code

# # Example usage
# directory = "/path/to/your/dataset"
# conditioning_file = "/path/to/your/conditioning_file.csv"
# main(directory, conditioning_file)

# # Example usage
# directory = "/path/to/your/dataset"
# conditioning_file = "/path/to/your/conditioning_file.csv"
# main(directory, conditioning_file)

# from torch.utils.data import Dataset
# from pathlib import Path

# import os
# import glob
# import pandas as pd
# from torch.utils.data import Dataset

# class HCPT1wDataset(Dataset):
#     def __init__(self, file_list, conditioning_file=None, with_conditioning=False, transform=None):
#         self.file_list = file_list
#         self.transform = transform
#         self.with_conditioning = with_conditioning
        
#         if self.with_conditioning and conditioning_file:
#             self.conditioning_data = self.load_conditioning_data(conditioning_file)
#         else:
#             self.conditioning_data = None

#     def load_conditioning_data(self, conditioning_file):
#         conditioning_df = pd.read_csv(conditioning_file, sep='\t')
#         conditioning_df['subjectkey'] = conditioning_df['subjectkey'].str.replace('_', '')
#         # Check for duplicates in the 'subjectkey' column
#         if conditioning_df['subjectkey'].duplicated().any():
#             print("Warning: Duplicate subjectkey values found. Handling duplicates by keeping the first occurrence.")
#             conditioning_df = conditioning_df.drop_duplicates(subset='subjectkey', keep='first')
        
#         # Create a dictionary with 'subjectkey' as the key for fast lookup
#         conditioning_dict = conditioning_df.set_index('subjectkey').to_dict(orient='index')
#         return conditioning_dict

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         img_path = self.file_list[idx]
#         data = {'image': img_path}
        
#         if self.transform:
#             data = self.transform(data)
        
#         if self.with_conditioning and self.conditioning_data:
#             subject_key = self.extract_subject_key(img_path)
#             if subject_key in self.conditioning_data:
#                 data['context'] = self.conditioning_data[subject_key]
#             else:
#                 print(f"Conditioning data not found for subject: {subject_key}")
        
#         return data
    
#     def extract_subject_key(self, img_path):
#         img_path = str(img_path)
#         # Assuming the subject key is embedded in the file path as 'sub-<subjectkey>'
#         parts = img_path.split('/')
#         for part in parts:
#             if part.startswith('sub-'):
                
#                 return part.split('-')[1]
#         return None
# # class HCPT1wDataset(Dataset):
# #     def __init__(self, file_list, conditioning_file=None, with_conditioning=False, transform=None):
# #         self.file_list = file_list
# #         self.transform = transform
# #         self.with_conditioning = with_conditioning
        
# #         if self.with_conditioning and conditioning_file:
# #             self.conditioning_data = self.load_conditioning_data(conditioning_file)
# #         else:
# #             self.conditioning_data = None

# #     def load_conditioning_data(self, conditioning_file):
# #         conditioning_df = pd.read_csv(conditioning_file, sep='\t')
# #         # Create a dictionary with 'subjectkey' as the key for fast lookup
# #         conditioning_dict = conditioning_df.set_index('subjectkey').to_dict(orient='index')
# #         return conditioning_dict

# #     def __len__(self):
# #         return len(self.file_list)

# #     def __getitem__(self, idx):
# #         img_path = self.file_list[idx]
# #         data = {'image': img_path}
        
# #         if self.transform:
# #             data = self.transform(data)
        
# #         print("self.conditioning_data", self.conditioning_data)
# #         if self.with_conditioning and self.conditioning_data:
# #             subject_key = self.extract_subject_key(img_path)
# #             if subject_key in self.conditioning_data:
# #                 data['context'] = self.conditioning_data[subject_key]
# #             else:
# #                 print(f"Conditioning data not found for subject: {subject_key}")
        
# #         return data
    
# #     def extract_subject_key(self, img_path):
# #         # Assuming the subject key is embedded in the file path as 'sub-<subjectkey>'
# #         parts = img_path.split('/')
# #         for part in parts:
# #             if part.startswith('sub-'):
# #                 return part.split('-')[1]
# #         return None

# # def main(directory, conditioning_file):
# #     pattern = os.path.join(directory, 'sub-NDARINV*/ses-baselineYear1Arm1/anat/*baselineYear1Arm1_run-01_T1w.nii')
# #     file_paths = glob.glob(pattern)
    
# #     if not file_paths:
# #         print(f"No files found matching the pattern: {pattern}")
# #         return
    
# #     dataset = HCPT1wDataset(file_paths, conditioning_file=conditioning_file, with_conditioning=True)
# #     for i in range(len(dataset)):
# #         data = dataset[i]
# #         print(data)  # You can replace this with actual batch processing code

# # if __name__ == "__main__":
# #     import sys
# #     if len(sys.argv) != 3:
# #         print("Usage: python convert_to_nii_gz.py <directory> <conditioning_file>")
# #         sys.exit(1)
    
# #     directory = sys.argv[1]
# #     conditioning_file = sys.argv[2]
# #     main(directory, conditioning_file)



# # class HCPT1wDataset(Dataset):
# #     def __init__(self, file_list, with_conditioning=False, transform=None):
# #         self.file_list = file_list
# #         self.transform = transform
# #         self.with_conditioning = with_conditioning

# #     def __len__(self):
# #         return len(self.file_list)


# #     def __getitem__(self, idx):
# #         img_path = self.file_list[idx]
# #         data = {'image': img_path}
        
# #         if self.transform:
# #             data = self.transform(data)
        
# #         if self.with_conditioning:
# #             data['condition']= 
# #         return data
    
    
    
#     # def __getitem__(self, idx):
#     #     img_path = self.file_list[idx]
#     #     data = {'image': img_path}

#     #     print(f"Image path: {img_path}")

#     #     if self.transform:
#     #         for t in self.transform.transforms:
#     #             data = t(data)
#     #             if isinstance(data['image'], torch.Tensor):
#     #                 print(f"After {t.__class__.__name__}: {data['image'].shape}")
#     #             elif isinstance(data['image'], PIL.Image.Image):
#     #                 print(f"After {t.__class__.__name__}: {data['image'].size}")
#     #             elif isinstance(data['image'], np.ndarray):
#     #                 print(f"After {t.__class__.__name__}: {data['image'].shape}")

#     #     return data





# # import random
# # def create_train_val_datasets(base_dir, train_transform,val_transform, val_ratio=0.2, seed=42):
# #     base_path = Path(base_dir)
# #     # all_files = list(base_path.rglob('*/**/3T/T1w_MPR1/*_3T_T1w_MPR1.nii.gz'))
# #     all_files = list(base_path.rglob('*/3T/T1w_MPR1/*_3T_T1w_MPR1.nii.gz'))
    
# #     # Shuffle the file list
# #     random.seed(seed)
# #     random.shuffle(all_files)
    
# #     # Split into train and validation
# #     val_size = int(len(all_files) * val_ratio)
# #     train_files = all_files[val_size:]
# #     val_files = all_files[:val_size]
    
# #     # Create datasets
# #     train_dataset = HCPT1wDataset(train_files, train_transform)
# #     val_dataset = HCPT1wDataset(val_files,val_transform)
    
# #     return train_dataset, val_dataset

