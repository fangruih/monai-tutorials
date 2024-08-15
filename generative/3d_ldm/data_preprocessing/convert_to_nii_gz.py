import nibabel as nib
import os
import sys
import glob

def convert_nii_to_nii_gz(file_path):
    try:
        img = nib.load(file_path)
        nii_gz_file_path = file_path + '.gz'
        nib.save(img, nii_gz_file_path)
        print(f"Converted {file_path} to {nii_gz_file_path}")
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")

def main(directory):
    pattern = os.path.join(directory, 'sub-NDARINV*/ses-baselineYear1Arm1/anat/*baselineYear1Arm1_run-01_T1w.nii')
    file_paths = glob.glob(pattern)
    
    if not file_paths:
        print(f"No files found matching the pattern: {pattern}")
        return
    
    for file_path in file_paths:
        nii_gz_file_path = file_path + '.gz'
        if os.path.exists(nii_gz_file_path):
            print(f"Skipped {file_path}, already converted.")
        else:
            convert_nii_to_nii_gz(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_nii_gz.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    main(directory)
