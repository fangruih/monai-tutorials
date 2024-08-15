import nibabel as nib
import os
import sys
import glob
import math

def convert_nii_to_nii_gz(file_path):
    try:
        img = nib.load(file_path)
        nii_gz_file_path = file_path + '.gz'
        nib.save(img, nii_gz_file_path)
        print(f"Converted {file_path} to {nii_gz_file_path}")
    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")

def main(directory, chunk_index, total_chunks):
    pattern = os.path.join(directory, 'sub-NDARINV*/ses-baselineYear1Arm1/anat/*baselineYear1Arm1_run-01_T1w.nii')
    file_paths = glob.glob(pattern)
    
    if not file_paths:
        print(f"No files found matching the pattern: {pattern}")
        return

    # Divide the files into chunks
    total_files = len(file_paths)
    chunk_size = math.ceil(total_files / total_chunks)
    start_index = chunk_index * chunk_size
    end_index = min(start_index + chunk_size, total_files)
    
    # Get the chunk of files to process
    file_paths_chunk = file_paths[start_index:end_index]
    
    for file_path in file_paths_chunk:
        nii_gz_file_path = file_path + '.gz'
        if os.path.exists(nii_gz_file_path):
            print(f"Skipped {file_path}, already converted.")
        else:
            convert_nii_to_nii_gz(file_path)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_to_nii_gz.py <directory> <chunk_index> <total_chunks>")
        sys.exit(1)
    
    directory = sys.argv[1]
    chunk_index = int(sys.argv[2])
    total_chunks = int(sys.argv[3])
    main(directory, chunk_index, total_chunks)
