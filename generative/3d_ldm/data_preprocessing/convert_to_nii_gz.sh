#!/bin/bash

# Find all the .nii files and store them in a variable
files=$(find /simurgh/group/mri_data/abcd/stru/t1/st2_registered -type f -path '*/ses-baselineYear1Arm1/anat/*baselineYear1Arm1_run-01_T1w.nii')

# Count the total number of files
total=$(echo "$files" | wc -l)
count=0

# Loop through each file and compress it
for file in $files; do
  gzip "$file"
  count=$((count + 1))
  echo "Processed $count of $total files"
done
