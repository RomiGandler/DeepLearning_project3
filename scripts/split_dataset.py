import os
import random
import shutil

def split_dataset(split_ratio=0.15):
    trainA_dir = "dataset/trainA"
    trainB_dir = "dataset/trainB"
    testA_dir = "dataset/testA"
    testB_dir = "dataset/testB"
    
    # Get list of all files in trainA (should match trainB)
    files = [f for f in os.listdir(trainA_dir) if f.endswith('.jpg')]
    
    # Calculate number of files to move
    num_test = int(len(files) * split_ratio)
    
    print(f"Total images: {len(files)}")
    print(f"Moving {num_test} images to test set...")
    
    # Randomly sample files
    test_files = random.sample(files, num_test)
    
    for filename in test_files:
        # Move from A
        shutil.move(os.path.join(trainA_dir, filename), os.path.join(testA_dir, filename))
        # Move from B
        shutil.move(os.path.join(trainB_dir, filename), os.path.join(testB_dir, filename))
        
    print("Split complete!")
    print(f"Train size: {len(os.listdir(trainA_dir))}")
    print(f"Test size: {len(os.listdir(testA_dir))}")

if __name__ == "__main__":
    split_dataset()



