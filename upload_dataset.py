new_branch = "roni-side-quest"
local_folder = "/home/avinoamd/roni/src/data/dataset"
repo_id = "roni-hershko/chess_data"
from huggingface_hub import HfApi

# 1. Initialize the API
api = HfApi()

# 2. Use the robust upload method
api.upload_large_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="dataset",
    revision=new_branch)