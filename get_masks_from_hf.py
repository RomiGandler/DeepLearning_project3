from huggingface_hub import snapshot_download
repo_id = "roni-hershko/chess_data"
repo_type="dataset"
allow_patterns=["train/masks/*", "val/masks/*", "test/masks/*"]
local_dir = "/home/avinoamd/roni/masks"
snapshot_download(repo_id=repo_id, repo_type=repo_type, allow_patterns=allow_patterns, local_dir=local_dir)