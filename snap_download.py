from huggingface_hub import snapshot_download


snapshot_download(repo_id="facebook/bart-large", local_dir = "cache/bart-large", allow_patterns=["pytorch_model.bin", "config.json", "tokenizer.*"], resume_download=True)
    

# snapshot_download(repo_id="google/mt5-small",local_dir = "cache/mt5-small",ignore_patterns=["tf_model.h5","flax_model.msgpack"],resume_download=True)

# snapshot_download(repo_id="google/mt5-base",local_dir = "cache/mt5-base",ignore_patterns=["tf_model.h5","flax_model.msgpack"],resume_download=True)

# snapshot_download(repo_id="google/umt5-small",local_dir = "cache/umt5-small",ignore_patterns=["tf_model.h5","flax_model.msgpack"],resume_download=True)

# snapshot_download(repo_id="google/umt5-base",local_dir = "cache/umt5-base",ignore_patterns=["tf_model.h5","flax_model.msgpack"],resume_download=True)

# snapshot_download(repo_id="fnlp/bart-base-chinese",local_dir = "cache/bart-base-chinese",ignore_patterns=["tf_model.h5","flax_model.msgpack","model.safetensors"],resume_download=True)

# snapshot_download(repo_id="fnlp/bart-large-chinese",local_dir = "cache/bart-large-chinese",ignore_patterns=["tf_model.h5","flax_model.msgpack","model.safetensors"],resume_download=True)