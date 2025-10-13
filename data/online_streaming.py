import os
import json
from streaming.base.format import reader_from_json
from tqdm import tqdm
import argparse
import fsspec
import tempfile
from typing import Optional

class HFStreamingReader:
    def __init__(self, hf_path: str, split: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Initialize reader for streaming dataset from HuggingFace
        
        Args:
            hf_path: HuggingFace path (e.g., "hf://datasets/org/dataset/path/to/data")
            split: Optional split name (subdirectory)
            cache_dir: Optional local cache directory. If None, uses temp directory
        """
        self.hf_path = hf_path.replace("hf://", "")
        self.split = split
        
        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="hf_streaming_")
            print(f"Using temporary cache: {self.cache_dir}")
        else:
            self.cache_dir = cache_dir
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Setup fsspec filesystem
        self.fs = fsspec.filesystem("hf", token=True)  # Uses HF_TOKEN env var if available
        
        # Load shards
        print(f"Loading shards from {hf_path}" + (f" split {split}" if split else ""))
        self.shards = self._load_shards()
        print(f"Loaded {len(self.shards)} shards")
        
        # Calculate total samples
        self.total_samples = sum(shard.samples for shard in self.shards)
        print(f"Total samples: {self.total_samples:,}")
    
    def _download_file(self, remote_path: str, local_path: str):
        """Download a file from HuggingFace if not already cached"""
        if os.path.exists(local_path):
            return
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            with self.fs.open(remote_path, 'rb') as remote_file:
                with open(local_path, 'wb') as local_file:
                    local_file.write(remote_file.read())
        except Exception as e:
            print(f"Error downloading {remote_path}: {e}")
            raise
    
    def _load_shards(self):
        """Load all shards from index.json"""
        # Construct paths
        if self.split:
            remote_dir = f"{self.hf_path}/{self.split}"
            local_dir = os.path.join(self.cache_dir, self.split)
        else:
            remote_dir = self.hf_path
            local_dir = self.cache_dir
        
        os.makedirs(local_dir, exist_ok=True)
        
        # Download index.json
        remote_index = f"{remote_dir}/index.json"
        local_index = os.path.join(local_dir, "index.json")
        
        print(f"Downloading index from {remote_index}")
        self._download_file(remote_index, local_index)
        
        # Read index
        with open(local_index) as f:
            obj = json.load(f)
        
        # Download all shard files
        shards = []
        for info in tqdm(obj["shards"], desc="Downloading shards"):
            # Download shard data files
            for file_info in info.get("raw_data", {}).get("files", []):
                basename = file_info if isinstance(file_info, str) else file_info.get("basename", "")
                if basename:
                    remote_file = f"{remote_dir}/{basename}"
                    local_file = os.path.join(local_dir, basename)
                    self._download_file(remote_file, local_file)
            
            # Also download the primary raw data file
            if "raw_data" in info and "basename" in info["raw_data"]:
                basename = info["raw_data"]["basename"]
                remote_file = f"{remote_dir}/{basename}"
                local_file = os.path.join(local_dir, basename)
                self._download_file(remote_file, local_file)
            
            # Create shard reader
            shard = reader_from_json(self.cache_dir, self.split, info)
            shards.append(shard)
        
        return shards
    
    def __iter__(self):
        """Iterate through all samples in all shards"""
        for shard in self.shards:
            for sample_idx in range(shard.samples):
                yield shard[sample_idx]
    
    def __len__(self):
        return self.total_samples
    
    def get_sample(self, global_idx: int):
        """Get a specific sample by global index"""
        if global_idx >= self.total_samples:
            raise IndexError(f"Index {global_idx} out of range (total: {self.total_samples})")
        
        cumulative = 0
        for shard in self.shards:
            if cumulative + shard.samples > global_idx:
                local_idx = global_idx - cumulative
                return shard[local_idx]
            cumulative += shard.samples
        
        raise IndexError(f"Sample {global_idx} not found")


if __name__ == "__main__":
    # Create streaming dataset with whatever path you want
    dataset = HFStreamingReader(
        hf_path="hf://datasets/jhu-clsp/mmBERT-decay-data/train/books-gutenberg-dup-sampled-decay/shard_00000-tokenized-chunked-8192-512-32-backfill-nodups",
        split=None
    )

    # Let's see what's in it
    for sample in dataset:
        print(sample.keys())
        text = sample.get('input_ids', '')
        id = sample.get('id', '')
        print(f"Text: {text}")
        print(f"ID: {id}")
        break
