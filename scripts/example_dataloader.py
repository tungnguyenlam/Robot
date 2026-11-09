"""Iterate the unified torch DataLoader and print one batch, to check the data stack end to end.

    python scripts/example_dataloader.py configs/experiment/example_dataloader.yaml [key=value ...]
"""

import sys
import time

import torch

from egograph.config import load_config
from egograph.data.torch_dataset import EgoClipDataset


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/experiment/example_dataloader.yaml",
                      sys.argv[2:])
    ds = EgoClipDataset.from_config(cfg["data"])
    print(f"{len(ds)} clips from {[s['dataset'] for s in cfg['data']['sources']]}")
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    t = time.time()
    batch = next(iter(dl))
    print(f"first batch in {time.time() - t:.1f}s")
    for k, v in batch.items():
        desc = f"{tuple(v.shape)} {v.dtype}" if torch.is_tensor(v) else v
        print(f"  {k:16s} {desc}")


if __name__ == "__main__":
    main()
