<p align="center">
  <img src="assets/mas22.jpg" alt="MAST Logo" width="300"/>
</p>

Our blogpost: https://sites.google.com/berkeley.edu/mast/

This repository contains the code and the data for the paper "Why Do Multi-Agent Systems Fail?" [https://arxiv.org/pdf/2503.13657](https://arxiv.org/pdf/2503.13657v2)


In this paper, we present the first comprehensive study of MAS challenges: MAST (Multi-Agent Systems Failure Taxonomy).

![A Taxonomy of MAS Failure Modes](assets/taxonomy_v11_cropped-1.png)
![Study Workflow](assets/arxiv_figure_v2_cropped-1.png)

## News! We just released our dataset with over 1K annotated MAS traces [https://arxiv.org/pdf/2503.13657v2](https://huggingface.co/datasets/mcemri/MAD)

For LLM-as-a-Judge annotated traces:
```
from huggingface_hub import hf_hub_download
import pandas as pd
import json

REPO_ID = "mcemri/MAD"
FILENAME = "MAD_full_dataset.json"

file_path =  hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
with open(file_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records (full dataset).")
```

For human annotated traces:
```
FILENAME = "MAD_human_labelled_dataset.json"

file_path =  hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
with open(file_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} records (human labelled).")
```

If you find this work useful, please cite it as follows:

```bibtex
@article{cemri2025multi,
  title={Why Do Multi-Agent LLM Systems Fail?},
  author={Cemri, Mert and Pan, Melissa Z and Yang, Shuyi and Agrawal, Lakshya A and Chopra, Bhavya and Tiwari, Rishabh and Keutzer, Kurt and Parameswaran, Aditya and Klein, Dan and Ramchandran, Kannan and others},
  journal={arXiv preprint arXiv:2503.13657},
  year={2025}
}
```
