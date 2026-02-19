# Datasets

## APIs

```python
from datasets import load_dataset, Dataset

raw_dataset = load_dataset("json", data_files=json_file, split="train")
filtered_dataset = raw_dataset.select(range(i,i+20))

raw_dataset = Dataset.from_pandas(df)
df = raw_dataset.to_pandas()
comment_dataset = raw_dataset.remove_columns(columns_to_remove)
```