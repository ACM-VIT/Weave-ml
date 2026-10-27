# Ranker training storage

`Ranking_Training_module/train_ranker.py --data-path` accepts the existing NPZ
archive or a directory containing one NumPy `.npy` file for each `DATA_KEYS`
entry: `user_embs`, `repo_embs`, `dense_features`, `y_ctr`, `y_save`, `y_gh`,
`y_dwell`, and `y_follow`. Arrays must have aligned chronological row order.
Embeddings have 384 columns; dense features follow the ten-column `FEATURE_ORDER`.
Each label array has shape `(rows,)` or `(rows, 1)`.

NPZ remains compatible and loads its source arrays into memory. NPZ members are
not memory mapped. For larger datasets, the directory format opens arrays as
read-only memory maps. Convert an existing archive without loading its complete
members into Python memory:

```python
from pathlib import Path
import shutil
import zipfile
from Ranking_Training_module.train_ranker import DATA_KEYS

output = Path("training_arrays")
output.mkdir(exist_ok=False)
with zipfile.ZipFile("training_data.npz") as archive:
    for key in DATA_KEYS:
        with archive.open(f"{key}.npy") as source:
            with (output / f"{key}.npy").open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
```

```sh
python Ranking_Training_module/train_ranker.py \
  --data-path training_arrays --batch-size 4096 --output-dir /tmp/ranker-output
```

Both paths assemble float32 model inputs and labels per batch. The trainer keeps
no full concatenated input matrix or full standardized dense matrix. Scaler
fitting uses two bounded passes over the first 80% of rows; validation remains
the final 20%, with the existing training shuffle and BatchNorm singleton policy.
Imputation, float32 serving transforms, clipping and feature order are unchanged.
Chunked float64 accumulation may change low-order scaler bits compared with a
single whole-array reduction; model input comparisons should use float32
precision tolerances. The preprocessing version and qualification rules remain
unchanged.

Logical dataset SHA-256 retains the exact ordered key/dtype/shape/C-order byte
identity across NPZ compression, array layout and directory storage. Hashing uses
bounded chunks. The trainer rechecks loaded array contents, original storage
bytes and training source identity before writing artifacts. Keep all input
files immutable during a run, including mapped files; replacements or in-place
edits fail export. Storage hashing adds sequential I/O at startup and export.

Memory-map resident pages and the shuffle sampler's row indices still contribute
to process RSS. Batch memory scales with batch size; OS page caching, model and
optimizer allocations remain outside that bound. Mapped random access and
repeated per-batch transforms can trade throughput for lower feature-buffer
memory. Choose a batch size for the available host memory; this change does not
qualify or replace the production model.
