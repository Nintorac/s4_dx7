from itertools import chain
from pathlib import Path
import pandas as pd
import pyarrow as pa
import os
CART_SOURCE = os.environ['CART_SOURCE']

def model(dbt, session):
    carts = []
    sources = [
        Path(CART_SOURCE).glob('**/*.syx'),
        Path(CART_SOURCE).glob('**/*.SYX'),
    ]
    for path in chain(*sources):
        if path.is_dir(): continue
        carts.append({
            'path': path.relative_to(CART_SOURCE).as_posix(),
            'bytes': bytes(path.read_bytes())
        })
    carts = pd.DataFrame(carts)
    return carts