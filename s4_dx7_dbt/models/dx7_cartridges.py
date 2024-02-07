from pathlib import Path
import pandas as pd
import pyarrow as pa
import os
CART_SOURCE = os.environ['CART_SOURCE']

def model(dbt, session):
    carts = []
    for path in Path(CART_SOURCE).glob('**/*.syx'):
        carts.append({
            'path': path.relative_to(CART_SOURCE).as_posix(),
            'bytes': bytes(path.read_bytes())
        })
    carts = pd.DataFrame(carts)
    return carts