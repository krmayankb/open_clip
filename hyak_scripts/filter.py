import dask
import dask.dataframe as dd
import numpy as np 
import pyarrow.parquet as pq
import pandas as pd

metadata_path = '/mmfs1/gscratch/krishna/mayank/dfn/dfn-medium/metadata'

indices = np.load('/mmfs1/gscratch/krishna/mayank/dfn/indicies/datacomp_medium_dfn_20m_inds.npy')
all_uids = set([f'{uid[0]:016x}{uid[1]:016x}' for uid in indices])

fp = pq.read_table(
    source=metadata_path,
    use_threads=True,
    filters=[('uid', 'in', all_uids)]
)
schema = pq.read_schema("/mmfs1/gscratch/krishna/mayank/dfn/dfn-medium/metadata/0a4a1e10352ec4366858927c33873cdc.parquet")
df = fp.to_pandas()

# save df to disk
dd.from_pandas(df, chunksize=500000).to_parquet(
    '/mmfs1/gscratch/krishna/mayank/dfn/dfn-medium/metadata_filtered', 
    schema=schema
    )

