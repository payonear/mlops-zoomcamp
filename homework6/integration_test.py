import pandas as pd
import batch

data = [
    (None, None, batch.dt(1, 2), batch.dt(1, 10)),
    (1, 1, batch.dt(1, 2), batch.dt(1, 10)),
    (1, 1, batch.dt(1, 2, 0), batch.dt(1, 2, 50)),
    (1, 1, batch.dt(1, 2, 0), batch.dt(2, 2, 1)),        
]

columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
df = pd.DataFrame(data, columns=columns)
S3_ENDPOINT_URL = 'http://localhost:4566'
options = {
    'client_kwargs': {
    'endpoint_url': S3_ENDPOINT_URL
    }
}

input_file = batch.get_input_path(2021, 1)
df.to_parquet(
    input_file,
    engine='pyarrow',
    compression=None,
    index=False,
    storage_options=options
)

batch.main(2021,1)