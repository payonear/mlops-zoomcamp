import batch
from cgi import test
import pandas as pd
from datetime import datetime
from pprint import pprint

def test_prepare_data():
    data = [
    (None, None, batch.dt(1, 2), batch.dt(1, 10)),
    (1, 1, batch.dt(1, 2), batch.dt(1, 10)),
    (1, 1, batch.dt(1, 2, 0), batch.dt(1, 2, 50)),
    (1, 1, batch.dt(1, 2, 0), batch.dt(2, 2, 1)),        
    ]

    columns = ['PUlocationID', 'DOlocationID', 'pickup_datetime', 'dropOff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PUlocationID', 'DOlocationID']
    actual = batch.prepare_data(df, categorical).to_dict()
    expected = {'DOlocationID': {0: '-1', 1: '1'},
                'PUlocationID': {0: '-1', 1: '1'},
                'dropOff_datetime': {0: datetime(2021, 1, 1, 1, 10),
                                    1: datetime(2021, 1, 1, 1, 10)
                                    },
                'duration': {0: 8.000000000000002, 1: 8.000000000000002},
                'pickup_datetime': {0: datetime(2021, 1, 1, 1, 2),
                                    1: datetime(2021, 1, 1, 1, 2)
                                    }
                }

    assert actual == expected