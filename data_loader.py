import re
import os
import json
import numpy as np
import pandas as pd
from transform import TimeSeriesTransform


cfg = json.load(open("config/config.json"))
ts_trans = TimeSeriesTransform(cfg)


def get_X_y(data_dir, filenames, load):
    X, y = [], []
    for filename in filenames:
        load_num = load[:-2]
        state = re.match(fr'(.*)_{load_num}', filename).group(1)
        df = pd.read_csv(os.path.join(data_dir, load, state, filename))
        tmp_cur = ts_trans.smoothing(ts_df=df, field='current')
        # tmp_pos = ts_transform.smoothing(ts_df=df, field='position_error')
        X.append(tmp_cur)
        y.append(state)
    return np.array(X), np.array(y)
