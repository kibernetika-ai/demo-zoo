import logging
import cv2
import numpy as np
import pandas as pd
import io
from sklearn.externals import joblib
import os
import lightgbm as lgb

LOG = logging.getLogger(__name__)

PARAMS = {
    'deps': 15,
    'model': './model',
}
model_path = None
xsc = None
ysc = None
gbm = None


def init_hook(**kwargs):
    global PARAMS, xsc, ysc, gbm, model_path
    PARAMS.update(kwargs)
    PARAMS['deps'] = int(PARAMS['deps'])
    model_path = PARAMS['model']
    xsc = joblib.load(os.path.join(model_path, 'xscaler.pkl'))
    ysc = joblib.load(os.path.join(model_path, 'yscaler.pkl'))
    gbm = lgb.Booster(model_file='model.data')
    LOG.info('init: {}'.format(kwargs))


def process(inputs, ctx, **kwargs):
    doc = inputs['doc'][0]
    operational_settings = ['o_{}'.format(i + 1) for i in range(3)]
    sensor_columns = ['m_{}'.format(i + 1) for i in range(24)]
    cols = ['no', 'cycle'] + operational_settings + sensor_columns
    metrics = cols[2:-3]
    data = pd.read_csv(io.BytesIO(doc), sep=' ', index_col=False, header=None, names=cols)
    data = data.drop(cols[-3:], axis=1)
    data = data.fillna(0)
    results = []
    for k, v in data.groupby(['no']):
        x = v.loc[v.shape[0] - PARAMS['deps'] - 1:, metrics].values
        x = np.expand_dims(np.reshape(x, (-1)), axis=0)
        x = xsc.transform(x)
        y = gbm.predict(x)
        y = np.reshape(ysc.inverse_transform(np.reshape(y, (-1, 1))), -1)
        results.append(int(y[0]))

    return {'results': np.array(results, np.int32)}
