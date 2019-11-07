import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import io

LOG = logging.getLogger(__name__)

CLASSES = ["Cat. N","Cat. S","Cat. V","Cat. F","Cat. Q"]

def init_hook(**kwargs):
    LOG.info('init: {}'.format(kwargs))

def process(inputs, ctx, **kwargs):
    data = inputs.get('data')[0]
    data = io.BytesIO(data)
    df = pd.read_csv(data, header=None)
    x = df.values
    if x.shape[1]>187:
        x = x[:, :187].astype(np.float32)
    x = np.expand_dims(x[0:1,:], 2)
    result = ctx.drivers[0].predict({'input_1':x})
    logging.info(result)
    logging.info(result.shape)
    result = int(result['softmax'].argmax())
    logging.info(result)
    plt.figure(figsize=(20, 12))
    v = np.arange(0, 187) * 8 / 1000
    plt.plot(v, x[0,:,0])
    plt.title("Result: {}".format(CLASSES[result]), fontsize=20)
    plt.ylabel("Amplitude", fontsize=15)
    plt.xlabel("Time (ms)", fontsize=15)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = np.array(bytearray(buf.read())).tostring()
    return {'output': image}