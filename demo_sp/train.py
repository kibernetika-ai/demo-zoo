import argparse
import os
import logging
import lightgbm as lgb
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from mlboard import mlboard,update_task_info, catalog_ref
import matplotlib.pyplot as plt
import io
import base64


def parse_args():
    logging.getLogger().setLevel('INFO')
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dst',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Destination dir',
    )
    parser.add_argument(
        '--data',
        default=os.environ.get('DATA_DIR', './data'),
        help='Data dir',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sp-demo',
        help='Model Name',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='1.0.'+os.environ.get('BUILD_ID', '1'),
        help='Model version',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.05,
        help='Learning rate',
    )

    parser.add_argument(
        '--exp',
        type=str,
        default='FD001',
        help='Device set name',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=100,
        help='Iterations',
    )


    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    os.mkdir(args.dst)
    update_task_info({'iteration':args.iterations,'exp':args.exp,'test_split':0.25,'learning_rate':args.learning_rate})
    npzfile = np.load(os.path.join(args.data,'train','train_'+args.exp+'.npz'))
    x = npzfile['x']
    y = npzfile['y']
    x_train, x_eval, y_train, y_eval = train_test_split(x, y, test_size = 0.25, random_state = 0)
    xsc = StandardScaler()
    x_train = xsc.fit_transform(x_train)
    x_eval = xsc.transform(x_eval)
    ysc = StandardScaler()
    y_train = np.reshape(ysc.fit_transform(np.reshape(y_train,(-1,1))),-1)
    y_eval =  np.reshape(ysc.fit_transform(np.reshape(y_eval,(-1,1))),-1)
    d_train = lgb.Dataset(x_train, label=y_train)
    d_eval = lgb.Dataset(x_eval, label=y_eval, reference=d_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'mae'},
        'learning_rate': args.learning_rate,
        'verbose': 1}
    gbm = lgb.train(params,d_train,num_boost_round=args.iterations,valid_sets=d_eval,early_stopping_rounds=5)
    gbm.save_model(os.path.join(args.dst,'model.data'))
    joblib.dump(xsc, os.path.join(args.dst,'xscaler.pkl'))
    joblib.dump(ysc, os.path.join(args.dst,'yscaler.pkl'))
    npzfile = np.load(os.path.join(args.data,'eval','test_'+args.exp+'.npz'))
    x = npzfile['x']
    y = npzfile['y']
    x_eval = xsc.transform(x)
    y_pred=gbm.predict(x_eval)
    y_pred = np.reshape(ysc.inverse_transform(np.reshape(y_pred,(-1,1))),-1)
    mae = mean_absolute_error(y_pred,y)
    z = np.abs(y_pred-y)
    plt.title('Absolute Error Distribution')
    plt.hist(z,20)
    plt.ylabel('Count')
    plt.xlabel('Abs Error')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    rpt = '<html><img src="data:image/png;base64,{}"/></html>'.format(base64.b64encode(buf.getvalue()).decode())
    update_task_info({'#documents.report.html':rpt})
    logging.info('Final MAE: {}'.format(int(mae)))
    update_task_info({'mae':mae})
    version = args.version+'-'+args.exp
    mlboard.model_upload(args.model,version, args.dst)
    update_task_info({'model_reference': catalog_ref(args.model, 'mlmodel', version)})


if __name__ == '__main__':
    main()
