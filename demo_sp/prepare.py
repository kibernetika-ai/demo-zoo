import argparse
import os
import logging
import pandas as pd
import glob
import numpy as np
from mlboard import mlboard, update_task_info, catalog_ref


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
        '--deps',
        type=int,
        default=15,
        help='Window to analyze',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='sp-demo',
        help='DataSet Name',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='1.0.'+os.environ.get('BUILD_ID', '1'),
        help='DataSet version',
    )

    args = parser.parse_args()
    return args


def convert(file,cols,metrics,deps=15):
    data = pd.read_csv(file, sep=' ',index_col=False, header=None,names=cols)
    data = data.drop(cols[-3:], axis=1)
    data = data.fillna(0)
    cycles={}
    for k,v in data.groupby(['no']):
        cycle = v.iloc[-1,1]
        cycles[k]=cycle
    x = []
    y = []
    for k,v in data.groupby(['no']):
        v = v.reset_index(drop=True)
        for i in range(0,v.shape[0]-deps,int(deps/2)):
            row = v.loc[i:i+deps,metrics].values
            row = np.reshape(row,(-1))
            cycle = cycles[k]-v.loc[i+deps-1,'cycle']
            x.append(row)
            y.append(cycle)

        row = v.loc[v.shape[0]-deps-1:,metrics].values
        row = np.reshape(row,(-1))
        cycle = cycles[k]-v.loc[v.shape[0]-1,'cycle']
        x.append(row)
        y.append(cycle)

    x = np.stack(x).astype(np.float32)
    y = np.array(y,np.float32)
    return x,y

def main():
    args = parse_args()
    operational_settings = ['o_{}'.format(i + 1) for i in range (3)]
    sensor_columns = ['m_{}'.format(i + 1) for i in range(24)]
    cols = ['no', 'cycle'] + operational_settings + sensor_columns
    metrics = cols[2:-3]
    os.mkdir(args.dst)
    os.mkdir(os.path.join(args.dst,'train'))
    os.mkdir(os.path.join(args.dst,'eval'))
    for f in glob.glob(os.path.join(args.data,'train_FD*.txt')):
        name = os.path.basename(f)
        name = name.rstrip('.txt')
        x,y = convert(f,cols,metrics,args.deps)
        np.savez(os.path.join(args.dst,'train',name),x=x,y=y)
    for f in glob.glob(os.path.join(args.data,'test_FD*.txt')):
        name = os.path.basename(f)
        name = name.rstrip('.txt')
        x,_ = convert(f,cols,metrics,args.deps)
        data = pd.read_csv(f.replace('test_','RUL_'), sep=' ',index_col=False, header=None)
        y = data[0].values.astype(np.float32)
        np.savez(os.path.join(args.dst,'eval',name),x=x,y=y)
    mlboard.dataset_upload(args.dataset,args.version, args.dst)
    update_task_info({'model_reference': catalog_ref(args.dataset, 'dataset', args.version)})

if __name__ == '__main__':
    main()
