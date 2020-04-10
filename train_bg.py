import argparse
import random
import numpy as np
import tensorflow as tf
import logging
from models.fastbg.model import FastBGNet
from models.fastbg.model import data_fn
from models.fastbg.model import augumnted_data_fn
from models.fastbg.model import video_data_fn
from mlboardclient.api import client

import os
import json
import time
import yaml


def null_dataset():
    def _input_fn():
        return None

    return _input_fn


def catalog_ref(name, ctype, version):
    return '#/{}/catalog/{}/{}/versions/{}'. \
        format(os.environ.get('WORKSPACE_NAME'), ctype, name, version)


def serving_spec():
    return {'ports': [{'name': 'http',
                       'protocol': 'TCP',
                       'port': 9000,
                       'targetPort': 9000}],
            'resources': {'accelerators': {'gpu': 0},
                          'requests': {'cpu': '100m', 'memory': '128Mi'},
                          'limits': {'cpu': '10', 'memory': '4Gi'}},
            'images': {'cpu': 'kuberlab/serving:latest',
                       'gpu': 'kuberlab/serving:latest-gpu'},
            'command': 'kibernetika-serving --driver model --model-path=$MODEL_DIR  --hooks hook.py',
            'workDir': '$SRC_DIR',
            'default_volume_mapping': False,
            'disabled': False,
            'skipPrefix': False,
            'type': 'model',
            "spec": {
                'params': [
                    {
                        'name': 'inputs',
                        'type': 'image',
                        'value': ''
                    }
                ],
                'response': [
                    {
                        'name': 'output',
                        'type': 'bytes',
                        'shape': [
                            1,
                            -1
                        ]
                    }
                ],
                'responseTemplate': '{"output": "base64_encoded_image"}',
                'options': {
                    'noCache': True
                },
                'rawInput': True,
                'template': 'image'
            },
            'sources': [
                {
                    'gitRepo': {
                        'repository': 'https://github.com/kibernetika-ai/demo-zoo'
                    },
                    'name': 'src',
                    'subPath': 'demo-zoo/models/fastbg'
                },
            ]}


def export(checkpoint_dir, params):
    m = client.Client()
    base_id = '0'
    if os.environ.get('BASE_TASK_BUILD_ID', '') != '':
        app = m.apps.get()
        base_id = os.environ['BASE_TASK_BUILD_ID']
        task_name = os.environ['BASE_TASK_NAME']
        task = app.get_task(task_name, base_id)
        checkpoint_dir = task.exec_info['checkpoint_path']
        params['num_chans'] = task.exec_info['num-chans']
        params['num_pools'] = task.exec_info['num-pools']
        params['resolution'] = task.exec_info['resolution']
        params['checkpoint'] = checkpoint_dir

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
    )
    params['batch_size'] = 1
    features_def = [int(i) for i in task.exec_info.get('features','3').split(',')]
    logging.info('Features Def: {}'.format(features_def))
    params['features'] = features_def
    feature_placeholders = {
        'image': tf.placeholder(tf.float32, [1, None, None, sum(features_def)], name='image'),
    }
    receiver = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholders)
    net = FastBGNet(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
    )
    models = os.path.join(checkpoint_dir, 'models')
    build_id = os.environ['BUILD_ID']
    export_dir = os.path.join(models, build_id)
    os.makedirs(export_dir, exist_ok=True)
    export_path = net.export_savedmodel(
        export_dir,
        receiver,
    )
    export_path = export_path.decode("utf-8")
    base = os.path.basename(export_path)
    driver_data = {'driver': 'tensorflow', 'path': base}
    with open(os.path.join(export_dir, '_model_config.yaml'), 'w') as f:
        yaml.dump(driver_data, f)
    params['num_chans'] = task.exec_info['num-chans']
    params['num_pools'] = task.exec_info['num-pools']
    params['resolution'] = task.exec_info['resolution']
    version = f'1.{base_id}.{build_id}'
    model_name = 'person-mask'
    m.model_upload(model_name, version, export_dir, spec=serving_spec())
    client.update_task_info({'model_path': export_path, 'num-chans': params['num_chans'],
                             'features':','.join([str(i) for i in features_def]),
                             'num-pools': params['num_pools'], 'resolution': params['resolution'],
                             'model_reference': catalog_ref(model_name, 'mlmodel', version)})


def train(mode, checkpoint_dir, params):
    logging.info("start build  model")

    save_summary_steps = params['save_summary_steps']
    save_checkpoints_secs = params['save_checkpoints_secs'] if params['save_checkpoints_steps'] is None else None
    save_checkpoints_steps = params['save_checkpoints_steps']

    conf = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=params['keep_checkpoint_max'],
        log_step_count_steps=params['log_step_count_steps'],
    )
    if params['coco'] is None:
        logging.info('Use generic')
        epoch_len, fn = data_fn(params, mode == 'train')
    else:
        logging.info('Use Coco: {}'.format(params['features']))
        if len(params['features'])<2:
            epoch_len, fn = augumnted_data_fn(params, mode == 'train')
        else:
            epoch_len, fn = video_data_fn(params, mode == 'train')
    logging.info('Samples count: {}'.format(epoch_len))
    params['epoch_len'] = epoch_len
    net = FastBGNet(
        params=params,
        model_dir=checkpoint_dir,
        config=conf,
        warm_start_from=params['warm_start_from']

    )
    logging.info("Start %s mode type %s", mode, conf.task_type)
    if mode == 'train' and conf.task_type != 'ps':
        if conf.master != '':
            train_fn = fn
            train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
            eval_fn = null_dataset()
            eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=1, start_delay_secs=10, throttle_secs=10)
            tf.estimator.train_and_evaluate(net, train_spec, eval_spec)
        else:
            net.train(input_fn=fn)
    else:
        train_fn = null_dataset()
        train_spec = tf.estimator.TrainSpec(input_fn=train_fn)
        eval_fn = fn
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn, steps=1, start_delay_secs=10, throttle_secs=10)
        tf.estimator.train_and_evaluate(net, train_spec, eval_spec)


def main(args):
    params = {
        'num_pools': args.num_pools,
        'num_epochs': args.num_epochs,
        'drop_prob': args.drop_prob,
        'num_chans': args.num_chans,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'weight_decay': args.weight_decay,
        'checkpoint': str(args.checkpoint_dir),
        'save_summary_steps': args.save_summary_steps,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'keep_checkpoint_max': args.keep_checkpoint_max,
        'log_step_count_steps': args.log_step_count_steps,
        'warm_start_from': args.warm_start_from,
        'use_seed': False,
        'resolution': args.resolution,
        'data_set': args.data_set,
        'optimizer': args.optimizer,
        'loss': args.loss,
        'coco': args.coco,
        'features': [int(i) for i in args.features.split(',')]
    }
    if args.export:
        export(args.checkpoint_dir, params)
        return
    if not tf.gfile.Exists(args.checkpoint_dir):
        tf.gfile.MakeDirs(args.checkpoint_dir)
    if args.worker:
        client.update_task_info({
            'num-pools': args.num_pools,
            'drop-prob': args.drop_prob,
            'num-chans': args.num_chans,
            'batch-size': args.batch_size,
            'lr.lr': args.lr,
            'lr.lr-step-size': args.lr_step_size,
            'lr.lr-gamma': args.lr_gamma,
            'weight-decay': args.weight_decay,
            'checkpoint_path': str(args.checkpoint_dir),
            'resolution': args.resolution,
            'features': args.features,
        })
        train('train', args.checkpoint_dir, params)
    else:
        cluster = {'chief': ['fake_worker1:2222'],
                   'ps': ['fake_ps:2222'],
                   'worker': ['fake_worker2:2222']}
        os.environ['TF_CONFIG'] = json.dumps(
            {
                'cluster': cluster,
                'task': {'type': 'evaluator', 'index': 0}
            })
        train('eval', args.checkpoint_dir, params)


def create_arg_parser():
    conf_parser = argparse.ArgumentParser(
        add_help=False
    )
    conf_parser.add_argument(
        '--checkpoint_dir',
        default=os.environ.get('TRAINING_DIR', 'training') + '/' + os.environ.get('BUILD_ID', '1'),
        help='Directory to save checkpoints and logs')
    args, remaining_argv = conf_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_dir = args.checkpoint_dir
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--resolution', default=320, type=int, help='Resolution of images')

    parser.add_argument('--data_set', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--coco', type=str, default=None,
                        help='Coco path')
    parser.add_argument('--features', type=str, default='3',
                        help='features defention')
    parser.add_argument(
        '--save_summary_steps',
        type=int,
        default=10,
        help="Log summary every 'save_summary_steps' steps",
    )
    parser.add_argument(
        '--save_checkpoints_secs',
        type=int,
        default=600,
        help="Save checkpoints every 'save_checkpoints_secs' secs.",
    )
    parser.add_argument(
        '--save_checkpoints_steps',
        type=int,
        default=100,
        help="Save checkpoints every 'save_checkpoints_steps' steps",
    )
    parser.add_argument(
        '--keep_checkpoint_max',
        type=int,
        default=2,
        help='The maximum number of recent checkpoint files to keep.',
    )
    parser.add_argument(
        '--log_step_count_steps',
        type=int,
        default=1,
        help='The frequency, in number of global steps, that the global step/sec and the loss will be logged during training.',
    )
    parser.add_argument(
        '--warm_start_from',
        type=str,
        default=None,
        help='Warm start from',
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='adiff',
        help='Loss function',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='RMSPropOptimizer',
        help='Optimizer',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(worker=False)
    group.set_defaults(evaluator=False)
    group.set_defaults(test=False)
    group.add_argument('--worker', dest='worker', action='store_true',
                       help='Start in Worker(training) mode.')
    group.add_argument('--evaluator', dest='evaluator', action='store_true',
                       help='Start in evaluation mode')
    group.add_argument('--test', dest='test', action='store_true',
                       help='Test mode')
    group.add_argument('--export', dest='export', action='store_true',
                       help='Export model')
    return parser


if __name__ == '__main__':
    logging.getLogger().setLevel('INFO')
    args = create_arg_parser().parse_args()
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    main(args)
