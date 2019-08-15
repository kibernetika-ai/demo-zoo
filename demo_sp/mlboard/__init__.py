import os

import logging

try:
    from mlboardclient.api import client
except ImportError:
    client = None


mlboard = None
mlboard_logging = False

if client:
    mlboard = client.Client()
    mlboard_logging = True
    try:
        mlboard.apps.get()
    except Exception:
        mlboard_logging = False
        logging.warning('Do not use mlboard parameters logging.')
    else:
        logging.info('Using mlboard parameters logging.')


def update_task_info(data):
    if mlboard and mlboard_logging:
        mlboard.update_task_info(data)


def catalog_ref(name, ctype, version):
    return '#/{}/catalog/{}/{}/versions/{}'. \
        format(os.environ.get('WORKSPACE_NAME'), ctype, name, version)



def dataset_upload(model_name, version, path,auto_create=True):
    workspace = os.environ.get('WORKSPACE_NAME')
    if not workspace:
        raise RuntimeError('workspace required')

    mlboard.datasets.push(
        workspace,
        model_name,
        version,
        path,
        type='dataset',
        create=auto_create,
    )