import requests

from sync import conf


def sync_progress(job_id: int, progress: int):
    '''
    用于同步计算任务的进度
    :param job_id: 通过start接口传递过来，运行任务的标识[job_id:11678]
    :param progress: 进度值，在0-100之间[progress:56]
    :return:
    '''
    if conf.prod:
        requests.post(conf.sync_server, {"id": job_id, "progress": progress, "type": "progress"})
    print("syncing progress")


def sync_finish(job_id, annotations):
    '''
    用于同步计算的结果
    :param job_id: 通过start接口传递过来，运行任务的标识[job_id:11678]
    :param annotations: 计算的图斑结果，详见图斑规则
    :return:
    '''
    if conf.prod:
        requests.post(conf.sync_server, {"id": job_id, "annotations": annotations})
    print("syncing finish")

def sync_error(job_id: int, error: str):
    '''
    用于同步错误原因
    :param job_id: 通过start接口传递过来，运行任务的标识[job_id:11678]
    :param error: 错误原因 [error: "file not found"]
    :return:
    '''
    if conf.prod:
        requests.post(conf.sync_server, {"id": job_id, "error": error, "type": "finish"})
    print("syncing error")