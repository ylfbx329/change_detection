import logging
import torch
from PIL import Image
import numpy as np


def get_logger(filename, verbosity=1, name=None):
    """
    设置文件流和标准输出流的输出格式和级别

    :param filename:
    :param verbosity:
    :param name:
    :return: 日志器
    """
    # 日志级别字典
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # 记录格式类
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    # 获取日志器
    logger = logging.getLogger(name)
    # 设置日志记录级别
    logger.setLevel(level_dict[verbosity])

    # 日志文件头
    fh = logging.FileHandler(filename, "w")
    # 设置文件头格式
    fh.setFormatter(formatter)
    # 添加文件头
    logger.addHandler(fh)

    # 输出流，默认标准输出
    sh = logging.StreamHandler()
    # 设置流格式
    sh.setFormatter(formatter)
    # 添加数据流头
    logger.addHandler(sh)

    return logger


def to8bits(img):
    result = np.ones([img.shape[0], img.shape[1]], dtype='int')
    result[img == 0] = 0
    result[img == 1] = 255
    return result


def save_pre_result(pre, flag, num, save_path):
    pre[pre >= 0.5] = 255
    pre[pre < 0.5] = 0
    outputs = torch.squeeze(pre).cpu().detach().numpy()
    outputs = Image.fromarray(np.uint8(outputs))
    outputs.save(save_path + '/%s_%d.png' % (flag, num))
