import logging
import datetime
import os
import shutil
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_PATH = Path(__file__).parent

DEBUG = True


class LogConfig(object):

    def __init__(self, ):
        self.name = "eaglet"
        self.log_dir = os.path.join(ROOT_PATH, "output", "log")
        os.makedirs(self.log_dir, exist_ok=True)
        self.formatter = '[%(asctime)s] [%(filename)s:%(lineno)d]  %(levelname)s - %(message)s'
        self.type_ = "RotatingFileHandler"  # 按文件大小自动分片
        # self.type_= "TimedRotatingFileHandler",  # 按日期自动分片
        self.maxByte = 300 * 1024 * 1024
        self.backupCount = 5


config_log = LogConfig()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def rotator(source, dest):
    dest = dest.split(".log")
    dest = "".join(dest)
    dest = f"{dest}.log"
    shutil.move(source, dest)


def namer(name):
    name = name.split(".log")
    name = "".join(name)
    name = f"{name}.log"
    return name


def get_logger(log_dir="default", name: str = "default") -> logging.Logger:
    """
    生成logger, 防止日志重复输出
    :params name: 值为default的时候会用LOG_CONFIG中的值作为文件名, 若希望使用独立的文件, 传入自定义的文件名
    """
    logger = logging.getLogger(name)
    # 处理log_name
    logger.propagate = False

    if log_dir == "default":
        log_dir = config_log.log_dir
    if name == "default":
        name = config_log.name
    log_name = "{}_{}.log".format(name, timestamp)

    has_stream = False
    # 处理logger的handlers, 防止重复添加
    for h in logger.handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            if h.baseFilename == os.path.abspath(os.path.join(log_dir, log_name)):
                return logger
        else:
            has_stream = True
            continue
    formatter = logging.Formatter(config_log.formatter)
    # 选择handler的形式并添加handler
    if config_log.type_ == "RotatingFileHandler":
        filehandler = RotatingFileHandler(
            os.path.join(log_dir, log_name),
            encoding="UTF-8",
            maxBytes=config_log.maxByte,
            backupCount=config_log.backupCount
        )
    filehandler.setFormatter(formatter)
    filehandler.rotator = rotator
    filehandler.namer = namer
    logger.addHandler(filehandler)

    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # 选择是否要在控制台输出
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger
