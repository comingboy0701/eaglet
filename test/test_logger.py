from script import get_logger

logger = get_logger(name="test_tmp")
# 第五步、日志输出
logger.debug('this is a logger debug message')
logger.info('this is a logger info message')
logger.warning('this is a logger warning message')
logger.error('this is a logger error message')
logger.critical('this is a logger critical message')