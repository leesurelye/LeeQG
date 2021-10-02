import logging
import logging.config
import logging.handlers
import os
import copy
from LeeQG import CONSTANT
from enum import Enum, unique


@unique
class DirMode(Enum):
    CONFIG = 0
    PACKAGE = 1


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "log_dir": CONSTANT.LOG_ROOT +'/info.log',
    "formatters": {
        "simple": {
            'format': '%(asctime)s | [%(name)s] | [%(levelname)s]- %(message)s'
        },
        'standard': {
            'format': '%(asctime)s | [%(name)s] | [%(levelname)s]- %(message)s'
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "debug": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "debug.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        },
        "info": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "simple",
            "filename": "info.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        },
        "warn": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "WARN",
            "formatter": "simple",
            "filename": "warn.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        },
        "error": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "ERROR",
            "formatter": "simple",
            "filename": "error.log",
            "when": "midnight",
            "interval": 1,
            "backupCount": 6,
            "encoding": 'utf-8'
        }
    },

    "root": {
        'handlers': ['debug', "info", "warn", "error", "console"],
        'level': "DEBUG",
        'propagate': False
    }
}


def get_filter(level):
    if level == logging.DEBUG:
        return lambda record: record.levelno < logging.INFO
    elif level == logging.INFO:
        return lambda record: record.levelno < logging.WARN
    elif level == logging.WARN:
        return lambda record: record.levelno < logging.ERROR
    else:
        return lambda record: record.levelno <= logging.FATAL


def adjust_config(logging_config, dir_mode=DirMode.CONFIG):
    # 使用配置目录
    if dir_mode == DirMode.CONFIG:
        dirName = logging_config['log_dir']
    # 使用logger.py同级目录
    else:
        currentdir = os.path.dirname(__file__).replace('\\', '/')
        dirName = currentdir + '/logs/'

    handlers = logging_config.get('handlers')
    for handler_name, handler_config in handlers.items():
        filename = handler_config.get('filename', None)
        if filename is None:
            continue
        if dirName is not None:
            if not os.path.exists(dirName):
                try:
                    os.makedirs(dirName)
                except Exception as e:
                    print(e)
            handler_config['filename'] = dirName + filename
    return logging_config


def get_logger(name=None):
    #  拷贝配置字典
    logging_config = copy.deepcopy(LOGGING_CONFIG)

    # 调整配置内容
    adjust_config(logging_config, DirMode.PACKAGE)

    # 使用调整后配置生成logger
    logging.config.dictConfig(logging_config)
    res_logger = logging.getLogger(name)

    for handler in res_logger.root.handlers:
        if handler.name == 'console':
            continue
        log_filter = logging.Filter()
        log_filter.filter = get_filter(handler.level)
        handler.addFilter(log_filter)
    return res_logger

