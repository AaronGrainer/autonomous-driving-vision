import logging
import logging.config
import os
import sys
from pathlib import Path

import pretty_errors  # NOQA: F401
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
LOGS_DIR = Path(BASE_DIR, "logs")
MODEL_DIR = Path(BASE_DIR, "model")
DATA_DIR = Path(BASE_DIR, "data")
MOVIELENS_RATING_DATA_DIR = Path(BASE_DIR, "data", "ml-20m", "ratings.csv")
MOVIELENS_MOVIE_DATA_DIR = Path(BASE_DIR, "data", "ml-20m", "movies.csv")

# Data
MOVIE_DATASET_DIR = Path(DATA_DIR, "movie")

# Create Dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Backend URL
BACKEND_HOST = os.getenv("BACKEND_HOST")
BACKEND_PORT = os.getenv("BACKEND_PORT")
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

# Recommender URL
RECOMMENDER_ENGINE_HOST = os.getenv("RECOMMENDER_ENGINE_HOST")
RECOMMENDER_ENGINE_PORT = os.getenv("RECOMMENDER_ENGINE_PORT")
RECOMMENDER_ENGINE_URL = f"http://{RECOMMENDER_ENGINE_HOST}:{RECOMMENDER_ENGINE_PORT}"

# Frontend
AUTO_AUTH = True
BACKEND_USERNAME = os.getenv("BACKEND_USERNAME")
BACKEND_PASSWORD = os.getenv("BACKEND_PASSWORD")

# Backend
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
if ACCESS_TOKEN_EXPIRE_MINUTES:
    ACCESS_TOKEN_EXPIRE_MINUTES = int(ACCESS_TOKEN_EXPIRE_MINUTES)

# Kafka
KAFKA_PRODUCER_HOST = os.getenv("KAFKA_PRODUCER_HOST")
KAFKA_CONSUMER_HOST = os.getenv("KAFKA_CONSUMER_HOST")
KAFKA_PORT = os.getenv("KAFKA_PORT")
KAFKA_PRODUCER_URL = f"{KAFKA_PRODUCER_HOST}:{KAFKA_PORT}"
KAFKA_CONSUMER_URL = f"{KAFKA_CONSUMER_HOST}:{KAFKA_PORT}"

# Recommender
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64

MASK = 1
PAD = 0
CAP = 0
MASK_PROBABILITY = 0.5
VAL_CONTEXT_SIZE = 5
HISTORY_SIZE = 120
DEFAULT_CONTEXT_SIZE = 120
CHANNELS = 128
DROPOUT = 0.4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
MODEL_URI = "gs://personal-mlflow-tracking/artifacts/1/{}/artifacts/model/"

# Postgresql Database
POSTGRESQL_USERNAME = os.getenv("POSTGRESQL_USERNAME")
POSTGRESQL_PASSWORD = os.getenv("POSTGRESQL_PASSWORD")
POSTGRESQL_HOST = os.getenv("POSTGRESQL_HOST")
POSTGRESQL_PORT = os.getenv("POSTGRESQL_PORT")
POSTGRESQL_MLFLOW_DB = os.getenv("POSTGRESQL_MLFLOW_DB")

# MongoDB Database
MONGODB_ROOT_USERNAME = os.getenv("MONGODB_ROOT_USERNAME")
MONGODB_ROOT_PASSWORD = os.getenv("MONGODB_ROOT_PASSWORD")
MONGODB_ROOT_HOST = os.getenv("MONGODB_ROOT_HOST")
MONGO_CLIENT = (
    f"mongodb://{MONGODB_ROOT_USERNAME}:{MONGODB_ROOT_PASSWORD}@{MONGODB_ROOT_HOST}:27017"
)

MLFLOW_HOST = os.getenv("MLFLOW_HOST")
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:5000/"

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        }
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)
