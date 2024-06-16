
import os
from config import config_manager
os.environ["CUDA_VISIBLE_DEVICES"] = config_manager.get('CUDA_VISIBLE_DEVICES') #指定cuda可见显卡编号

from typing import Type
from api.api import create_app
from utils.logger import logger
from model.common import AbstractModel
import uvicorn
import torch

from model.llama import Llama3
from model.glm import GLM

_NAME2MODEL = {
    "llama": Llama3,
    "glm": GLM
}

def get_model_cls(name) -> Type[AbstractModel]:
    model = name.split('-')[0]
    if model not in _NAME2MODEL:
        raise ValueError(f"{name} not support!")
    return _NAME2MODEL[model]


if __name__ == "__main__":
    logger.info(torch.cuda.current_device())
    models_path = config_manager.get('models')
    models = {}
    for name in models_path:
        model_cls = get_model_cls(name)
        model: AbstractModel = model_cls(model_path=models_path[name], device_map="auto")
        logger.info(f"loading {name} with model:{model}")
        models[name] = model

    app = create_app(models)
    uvicorn.run(app, host="0.0.0.0", port=config_manager.get('port'))
