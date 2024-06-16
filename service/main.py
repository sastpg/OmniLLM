
import os
from config import config_manager
os.environ["CUDA_VISIBLE_DEVICES"] = config_manager.get('CUDA_VISIBLE_DEVICES') #指定cuda可见显卡编号

from api.api import create_app
from utils.logger import logger
from model.llama import Llama3
import uvicorn
import torch

if __name__ == "__main__":
    logger.info(torch.cuda.current_device())
    models_path = config_manager.get('models')
    models = {}
    for name in models_path:
        if name.startswith('llama-3'):
            models[name] = Llama3(model_path=models_path[name], device_map="auto")

    app = create_app(models)
    uvicorn.run(app, host="0.0.0.0", port=config_manager.get('port'))
