from fastapi import FastAPI

from pydantic import BaseModel, StringConstraints, Field
from typing import List, Dict, Optional
from typing_extensions import Annotated

class LLMResponse(BaseModel):
    status: int
    data: Optional[dict]

# class Message(BaseModel):
#     role: Annotated[str, StringConstraints(strict=True, pattern=r"^(system|user|assistant)$")]
#     content: str

class LLMConfig(BaseModel):
    do_sample: Optional[bool] = Field(description="是否进行采样", default=True)
    top_p: Optional[float] = Field(description="Top-p采样值", default=0.9)
    top_k: Optional[int] = Field(description="Top-k采样值", default=50)
    temperature: Optional[float] = Field(description="温度系数", default=0.7)
    max_new_tokens: Optional[int] = Field(description="生成tokens的最大数量", default=512)

class LLMRequest(BaseModel):
    messages: List[Dict] = Field(
        description="消息列表",
        default=[
            {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {"role": "user", "content": "你好，你是谁？"}
        ]
    )
    tools: Optional[List[Dict]] = Field(description="工具列表", default=None)
    stream: Optional[bool] = Field(description="流式输出", default=False)
    llm_config: Optional[LLMConfig] = Field(description="模型参数配置", default=LLMConfig())
    


def create_app(models)->FastAPI:
    app = FastAPI()

    @app.get("/ping", summary="检查接口状态")
    def ping():
        return {"data": "OK"}
    
    @app.get("/models", summary="列出所有模型")
    def get_models():
        return {"data": list(models.keys())}

    def create_route(model_name):
        @app.post(f"/{model_name}", response_model=LLMResponse)
        def chat(request: LLMRequest):
            response = models[model_name](messages=request.messages, tools=request.tools, stream=request.stream, **request.llm_config.model_dump())
            return {"status": 0, "data": response}
        
    for name in models.keys():
        create_route(name)
    
    return app


# class Message(BaseModel):
#     role: Annotated[str, StringConstraints(strict=True, pattern=r"^(system|user|assistant)$")] = Field(description="消息角色", default="user")
#     content: str = Field(description="消息内容", default="你好，你是谁？")

# class LLMRequest(BaseModel):
#     messages: List[Message] = Field(
#         ...,
#         description="消息列表"
#     )
