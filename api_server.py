#!/usr/bin/env python3
"""
Nano-vLLM API Server
基于 FastAPI 的在线推理服务，兼容 OpenAI API 格式
"""
import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from nanovllm import LLM, SamplingParams


# ============ 数据模型 ============

class Message(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: system/user/assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求"""
    model: str = Field(default="default", description="模型名称")
    messages: List[Message] = Field(..., description="对话历史")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=256, ge=1, le=4096, description="最大生成token数")
    stream: bool = Field(default=False, description="是否流式输出")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="nucleus采样参数")
    n: int = Field(default=1, ge=1, le=10, description="生成的响应数量")
    stop: Optional[List[str]] = Field(default=None, description="停止词列表")


class CompletionRequest(BaseModel):
    """文本完成请求"""
    model: str = Field(default="default", description="模型名称")
    prompt: str = Field(..., description="提示文本")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    stream: bool = Field(default=False)


class Choice(BaseModel):
    """响应选择"""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """token使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """聊天完成响应"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class StreamChoice(BaseModel):
    """流式响应选择"""
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """流式聊天完成响应"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """模型列表"""
    object: str = "list"
    data: List[ModelInfo]


# ============ 全局变量 ============

llm_engine: Optional[LLM] = None
model_config: Dict[str, Any] = {}


# ============ 生命周期管理 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("🚀 正在初始化 Nano-vLLM API Server...")
    if llm_engine is None:
        print("⚠️  警告: 模型未加载，请先调用 init_model()")
    else:
        print(f"✅ 模型已加载: {model_config.get('model_path', 'unknown')}")
    
    yield
    
    # 关闭时
    print("🛑 正在关闭 API Server...")
    if llm_engine:
        try:
            llm_engine.exit()
            print("✅ 模型已卸载")
        except Exception as e:
            print(f"⚠️  模型卸载时出错: {e}")


# ============ FastAPI 应用 ============

app = FastAPI(
    title="Nano-vLLM API",
    description="轻量级 vLLM 推理服务，兼容 OpenAI API",
    version="0.2.0",
    lifespan=lifespan
)


# ============ 辅助函数 ============

def init_model(
    model_path: str,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    max_num_seqs: int = 512,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9
):
    """初始化模型"""
    global llm_engine, model_config
    
    print(f"📦 正在加载模型: {model_path}")
    print(f"   - Tensor Parallel: {tensor_parallel_size}")
    print(f"   - Eager Mode: {enforce_eager}")
    print(f"   - Max Sequences: {max_num_seqs}")
    print(f"   - Max Model Length: {max_model_len}")
    
    try:
        llm_engine = LLM(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization
        )
        
        model_config = {
            "model_path": model_path,
            "model_name": model_path.rstrip("/").split("/")[-1],
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": enforce_eager,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
        }
        
        print(f"✅ 模型加载成功: {model_config['model_name']}")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def messages_to_prompt(messages: List[Message]) -> str:
    """将消息列表转换为 prompt"""
    # 简单实现，实际应该使用模型的 chat template
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            prompt += f"User: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
    prompt += "Assistant: "
    return prompt


async def generate_stream(
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    model_name: str
) -> AsyncGenerator[str, None]:
    """真正的流式生成 - 逐个 token 返回"""
    created = int(time.time())
    
    loop = asyncio.get_event_loop()
    
    # 创建一个迭代器来获取流式生成的结果
    def stream_generator():
        return llm_engine.generate_stream(prompt, sampling_params)
    
    # 在后台线程执行
    stream_iter = await loop.run_in_executor(None, stream_generator)
    
    try:
        # 逐个获取生成的token
        while True:
            # 在后台线程获取下一个chunk
            try:
                chunk_data = await loop.run_in_executor(None, lambda: next(stream_iter, None))
                
                if chunk_data is None:
                    break
                
                # 发送新生成的文本
                if chunk_data["text"]:
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model_name,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta={"content": chunk_data["text"]},
                                finish_reason="stop" if chunk_data["finished"] else None
                            )
                        ]
                    )
                    yield f"data: {response.model_dump_json()}\n\n"
                
                # 如果完成，退出
                if chunk_data["finished"]:
                    break
                    
            except StopIteration:
                break
                
    except Exception as e:
        print(f"流式生成错误: {e}")
        # 发送错误信息
        error_response = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[
                StreamChoice(
                    index=0,
                    delta={"content": f"\n[错误: {str(e)}]"},
                    finish_reason="error"
                )
            ]
        )
        yield f"data: {error_response.model_dump_json()}\n\n"
    
    finally:
        # 发送结束标记
        yield "data: [DONE]\n\n"


# ============ API 路由 ============

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Nano-vLLM API Server",
        "version": "0.2.0",
        "status": "running" if llm_engine else "not_initialized",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completion": "/v1/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy" if llm_engine else "not_initialized",
        "model_loaded": llm_engine is not None,
        "model_name": model_config.get("model_name", "none")
    }


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """列出可用模型"""
    if not llm_engine:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    return ModelList(
        data=[
            ModelInfo(
                id=model_config.get("model_name", "default"),
                created=int(time.time()),
                owned_by="nano-vllm"
            )
        ]
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """创建聊天完成"""
    if not llm_engine:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    # 生成请求ID
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    
    # 转换消息为 prompt
    prompt = messages_to_prompt(request.messages)
    
    # 创建采样参数
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    # 流式响应
    if request.stream:
        return StreamingResponse(
            generate_stream(
                prompt,
                sampling_params,
                request_id,
                request.model
            ),
            media_type="text/event-stream"
        )
    
    # 非流式响应
    try:
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: llm_engine.generate([prompt], sampling_params, use_tqdm=False)
        )
        
        output = outputs[0]
        completion_text = output["text"]
        completion_tokens_count = len(output["token_ids"])
        
        # 估算 prompt tokens（简化版）
        prompt_tokens_count = len(prompt.split())
        
        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=completion_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens_count,
                completion_tokens=completion_tokens_count,
                total_tokens=prompt_tokens_count + completion_tokens_count
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """创建文本完成"""
    if not llm_engine:
        raise HTTPException(status_code=503, detail="模型未初始化")
    
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    try:
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: llm_engine.generate([request.prompt], sampling_params, use_tqdm=False)
        )
        
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "text": outputs[0]["text"],
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


# ============ 主函数 ============

def serve(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    enforce_eager: bool = False,
    max_num_seqs: int = 512,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.9,
    log_level: str = "info"
):
    """启动 API 服务器"""
    # 初始化模型
    init_model(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    # 启动服务器
    print(f"\n🌐 服务器启动在: http://{host}:{port}")
    print(f"📖 API 文档: http://{host}:{port}/docs")
    print(f"🔧 健康检查: http://{host}:{port}/health\n")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Nano-vLLM API Server")
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--enforce-eager", action="store_true", help="强制使用 eager 模式")
    parser.add_argument("--max-num-seqs", type=int, default=512, help="最大并发序列数")
    parser.add_argument("--max-model-len", type=int, default=4096, help="最大模型长度")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU内存利用率")
    parser.add_argument("--log-level", default="info", help="日志级别")
    
    args = parser.parse_args()
    
    serve(
        model_path=args.model_path,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        log_level=args.log_level
    )

