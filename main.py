import argparse
import os
import logging
import datetime
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
import json
from ollama_client_lib import (
    load_config,
    manage_keep_alive,
    truncate_history,
    stream_ollama_openai_style_events,
    send_chat_message_sync
)
from rag_utils import KnowledgeBase

# --------- FastAPI 应用对象 ---------
app = FastAPI(
    title="Ollama Bridge Service",
    description="A service acting as a bridge to an Ollama API, providing streaming chat completions.",
    version="1.4.0"
)

# --------- CORS 配置 ---------
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- 日志配置 ---------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
file_handler = None
main_logger = logging.getLogger(__name__)

try:
    config = load_config()
    OLLAMA_URL = config['ollama_url']
    MODEL = config['model']
    KEEP_ALIVE_SETTING = config.get('keep_alive', 0)
    MAX_TURNS = config.get('max_conversation_turns', 0)
    SYSTEM_PROMPT_PATH = config.get('system_prompt_path', None)
    SYSTEM_RAG_PROMPT_PATH = config.get('system_rag_prompt_path', None)
    SYSTEM_PROMPT_CONTENT = None
    if SYSTEM_PROMPT_PATH:
        full_path = os.path.join(os.path.dirname(__file__), SYSTEM_PROMPT_PATH)
        with open(full_path, 'r', encoding='utf-8') as f:
            SYSTEM_PROMPT_CONTENT = f.read().strip()
    if SYSTEM_RAG_PROMPT_PATH:
        full_path = os.path.join(os.path.dirname(__file__), SYSTEM_RAG_PROMPT_PATH)
        with open(full_path, 'r', encoding='utf-8') as f:
            SYSTEM_RAG_PROMPT_PATH = f.read().strip()
except Exception as e:
    logger.critical(f"无法加载配置文件或配置项缺失: {e}", exc_info=True)
    exit(1)

SERVICE_HOST = config.get('service_host', '127.0.0.1')
SERVICE_PORT = config.get('service_port', 8000)
LOG_DIRECTORY = config.get('log_directory', 'log')
try:
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    current_date_str = datetime.date.today().strftime("%Y-%m-%d")
    log_file_name = f"{current_date_str}.log"
    log_file_path = os.path.join(LOG_DIRECTORY, log_file_name)
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    main_logger.info(f"日志将追加写入文件: {log_file_path}")
except Exception as log_e:
    log_path_info = f" (尝试目录: {LOG_DIRECTORY})" if 'LOG_DIRECTORY' in locals() else ""
    main_logger.error(f"无法配置日志文件写入{log_path_info}: {log_e}")

# --------- RAG知识库参数 ---------
RAG_KB_DIR = config.get('rag_kb_dir', 'knowledge_base')
RAG_TOP_K = config.get('rag_top_k', 5)

# --------- Chat接口用模型 ---------
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息的角色 ('system', 'user', or 'assistant')")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="包含对话历史的消息列表")

class RagRequest(BaseModel):
    question: str = Field(..., description="用户查询问题")

@app.get("/")
async def get_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)

# --------- RAG 拼接prompt ---------
def build_rag_prompt(query: str, kb_chunks: list) -> str:
    knowledge = "\n".join([f"【知识片段{i+1}】{text.strip()}" for i, (text, score) in enumerate(kb_chunks)])
    prompt = f"请基于以下知识库片段回答用户问题 ，尽量使用原文回答，未找到相关内容时请委婉告诉用户自己正在学习这方面的知识。\n\n{knowledge}\n\n【用户问题】：{query}"
    return prompt

def make_event_generator(messages_to_send, actual_turns, logger, config):
    async def event_generator():
        import time
        generator_start_time = time.perf_counter()
        ttfb_logged = False
        request_id_for_log = "unknown"
        full_assistant_response = ""
        try:
            async for event in stream_ollama_openai_style_events(messages_to_send, config, actual_turns):
                try:
                    data_part = event.split("data: ", 1)[1]
                    import json
                    json_data = json.loads(data_part)
                    request_id_for_log = json_data.get("id", request_id_for_log)
                except Exception:
                    pass
                try:
                    if event_content := json_data.get("choices", [{}])[0].get("delta", {}).get("content"):
                        full_assistant_response += event_content
                except Exception:
                    pass
                yield event
                if not ttfb_logged:
                    logger.debug(f"Request {request_id_for_log}: First event yielded to client.")
                    ttfb_logged = True
        except ConnectionError as e:
            logger.error(f"Request {request_id_for_log}: Ollama 连接错误: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        except RuntimeError as e:
            logger.error(f"Request {request_id_for_log}: 流处理运行时错误: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Request {request_id_for_log}: 处理流时发生意外错误: {e}", exc_info=True)
            yield f"event: error\ndata: {{'error': 'Internal server error during streaming.'}}\n\n"
        finally:
            request_end_time = time.perf_counter()
            total_request_time = request_end_time - generator_start_time
            final_request_id = request_id_for_log if request_id_for_log != "unknown" else "N/A"
            logger.info(f"事件流生成结束。总生成器时间: {total_request_time:.4f}s")
    return event_generator


@app.post("/chat")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    request_start_time = time.time()
    client_host = http_request.client.host if http_request.client else "unknown"
    logger.info(f"收到来自 {client_host} 的 /chat 请求。")
    if request.messages:
        last_user_msg = request.messages[-1].content
        logger.info(f"收到用户输入 (最后一条): '{last_user_msg[:150]}...'")
    else:
        logger.info("收到的请求消息列表为空。")

    incoming_messages = [msg.dict() for msg in request.messages]

    messages_with_system_prompt = []
    if SYSTEM_PROMPT_CONTENT and not any(msg.get('role') == 'system' for msg in incoming_messages):
        messages_with_system_prompt.append({"role": "system", "content": SYSTEM_PROMPT_CONTENT})
    messages_with_system_prompt.extend(incoming_messages)
    messages_to_send, actual_turns = truncate_history(
        messages_with_system_prompt, MAX_TURNS
    )

    event_generator = make_event_generator(messages_to_send, actual_turns, logger, config)
    try:
        if not OLLAMA_URL or not MODEL:
            raise HTTPException(status_code=500, detail="服务器配置不完整 (Ollama URL 或模型缺失)")
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except ConnectionError as e:
        logger.error(f"创建流响应时 Ollama 连接失败: {e}")
        raise HTTPException(status_code=503, detail=f"无法连接到 Ollama 服务: {e}")
    except Exception as e:
        logger.error(f"创建流响应时发生意外错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="创建流响应时发生内部服务器错误")

@app.post("/rag_chat")
async def rag_chat_endpoint(request: RagRequest, http_request: Request):
    if not kb or kb.vectorstore is None:
        raise HTTPException(status_code=500, detail="RAG知识库未加载")
    user_query = request.question.strip()
    kb_results = kb.query(user_query, top_k=RAG_TOP_K)
    rag_prompt = build_rag_prompt(user_query, kb_results)

    messages_for_rag = []
    messages_for_rag.append({"role": "system", "content": SYSTEM_RAG_PROMPT_PATH})
    messages_for_rag.append({"role": "user", "content": rag_prompt})
    messages_to_send, actual_turns = truncate_history(
        messages_for_rag, MAX_TURNS
    )

    event_generator = make_event_generator(messages_to_send, actual_turns, logger, config)
    try:
        if not OLLAMA_URL or not MODEL:
            raise HTTPException(status_code=500, detail="服务器配置不完整 (Ollama URL 或模型缺失)")
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except ConnectionError as e:
        logger.error(f"创建流响应时 Ollama 连接失败: {e}")
        raise HTTPException(status_code=503, detail=f"无法连接到 Ollama 服务: {e}")
    except Exception as e:
        logger.error(f"创建流响应时发生意外错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="创建流响应时发生内部服务器错误")


kb = None  # 必须全局，供接口和预热事件使用

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI 服务启动中...")
    global kb
    if kb is None:
        try:
            kb = KnowledgeBase(RAG_KB_DIR, build_new=False)
            logger.info("知识库已加载，支持RAG检索。")
        except Exception as e:
            logger.error(f"RAG知识库初始化失败: {e}")
    # LLM keep-alive与预热逻辑
    if KEEP_ALIVE_SETTING != 0 and KEEP_ALIVE_SETTING != "0":
        logger.info(f"正在尝试设置模型 '{MODEL}' 的 keep-alive 为 '{KEEP_ALIVE_SETTING}'...")
        success = manage_keep_alive(OLLAMA_URL, MODEL, KEEP_ALIVE_SETTING)
        if success:
            logger.info(f"模型 '{MODEL}' keep-alive 设置成功。")
        else:
            logger.warning(f"未能设置模型 '{MODEL}' 的 keep-alive。服务将继续运行，但模型可能不会按预期保持加载。")
    else:
        logger.info("Keep-alive 设置为 0 或未配置，跳过启动时的 keep-alive 设置。")
    if SYSTEM_PROMPT_CONTENT:
        logger.info("正在执行启动预热对话...")
        warmup_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CONTENT},
            {"role": "user", "content": "你如果清楚的知道了任务，就回复准备好了。"}
        ]
        warmup_options = config.get('options', {})
        response_content = send_chat_message_sync(OLLAMA_URL, MODEL, warmup_messages, warmup_options)
        if response_content:
            logger.info(f"启动预热完成，Ollama 回复: '{response_content}'")
        else:
            logger.warning("启动预热对话未能从 Ollama 获取有效回复。")
    else:
        logger.info("未加载系统提示，跳过启动预热对话。")
if __name__ == "__main__":
    print(f"将在 {SERVICE_HOST}:{SERVICE_PORT} 上启动服务...")
    uvicorn.run("main:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=False)
