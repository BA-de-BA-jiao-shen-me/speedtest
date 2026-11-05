import httpx
import json
import time
import logging
import uuid
from typing import List, Dict, Any, AsyncGenerator, Tuple, Optional
import re

# --- Configuration Loading ---

def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"错误：配置文件 '{config_path}' 未找到。")
        raise
    except json.JSONDecodeError:
        logging.error(f"错误：配置文件 '{config_path}' 格式无效。")
        raise
    except Exception as e:
        logging.error(f"加载配置文件 '{config_path}' 时发生未知错误: {e}")
        raise

# --- Keep-alive Management ---

def manage_keep_alive(api_url: str, model: str, keep_alive_value: Any) -> bool:
    """使用 /api/generate 控制模型的 keep_alive 状态"""
    import requests
    api_endpoint = f"{api_url}/api/generate"
    payload = {
        "model": model,
        "prompt": " ", # 需要一个非空 prompt 来触发加载/卸载
        "keep_alive": keep_alive_value,
        "stream": False
    }
    try:
        logging.info(f"正在发送请求以设置 keep_alive={keep_alive_value} for model '{model}' at {api_endpoint}...")
        response = requests.post(api_endpoint, json=payload, timeout=60)
        response.raise_for_status()
        logging.info(f"模型 '{model}' 的 keep_alive 已设置为 {keep_alive_value}。")
        return True
    except requests.exceptions.Timeout:
        logging.error(f"错误：连接到 Ollama API ({api_endpoint}) 以管理 keep_alive 时超时。")
        return False
    except requests.exceptions.ConnectionError:
         logging.error(f"错误：无法连接到 Ollama API ({api_endpoint}) 以管理 keep_alive。请确保服务正在运行。")
         return False
    except requests.exceptions.RequestException as e:
        error_msg = f"错误：设置 keep_alive={keep_alive_value} 时出错: {e}"
        try:
            error_msg += f" Ollama 响应: {e.response.text}"
        except:
            pass
        logging.error(error_msg)
        return False
    except Exception as e:
        logging.error(f"设置 keep_alive 时发生意外错误: {e}")
        return False

# --- History Truncation ---

def truncate_history(messages: List[Dict[str, str]], max_turns: int) -> Tuple[List[Dict[str, str]], int]:
    """
    根据最大轮次数截断对话历史。
    保留第一个 system 消息（如果存在）和最后 max_turns 轮对话。
    返回截断后的消息列表和实际轮次数。
    """
    if max_turns <= 0: # 不限制轮次
        actual_turns = sum(1 for msg in messages if msg.get("role") == "user")
        return messages, actual_turns

    system_message = None
    conversation_messages = []
    for msg in messages:
        if msg.get("role") == "system" and system_message is None: # 只保留第一个 system message
            system_message = msg
        elif msg.get("role") in ["user", "assistant"]:
            conversation_messages.append(msg)

    # 需要保留的消息数量 = max_turns * 2 (user + assistant)
    messages_to_keep_count = max_turns * 2
    truncated_conversation = conversation_messages[-messages_to_keep_count:]

    final_messages = []
    if system_message:
        final_messages.append(system_message)
    final_messages.extend(truncated_conversation)

    # 计算实际轮次 (只计算 user 消息)
    actual_turns = sum(1 for msg in truncated_conversation if msg.get("role") == "user")

    if len(messages) > len(final_messages):
         logging.debug(f"History truncated from {len(messages)} to {len(final_messages)} messages ({actual_turns} turns).")

    return final_messages, actual_turns

# --- Synchronous Chat Call ---

def send_chat_message_sync(
    api_url: str,
    model: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any]
) -> Optional[str]:
    """发送单次、非流式聊天请求并获取完整响应"""
    import requests
    api_endpoint = f"{api_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False, # <-- 非流式
        "options": options
    }
    try:
        logging.info(f"向 {api_endpoint} 发送同步聊天请求...")
        try:
            payload_str = json.dumps(payload, ensure_ascii=False, indent=2)
            logging.debug(f"同步请求 Payload:\n{payload_str}")
        except Exception as log_e:
             logging.error(f"无法序列化或记录同步 payload: {log_e}")

        response = requests.post(api_endpoint, json=payload, timeout=300) # 较长超时以等待响应
        response.raise_for_status()
        response_data = response.json()

        # 从响应中提取助手消息内容
        assistant_message = response_data.get('message', {}).get('content')
        if assistant_message:
            logging.info(f"收到来自 {api_endpoint} 的同步响应。")
            return assistant_message.strip()
        else:
            logging.warning(f"从 {api_endpoint} 收到的同步响应中未找到助手消息内容。响应: {response_data}")
            return None

    except requests.exceptions.Timeout:
        logging.error(f"错误：连接到 Ollama API ({api_endpoint}) 进行同步聊天时超时。")
        return None
    except requests.exceptions.ConnectionError:
         logging.error(f"错误：无法连接到 Ollama API ({api_endpoint}) 进行同步聊天。")
         return None
    except requests.exceptions.RequestException as e:
        error_msg = f"错误：调用 Ollama API 进行同步聊天时出错: {e}"
        try:
            error_msg += f" Ollama 响应: {e.response.text}"
        except:
            pass
        logging.error(error_msg)
        return None
    except json.JSONDecodeError:
        logging.error(f"错误：无法解析来自 Ollama API ({api_endpoint}) 的同步响应 JSON。响应文本: {response.text}")
        return None
    except Exception as e:
        logging.error(f"进行同步聊天时发生意外错误: {e}", exc_info=True)
        return None

# --- Streaming Logic (异步流式重写，完全无requests依赖) ---

async def stream_ollama_openai_style_events(
    messages: List[Dict[str, str]],
    config: Dict[str, Any],
    turn_count: int
) -> AsyncGenerator[str, None]:
    """
    调用 Ollama /api/chat 流式接口，并将响应转换为仿 OpenAI SSE 事件格式。
    """
    api_url = config['ollama_url']
    model = config['model']
    options = config.get('options', {})
    api_endpoint = f"{api_url}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options
    }

    request_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())
    start_time = time.perf_counter()
    first_token_time = None
    ttfb = None
    ttc = None
    finish_reason = "stop" # 默认值

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", api_endpoint, json=payload) as response:
                response.raise_for_status()
                first_chunk = True
                accumulated_content = ""
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        ollama_chunk = json.loads(line)
                        delta = {}
                        current_content_part = ""

                        # 第一个块，发送role
                        if first_chunk and ollama_chunk.get('message', {}).get('role') == 'assistant':
                            delta["role"] = "assistant"
                            first_chunk = False
                            sse_event = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                                "usage": {"turn_count": turn_count}
                            }
                            yield f"data: {json.dumps(sse_event, ensure_ascii=False)}\n\n"
                            delta = {}

                        # 提取内容
                        if 'message' in ollama_chunk and 'content' in ollama_chunk['message']:
                            current_content_part = ollama_chunk['message']['content']
                            # current_content_part = re.sub(r'<think>.*?</think>', '', current_content_part, flags=re.DOTALL)
                            # current_content_part = re.sub(r'<think\s*/?>', '', current_content_part)
                            # current_content_part = re.sub(r'</think>', '', current_content_part)
                            delta["content"] = current_content_part
                            accumulated_content += current_content_part

                        # 第一个内容到达
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            ttfb = first_token_time - start_time
                            logging.debug(f"Request {request_id}: First chunk received at {ttfb:.4f}s")

                        # 检查是否结束
                        is_done = ollama_chunk.get('done', False)
                        if is_done:
                            finish_reason = ollama_chunk.get('done_reason', 'stop')
                            logging.debug(f"Request {request_id}: Stream done detected.")

                        # 只有当 delta 不为空时才发送事件
                        if delta:
                            sse_event = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": model,
                                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                                "usage": {"turn_count": turn_count}
                            }
                            yield f"data: {json.dumps(sse_event, ensure_ascii=False)}\n\n"

                        # 结束时发送最后块
                        if is_done:
                            final_event = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": model,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                                "usage": {"turn_count": turn_count}
                            }
                            yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
                            break

                    except json.JSONDecodeError:
                        logging.warning(f"无法解析来自 Ollama 的 JSON 块: {line}")
                        continue
                    except Exception as e_inner:
                        logging.error(f"处理 Ollama 块时发生错误: {e_inner}", exc_info=True)
                        continue

    except httpx.TimeoutException:
        logging.error(f"错误：连接到 Ollama API ({api_endpoint}) 时超时。")
        raise ConnectionError("Ollama API timeout")
    except httpx.HTTPError as e:
        logging.error(f"HTTP错误：{e}")
        raise ConnectionError(f"Ollama API HTTP error: {e}")
    except Exception as e:
        logging.error(f"处理流时发生意外错误: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error during streaming: {e}")
    finally:
        end_time = time.perf_counter()
        ttc = end_time - start_time
        ttfb_str = f"{ttfb:.4f}s" if ttfb is not None else "N/A"
        ttc_str = f"{ttc:.4f}s" if ttc is not None else "N/A"
        logging.info(f"请求 {request_id}: 流处理完成。TTFB: {ttfb_str}, TTC: {ttc_str}")

# --- Logging Setup (Basic) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
