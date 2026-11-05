import requests
import json
import time
import os

def load_config(config_path='config.json'):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到。")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误：配置文件 '{config_path}' 格式无效。")
        exit(1)

def manage_model_keep_alive(api_url, model, keep_alive_value):
    """发送 /api/generate 请求来控制模型的 keep_alive 状态"""
    api_endpoint = f"{api_url}/api/generate"
    payload = {
        "model": model,
        "prompt": " ", # 需要一个非空 prompt 来触发加载/卸载
        "keep_alive": keep_alive_value,
        "stream": False # 不需要流式响应
    }
    try:
        print(f"正在发送请求以设置 keep_alive={keep_alive_value} for model '{model}'...")
        response = requests.post(api_endpoint, json=payload, timeout=60) # 增加超时
        response.raise_for_status()
        print(f"模型 '{model}' 的 keep_alive 已设置为 {keep_alive_value}。")
        return True
    except requests.exceptions.Timeout:
        print(f"\n错误：连接到 Ollama API ({api_endpoint}) 以管理 keep_alive 时超时。")
        return False
    except requests.exceptions.ConnectionError:
         print(f"\n错误：无法连接到 Ollama API ({api_endpoint}) 以管理 keep_alive。请确保服务正在运行。")
         return False
    except requests.exceptions.RequestException as e:
        print(f"\n错误：设置 keep_alive={keep_alive_value} 时出错: {e}")
        # 尝试打印响应体以获取更多信息
        try:
            print(f"Ollama 响应: {e.response.text}")
        except:
            pass
        return False
    except Exception as e:
        print(f"\n设置 keep_alive 时发生意外错误: {e}")
        return False


def call_ollama(api_url, model, messages, options=None):
    """
    调用 Ollama API 并返回流式响应、首个 token 时间和总时间。
    """
    api_endpoint = f"{api_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        # 如果 options 存在且不为 None，则添加到 payload 中
        **({"options": options} if options else {})
    }

    start_time = time.perf_counter()
    first_token_time = None
    full_response = ""
    time_to_first_token = None

    try:
        # 使用 with 语句确保连接被关闭
        with requests.post(api_endpoint, json=payload, stream=True, timeout=300) as response: # 增加超时时间
            response.raise_for_status()  # 检查 HTTP 错误，例如 404, 500 等
            for i, chunk in enumerate(response.iter_lines()):
                if chunk:
                    # 在接收到第一个数据块时记录时间
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        time_to_first_token = first_token_time - start_time

                    decoded_chunk = chunk.decode('utf-8')
                    try:
                        response_part = json.loads(decoded_chunk)
                        # 提取并打印内容
                        if 'message' in response_part and 'content' in response_part['message']:
                             content = response_part['message']['content']
                             print(content, end='', flush=True) # 实时打印 token
                             full_response += content
                        # 检查流是否结束 ('done' 字段通常表示流结束)
                        if response_part.get('done'):
                            break
                    except json.JSONDecodeError:
                        # 如果某个块不是有效的 JSON，记录错误并尝试继续
                        print(f"\n警告：无法解析来自 Ollama 的 JSON 块: {decoded_chunk}")
                        continue # 跳过这个无效的块

        # 循环结束后记录总时间
        end_time = time.perf_counter()
        time_to_complete = end_time - start_time
        print() # 添加换行符，因为之前的 print 没有换行

        return full_response, time_to_first_token, time_to_complete

    except requests.exceptions.Timeout:
        print(f"\n错误：连接到 Ollama API 超时 ({api_endpoint})。请检查 Ollama 服务是否正在运行以及网络连接。")
        return None, None, None
    except requests.exceptions.ConnectionError:
         print(f"\n错误：无法连接到 Ollama API ({api_endpoint})。请确保 Ollama 服务正在运行并且 URL 正确。")
         return None, None, None
    except requests.exceptions.RequestException as e:
        print(f"\n错误：调用 Ollama API 时出错: {e}")
        return None, None, None
    except Exception as e: # 捕获其他潜在错误
        print(f"\n处理响应时发生意外错误: {e}")
        return None, None, None


def main():
    """主程序逻辑"""
    config = load_config()
    ollama_url = config.get('ollama_url')
    model = config.get('model')
    # 从配置加载 options，默认为 None
    options = config.get('options', None)
    # initial_prompt 不再从 config.json 读取
    # 从配置加载 keep_alive 设置，默认为 0 (不保持)
    keep_alive_setting = config.get('keep_alive_on_start', 0)
    # 从配置获取 initial_prompt_path
    initial_prompt_path = config.get('initial_prompt_path')

    if not ollama_url or not model:
        print("错误：配置文件中缺少 'ollama_url' 或 'model'。")
        exit(1)

    conversation_history = []
    # 尝试从指定的路径加载初始 prompt 文件
    if initial_prompt_path:
        initial_prompt_content = ""
        try:
            # 使用配置中指定的路径
            with open(initial_prompt_path, 'r', encoding='utf-8') as f:
                initial_prompt_content = f.read().strip()
            if initial_prompt_content:
                print(f"已从 '{initial_prompt_path}' 加载初始 Prompt。")
                conversation_history.append({"role": "system", "content": initial_prompt_content})
            else:
                print(f"文件 '{initial_prompt_path}' 为空，未加载初始 Prompt。")
        except FileNotFoundError:
            print(f"错误：未找到指定的初始 Prompt 文件 '{initial_prompt_path}'。")
        except Exception as e:
            print(f"读取初始 Prompt 文件 '{initial_prompt_path}' 时出错: {e}")
    else:
        print("配置文件中未指定 'initial_prompt_path'，将不使用初始 Prompt。")
# Removed duplicate/leftover except block

    # 启动时预加载并设置 keep_alive (如果配置了)
    if keep_alive_setting != 0:
        if not manage_model_keep_alive(ollama_url, model, keep_alive_setting):
             print("警告：无法在启动时设置模型的 keep_alive。程序将继续，但模型可能不会按预期保持加载。")

    print(f"Ollama 对话客户端已启动 (模型: {model}, Keep Alive: {keep_alive_setting})。输入 'quit' 或 'exit' 退出。")
    print("-" * 30)

    try:
        while True:
            # --- 开始 while 循环内的代码块 ---
            try:
                user_input = input("你: ")
                if user_input.lower() in ['quit', 'exit']:
                    break # 退出 while 循环

                if not user_input.strip(): # 忽略空输入
                    continue

                # 将用户输入添加到历史记录
                conversation_history.append({"role": "user", "content": user_input})

                print("VTS精灵: ", end='', flush=True) # 提示助手正在回复
                response_content, ttfb, ttc = call_ollama(ollama_url, model, conversation_history, options)

                if response_content is not None:
                    # 只有成功获取回复才将其添加到历史记录
                    conversation_history.append({"role": "assistant", "content": response_content})

                    # 打印计时信息
                    print("-" * 30)
                    if ttfb is not None:
                        print(f"Time to first token (TTFB): {ttfb:.4f} seconds")
                    else:
                        print("Time to first token (TTFB): 未记录")
                    if ttc is not None:
                        print(f"Time to complete sentence (TTC): {ttc:.4f} seconds")
                    else:
                        print("Time to complete sentence (TTC): 未记录")
                    print("-" * 30)
                else:
                    # 如果 API 调用失败，从历史记录中移除刚才添加的用户输入
                    if conversation_history and conversation_history[-1]["role"] == "user":
                         conversation_history.pop()
                    print("未能从 Ollama 获取回复，请检查错误信息并重试。")
                    print("-" * 30)

            except EOFError: # 处理 Ctrl+D
                print("\n检测到文件结束符，正在退出...")
                break # 退出 while 循环
            except KeyboardInterrupt: # 处理 Ctrl+C
                 print("\n收到中断信号，正在退出...")
                 break # 退出 while 循环
            # --- 结束 while 循环内的代码块 ---

    # --- 开始 finally 块 (在 try...except 外部, 但仍在 main 函数内) ---
    finally:
        # 程序退出前卸载模型 (无论循环如何结束)
        print("-" * 30)
        print("正在准备退出程序...")
        if keep_alive_setting != 0: # 只有在启动时设置了保持活动才需要卸载
            print(f"正在尝试卸载模型 '{model}' (设置 keep_alive=0)...")
            if not manage_model_keep_alive(ollama_url, model, 0):
                 print("警告：尝试卸载模型时出错。")
            else:
                 print(f"模型 '{model}' 卸载请求已发送。")
        else:
            print("启动时未设置 keep_alive，无需发送卸载请求。")
        print("-" * 30)
    # --- 结束 finally 块 ---

    print("\n再见！") # 这个应该在 finally 块之后

if __name__ == "__main__":
    # 确保脚本在正确的目录下运行，以便找到 config.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    main()