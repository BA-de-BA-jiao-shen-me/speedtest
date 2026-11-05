# Ollama Bridge Service API 文档

## 概述

本服务作为一个桥梁，连接客户端应用程序和后端的 Ollama API。它提供了一个流式的聊天接口，封装了与 Ollama 的交互逻辑，包括系统提示注入、对话历史管理、模型 Keep-alive 以及响应格式化。

## 基本 URL

服务运行在 `http://<service_host>:<service_port>/` 上。默认配置下为 `http://127.0.0.1:8000/`。
`<service_host>` 和 `<service_port>` 在服务端的 `config.json` 文件中配置。

## 认证

本接口目前不需要认证。

## 配置

服务的核心行为（如使用的 Ollama 模型、上下文长度 `num_ctx`、温度 `temperature`、模型保持时间 `keep_alive`、最大对话轮次 `max_conversation_turns`、系统提示文件路径 `system_prompt_path`、日志文件路径 `log_directory` 等）均通过服务端的 `config.json` 文件进行配置。客户端无法在请求时覆盖这些设置。

## 端点 (Endpoints)

### 1. 提供测试前端

*   **路径**: `/`
*   **方法**: `GET`
*   **描述**: 返回一个简单的 HTML 页面 (`index.html`)，可用于在本服务上进行基本的聊天测试。
*   **成功响应**:
    *   **状态码**: `200 OK`
    *   **内容类型**: `text/html`
    *   **响应体**: `index.html` 的文件内容。
*   **错误响应**:
    *   **状态码**: `404 Not Found` - 如果服务器找不到 `index.html` 文件。

### 2. 聊天接口 (流式)

*   **路径**: `/chat`
*   **方法**: `POST`
*   **描述**: 接收对话历史，与配置的 Ollama 模型进行交互，并以 Server-Sent Events (SSE) 的形式流式返回模型的回复。服务会自动处理系统提示的添加和对话历史的截断。
*   **请求头**:
    *   `Content-Type: application/json`
*   **请求体 (JSON)**:
    ```json
    {
      "messages": [
        {
          "role": "system", // 可选，如果提供，将覆盖服务器配置的系统提示
          "content": "你是系统提示词。"
        },
        {
          "role": "user",
          "content": "你好！"
        },
        {
          "role": "assistant",
          "content": "你好！有什么可以帮你的吗？"
        },
        {
          "role": "user",
          "content": "请介绍一下你自己。"
        }
        // ... 更多历史消息
      ]
    }
    ```
    *   `messages` (必填, array): 一个包含对话消息对象的列表。
        *   `role` (必填, string): 消息的角色，必须是 "system", "user", 或 "assistant"。
        *   `content` (必填, string): 消息的文本内容。
*   **成功响应**:
    *   **状态码**: `200 OK`
    *   **内容类型**: `text/event-stream`
    *   **响应体**: 一系列 Server-Sent Events (SSE)。每个事件遵循以下格式：
        ```
        data: <json_string>\n\n
        ```
        其中 `<json_string>` 是一个符合以下结构的 JSON 对象 (仿 OpenAI API 结构，并加入了 `usage.turn_count`):
        ```json
        {
          "id": "chatcmpl-xxxxxxxxxxxx", // 本次请求的唯一 ID
          "object": "chat.completion.chunk",
          "created": 1677652288, // 请求创建的 Unix 时间戳
          "model": "config_model_name", // 服务端配置的模型名称
          "choices": [
            {
              "index": 0,
              "delta": {
                // 可能包含 "role": "assistant" (通常在第一个有效块)
                // 或 "content": "模型生成的文本块"
                // 或 {} (在结束块中)
              },
              "finish_reason": null | "stop" | "length" // 在最后一个块中指示结束原因，否则为 null
            }
          ],
          "usage": {
            "turn_count": 5 // 当前对话上下文中包含的有效轮次数 (基于服务器截断后)
          }
        }
        ```
*   **错误响应 (非流式错误)**:
    *   **状态码**: `422 Unprocessable Entity` - 请求体 JSON 格式错误或缺少 `messages` 字段。
    *   **状态码**: `500 Internal Server Error` - 服务器内部错误（例如配置错误、意外异常）。
    *   **状态码**: `503 Service Unavailable` - 无法连接到后端的 Ollama 服务。
    *   **响应体 (JSON)**:
        ```json
        {
          "detail": "错误描述信息"
        }
        ```
*   **错误响应 (流式错误)**:
    *   如果流在传输过程中出错（例如 Ollama 连接中断），流可能会非正常终止。服务器端会记录错误日志。客户端可能会收到一个 `event: error` 事件（如果实现），或者只是连接被关闭。

## HTTP 标准测试用例 (`curl`)

假设服务运行在默认的 `http://127.0.0.1:8000`。

```bash
curl -N -X POST http://127.0.0.1:8000/chat \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "你好，请用中文简单介绍一下你自己"}
  ]
}'
```

**命令解释**:

*   `curl`: HTTP 客户端工具。
*   `-N`: `--no-buffer`，禁用输出缓冲，对于接收流式响应 (SSE) 很重要，可以立即看到每个事件。
*   `-X POST`: 指定 HTTP 请求方法为 POST。
*   `http://127.0.0.1:8000/chat`: 请求的目标 URL。
*   `-H "Content-Type: application/json"`: 设置请求头，表明请求体是 JSON 格式。
*   `-d '{...}'`: 设置请求体内容。注意 JSON 字符串在命令行中可能需要适当的转义（在此简单示例中通常不需要）。

**预期输出**:

你会看到一系列以 `data: ` 开头的行，每行包含一个 JSON 对象，实时显示模型的回复，直到最后一个事件包含 `finish_reason`。

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}],"usage":{"turn_count":1}}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}],"usage":{"turn_count":1}}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"！我"},"finish_reason":null}],"usage":{"turn_count":1}}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"是一个"},"finish_reason":null}],"usage":{"turn_count":1}}

# ... 更多内容块 ...

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"turn_count":1}}