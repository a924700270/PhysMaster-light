# MCP Sandbox

[English](#mcp-sandbox) | [返回主目录](../README_CN.md)

PhysMaster 内置的 MCP (Model Context Protocol) 服务器，提供 `web_search` 和 `web_parse` 工具，用于在线文献检索与网页内容解析。

---

## 目录

- [环境安装](#环境安装)
- [配置文件](#配置文件)
- [启动服务](#启动服务)
- [停止服务](#停止服务)
- [服务架构](#服务架构)
- [常见问题](#常见问题)

---

## 环境安装

```bash
cd mcp_sandbox
pip install -r requirements.txt
```

依赖 Python >= 3.10（推荐 3.11）。

---

## 配置文件

启动前需要配置 `configs/` 目录下的 JSON 文件：

### `llm_call.json` — LLM 模型配置

配置各模型的 API 地址和密钥。`web_search` 和 `web_parse` 在解析网页内容时需要调用 LLM。

```json
{
    "DeepSeek/DeepSeek-V3-0324": {
        "url": "https://your-api-endpoint/v1/chat/completions",
        "authorization": "Bearer sk-your-api-key",
        "retry_time": 5
    }
}
```

可以配置多个模型，`web_agent.json` 和 `paper_agent.json` 中通过 `USE_MODEL` 字段指定实际使用的模型名。

### `mcp_config.json` — MCP 端口配置

```json
{
    "tool_api_url": "http://127.0.0.1:1234",
    "mcp_server_url": "http://127.0.0.1:8002"
}
```

- `tool_api_url`：api_proxy 服务地址（用于网页抓取代理）
- `mcp_server_url`：MCP 服务器暴露地址（即 PhysMaster `config.yaml` 中的 `mcp_url`）

### `web_agent.json` — 网页搜索配置

```json
{
    "serper_api_key": "your-serper-api-key",
    "search_region": "us",
    "search_lang": "en",
    "USE_MODEL": "DeepSeek/DeepSeek-V3-0324",
    "BASE_MODEL": "DeepSeek/DeepSeek-V3-0324"
}
```

- `serper_api_key`：Serper 搜索引擎 API 密钥，需要在 [serper.dev](https://serper.dev) 注册获取
- `search_region` / `search_lang`：默认搜索区域和语言
- `USE_MODEL`：搜索结果解析使用的模型（需在 `llm_call.json` 中配置）

### `paper_agent.json` — 论文解析配置

```json
{
    "USE_MODEL": "DeepSeek/DeepSeek-V3-0324",
    "BASE_MODEL": "DeepSeek/DeepSeek-V3-0324"
}
```

- `USE_MODEL`：论文内容解析使用的模型

### `server_list.json` — 服务列表配置

定义 MCP 服务器暴露的工具列表。通常无需修改。

---

## 启动服务

```bash
cd mcp_sandbox
./start_all.sh
```

启动成功后会显示：

```
============================================
X-Master MCP Services 状态
============================================
  ● api: 运行中 (PID: xxxxx, 端口: 1234)
  ● sandbox: 运行中 (PID: xxxxx, 端口: 8002)
============================================

MCP 端点:
  - mcp-sandbox: http://127.0.0.1:8002/mcp
```

其中 `sandbox` 的 8002 端口即对应 PhysMaster `config.yaml` 中的 `mcp_url`。

### 自定义端口

通过环境变量修改默认端口：

```bash
SANDBOX_PORT=9002 API_PORT=2234 ./start_all.sh
```

### 查看状态

```bash
./start_all.sh status
```

---

## 停止服务

```bash
./start_all.sh stop
```

也可以使用 `restart` 重启：

```bash
./start_all.sh restart
```

> **注意**：本地启动 MCP 时需要保持终端运行（或使用 nohup/screen/tmux）。终止启动终端后 MCP 服务器将断开。

---

## 服务架构

```
┌─────────────────────────────────────────┐
│              PhysMaster                 │
│  (library_retrive.py → MCP Client)     │
└──────────────┬──────────────────────────┘
               │ Streamable HTTP
               ▼
┌─────────────────────────────────────────┐
│         MCP Sandbox (:8002)             │
│  ┌─────────────┐  ┌──────────────┐     │
│  │ web_search   │  │  web_parse   │     │
│  └──────┬──────┘  └──────┬───────┘     │
│         │                │              │
│         ▼                ▼              │
│  ┌─────────────────────────────┐       │
│  │     api_proxy (:1234)       │       │
│  │  (网页抓取 / Serper 搜索)    │       │
│  └─────────────────────────────┘       │
└─────────────────────────────────────────┘
```

- **MCP Sandbox** (端口 8002)：MCP 协议服务器，暴露 `web_search` 和 `web_parse` 工具
- **api_proxy** (端口 1234)：后端代理，负责实际的网页抓取和 Serper API 调用
- **web_search**：调用 Serper 搜索引擎检索相关网页
- **web_parse**：抓取指定 URL 的网页内容，使用 LLM 提取与用户问题相关的信息

---

## 常见问题

**Q: 启动后 `web_search` 返回空结果**

检查 `configs/web_agent.json` 中的 `serper_api_key` 是否已填入有效密钥。

**Q: `web_parse` 返回 "failed to fetch web content"**

api_proxy 无法访问目标 URL。检查网络连接或代理设置。如果在内网环境，可能需要配置 HTTP 代理。

**Q: 端口被占用**

使用环境变量指定其他端口：`SANDBOX_PORT=9002 ./start_all.sh`，同时更新 PhysMaster `config.yaml` 中的 `mcp_url`。

**Q: `llm_call.json` 中的模型调用失败**

确保 `url` 和 `authorization` 填写正确，且 API 服务可访问。`url` 应指向 OpenAI 兼容的 `/v1/chat/completions` 端点。
