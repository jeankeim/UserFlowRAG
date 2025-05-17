import gradio as gr
import requests
import os
import json
from typing import Optional

import yaml
from pathlib import Path

# 从config.yaml读取配置
config_path = Path("config/config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# 后端API配置
API_BASE = "http://localhost:8000"
API_TOKEN = config['auth']['tokens'][0]  # 使用配置中的第一个token

def call_query_api(question: str):
    """调用查询API(流式版本)"""
    try:
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        print(f"调用API: {API_BASE}/query, 问题: {question}")  # 调试日志
        
        with requests.post(
            f"{API_BASE}/query",
            json={"question": question},
            headers=headers,
            stream=True
        ) as response:
            response.raise_for_status()
            buffer = ""
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        print(f"收到原始响应: {decoded_line}")  # 调试日志
                        data = json.loads(decoded_line)
                        
                        # 处理不同响应格式
                        if 'answer' in data:
                            buffer += data['answer']
                            yield buffer
                        elif 'output' in data:
                            buffer += data['output']
                            yield buffer
                        else:
                            yield f"未知响应格式: {data}"
                            
                    except json.JSONDecodeError as je:
                        print(f"JSON解析错误: {je}, 原始数据: {line}")
                        continue
                    except Exception as e:
                        print(f"处理响应错误: {e}")
                        continue
                        
    except requests.exceptions.RequestException as re:
        error_msg = f"API请求失败: {str(re)}"
        print(error_msg)
        yield error_msg
    except Exception as e:
        error_msg = f"发生未知错误: {str(e)}"
        print(error_msg)
        yield error_msg

def call_upload_api(file) -> Optional[str]:
    """调用文件上传API"""
    try:
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        files = {"file": (file.name, file)}
        response = requests.post(
            f"{API_BASE}/documents/upload",
            files=files,
            headers=headers
        )
        response.raise_for_status()
        return "文件上传成功！"
    except Exception as e:
        return f"上传失败: {str(e)}"

def update_token(new_token: str) -> str:
    """更新API令牌"""
    global API_TOKEN
    API_TOKEN = new_token
    return "API令牌已更新"

with gr.Blocks(title="RAG系统前端") as demo:
    gr.Markdown("# RAG系统交互界面")
    
    with gr.Tab("查询"):
        with gr.Row():
            query_input = gr.Textbox(
                label="输入您的问题",
                placeholder="请输入您想查询的内容..."
            )
            query_btn = gr.Button("提交查询")
        
        query_output = gr.Textbox(
            label="查询结果",
            interactive=False,
            every=0.1  # 设置更快的更新频率
        )
        
        query_btn.click(
            call_query_api,
            inputs=[query_input],
            outputs=[query_output],
            api_name="query"
        )
    
    with gr.Tab("上传文档"):
        file_input = gr.File(
            label="选择文档(PDF/TXT/DOCX)",
            file_types=[".pdf", ".txt", ".docx"]
        )
        upload_btn = gr.Button("上传文档")
        
        upload_output = gr.Textbox(
            label="上传状态",
            interactive=False
        )
        
        upload_btn.click(
            call_upload_api,
            inputs=[file_input],
            outputs=[upload_output]
        )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
