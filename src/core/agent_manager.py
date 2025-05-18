import asyncio
from queue import Queue
import sys
from typing import Any, Dict, List, Optional, AsyncGenerator
from fastapi import HTTPException
import yaml
import os
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, AgentAction, AgentFinish

from dotenv import load_dotenv
load_dotenv()


class AgentManager:
    def __init__(self, agents_config_path: str, tasks_config_path: str):
        self.agents_config = self._load_config(agents_config_path)
        self.tasks_config = self._load_config(tasks_config_path)
        self.agents = {}
        self.tools = []
        self.stream_queues = {}  # 存储各agent的流式队列
        self.stream_handlers = {}  # 存储各agent的回调处理器
        self.initialize_agents()

    def _load_config(self, path: str) -> dict:
        """加载配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    class TokenStreamHandler(BaseCallbackHandler):
        def __init__(self, queue, done):
            self.queue = queue
            self.done = done
            
        async def on_llm_new_token(self, token: str, **kwargs) -> None:
            await self.queue.put(token)
            
        async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
            self.done.set()
            
        async def on_llm_error(self, error: BaseException, **kwargs) -> None:
            await self.queue.put(f"[ERROR] {str(error)}")
            self.done.set()

    def initialize_agents(self):
        """使用LangChain初始化所有代理"""
        for agent_id, config in self.agents_config['agents'].items():
            # 为每个agent创建流式处理队列和回调处理器
            queue = asyncio.Queue()
            done = asyncio.Event()
            handler = self.TokenStreamHandler(queue, done)
            
            # 创建带流式回调的LLM
            llm = ChatOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL"),
                model="deepseek-chat",
                temperature=0.7,
                streaming=True,
                callbacks=[handler]
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""
                Role: {config['role']}
                Goal: {config['goal']}
                Backstory: {config['backstory']}
                """),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])

            # 创建代理
            agent = initialize_agent(
                tools=self.tools,
                llm=llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                prompt=prompt,
                verbose=False
            )
            
            self.agents[agent_id] = agent
            self.stream_queues[agent_id] = queue
            self.stream_handlers[agent_id] = (handler, done)



    async def generate_stream(self, agent_id: str, input_text: str) -> AsyncGenerator[str, None]:
        """真正的token级流式处理"""
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        print(f"Starting true token streaming for agent {agent_id}")
        queue = self.stream_queues[agent_id]
        handler, done = self.stream_handlers[agent_id]
        done.clear()  # 重置完成状态
        
        # 启动任务
        task = asyncio.create_task(self.agents[agent_id].arun(input_text))
        
        # 流式返回token
        try:
            while not done.is_set() or not queue.empty():
                if not queue.empty():
                    token = await queue.get()
                    yield token
                else:
                    await asyncio.sleep(0.05)
        except Exception as e:
            yield f"[ERROR] {str(e)}"
        finally:
            yield "[DONE]"
