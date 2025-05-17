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

# class StreamingCallbackHandler(BaseCallbackHandler):
#     """自定义回调处理器用于捕获流式输出"""
#     def __init__(self):
#         self.queue = asyncio.Queue()
#         self.done = asyncio.Event()

#     async def on_llm_new_token(self, token: str, **kwargs) -> None:
#         """处理新的token"""
#         sys.stdout.write(token)
#         sys.stdout.flush()
#         await self.queue.put({"token": token})



#     async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
#         """处理LLM结束"""
#         self.done.set()

#     async def on_agent_action(self, action: AgentAction, **kwargs) -> None:
#         """处理代理动作"""
#         await self.queue.put({"action": action.log})

#     async def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
#         """处理代理结束"""
#         await self.queue.put({"output": finish.return_values["output"]})
#         self.done.set()

#     async def generate(self) -> AsyncGenerator[dict, None]:
#         """生成器函数返回流式输出"""
#         while not self.done.is_set() or not self.queue.empty():
#             if not self.queue.empty():
#                 yield await self.queue.get()
#             else:
#                 await asyncio.sleep(0.05)


# import os
# from langchain_openai import ChatOpenAI
# from typing import Iterator, Any

# class StreamingGeneratorCallback(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs: Any) -> Iterator[str]:
#         yield token



class StreamingGeneratorCallback(BaseCallbackHandler):
    """Callback handler that processes LLM streaming output."""
    
    def __init__(self, q: Queue):
        self.q = q
        print("StreamingGeneratorCallback initialized")
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        print("LLM streaming started")
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Received token: {token}")
        self.q.put(token)
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"LLM streaming completed with response: {str(response)[:200]}...")
        self.q.put(None)
        
    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        print(f"LLM streaming error: {str(error)}")
        self.q.put(error)



class AgentManager:
    def __init__(self, agents_config_path: str, tasks_config_path: str):
        self.agents_config = self._load_config(agents_config_path)
        self.tasks_config = self._load_config(tasks_config_path)
        self.agents = {}
        self.tools = []
        # 初始化LLM实例
        self.llm = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            model="deepseek-chat",
            temperature=0.7,
            streaming=True
        )
        self.initialize_agents()

    def _load_config(self, path: str) -> dict:
        """加载配置文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def initialize_agents(self):
        """使用LangChain初始化所有代理"""
        llm = ChatOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            model="deepseek-chat",
            temperature=0.7,
            verbose=True,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            streaming_options={
                "include_usage": True,
                "include_extra": True
            }
        )

        for agent_id, config in self.agents_config['agents'].items():
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
                # return_intermediate_steps=False
            )
            self.agents[agent_id] = agent





    def execute_task(self, agent_id: str, input_text: str) -> str:
        """执行单个任务"""
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("Input text must be a non-empty string")
            
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        try:
            agent = self.agents[agent_id]
            result = agent.run(input=input_text.strip())
            
            # 处理返回结果
            if isinstance(result, dict):
                return result.get('output', str(result))
            return str(result)
        except Exception as e:
            logging.error(f"Error executing task for agent {agent_id}: {str(e)}")
            raise ValueError(f"Task execution failed: {str(e)}") from e

    async def stream_task_chain(self, input_text: str):
        """流式执行任务链"""
        if not hasattr(self, 'tasks_config') or 'task_chain' not in self.tasks_config:
            raise ValueError("Task chain configuration not found")
            
        for task_id in self.tasks_config['task_chain']:
            task_config = self.tasks_config['tasks'][task_id]
            agent = self.agents[task_config['agent']]
           
            
            # 使用流式调用
            print(f"\nStarting stream for task: {task_id}")
            full_output = ""
            async for chunk in agent.astream(
                {"input": f"{task_config['description']}: {input_text}"}
            ):
                if isinstance(chunk, dict) and 'output' in chunk:
                    full_output += chunk['output']
                elif isinstance(chunk, str):
                    full_output += chunk
            
            if full_output:
                yield full_output
        
                
    def execute_task_chain(self, input_text: str) -> List[str]:
        """执行任务链中的所有任务"""
        if not hasattr(self, 'tasks_config') or 'task_chain' not in self.tasks_config:
            raise ValueError("Task chain configuration not found")
            
        results = []
        for task_id in self.tasks_config['task_chain']:
            task_config = self.tasks_config['tasks'][task_id]
            result = self.execute_task(
                task_config['agent'],
                f"{task_config['description']}: {input_text}"
            )
            results.append(result)
                
        return results

    async def stream_llm_response(self, agent_id: str, input_text: str) -> AsyncGenerator[dict, None]:
        """流式输出LLM响应"""
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("Input text must be a non-empty string")
            
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        callback_handler = StreamingGeneratorCallback()
        agent = self.agents[agent_id]
        # agent.callbacks = [callback_handler]
        # agent.invoke(input=input_text.strip())
        response = agent.stream(input=input_text.strip())  # Returns AgentFinish object
        
        for chunk in response:
            for token in callback_handler.on_llm_new_token(chunk):
                yield token


        # Handle the response
        # if hasattr(response, 'return_values') and 'output' in response.return_values:
        #     yield {'output': response.return_values['output']}
        # else:
        #     yield {'output': str(response)}
            # # 启动异步任务
            # task = asyncio.create_task(
            #     agent.arun(input=input_text.strip())
            # )
        
        # # 返回流式输出
        # async for chunk in callback_handler.generate():
        #     yield chunk
            
        # await task  # 确保任务完成

    async def generate_stream(self, agent_id: str, input_text: str) -> AsyncGenerator[str, None]:
        """真正的token级流式处理"""
        print(f"Starting true token streaming for agent {agent_id}")
        
        try:
            async for chunk in self.llm.astream(input_text):
                if hasattr(chunk, 'content'):
                    yield f"{chunk.content}"
                elif isinstance(chunk, str):
                    yield f"{chunk}"
                
        except Exception as e:
            print(f"Stream error: {str(e)}")
            yield f"data: [ERROR] {str(e)}\n\n"
        finally:
            yield "[DONE]\n\n"
