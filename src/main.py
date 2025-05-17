import asyncio
from .core.rag_system import EnhancedRAGSystem
import uvicorn
from .api.endpoints import app
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def main():
    print("Starting RAG system initialization...")
    # 初始化RAG系统
    rag_system = EnhancedRAGSystem('config/config.yaml')
    try:
        await rag_system.initialize()
        # await rag_system.initialize(load_documents=True)

        print("RAG system initialized successfully")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return
    
    # 示例问题
    questions = ["妩媚是谁？"]
    print(f"Processing {len(questions)} sample questions...")
    
    
    
    # 处理问题
    for question in questions:
        print(f"Question: {question} processing ...")
        try:
            print("\nAnswer:")
            full_answer = ""
            async for result in rag_system.process_query(question): 
                print(result, end='', flush=True)
                # print("\n")
                # print(f"Metrics: {rag_system.monitor.get_metrics()}\n")
                # print(f"- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
        except Exception as e:
            print(f"Error processing question: {str(e)}")

if __name__ == "__main__":
    # 运行示例
    # asyncio.run(main())
    
    # 或者启动API服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
