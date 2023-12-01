import asyncio
import getpass
import logging
import os
from datetime import datetime
from typing import Optional, AsyncGenerator, Any

from fastapi import FastAPI
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streaming_aiter_final_only import AsyncFinalIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores import VectorStore
from langchain.vectorstores.chroma import Chroma
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from file_vector import ingest_docs
from zhipuaiEmbed import ZhipuAiEmbeddings

app = FastAPI()
# openai向量数据库
openai_vectorstore: Optional[VectorStore] = None
# zhipuai向量数据库
zhipuai_vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    """程序启动后需要初始化的逻辑:向量初始化等"""
    logging.info("loading vectorstore")

    print('on start up ....')
    os.environ["OPENAI_API_KEY"] = 'your openai key'
    os.environ['ZHIPUAI_API_KEY'] = 'your zhipu ai key'

    # 将本地文档向量化
    ingest_docs(collection='openai_collection', llm_type='openai')
    ingest_docs(collection='zhipuai_collection', llm_type='zhipuai')

    openai_embeddings = OpenAIEmbeddings()
    zhipuai_embeddings = ZhipuAiEmbeddings()
    # 定义向量库
    global openai_vectorstore
    openai_vectorstore = Chroma(persist_directory='./assets/vector_index/openai',
                                embedding_function=openai_embeddings)

    global zhipuai_vectorstore
    zhipuai_vectorstore = Chroma(persist_directory='./assets/vector_index/zhipuai',
                                 embedding_function=zhipuai_embeddings)

    print(f'init vectorstore success')


class MyRequest(BaseModel):
    question: str
    chat_history: []
    scene: Optional[str]
    llm_type: Optional[str] = 'openai'


@app.post("/chat/sse")
async def sse_http(params: MyRequest):
    return EventSourceResponse(respx(params.question, params.chat_history))


async def respx(question: str, chat_history: []) -> AsyncGenerator[str, None]:

    handler = AsyncFinalIteratorCallbackHandler(answer_prefix_tokens=["AI", ":"])
    llm = ChatOpenAI(streaming=True, model_name='gpt-4', callbacks=[handler],
                     temperature=0)

    retriever = openai_vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever=retriever, name="搜索", description="当用户咨询[某某app]的使用方法,某某app的各种问题时使用本工具."
    )

    tools = [OrderSearch(), ExpressChange(), retriever_tool]

    agent_keyword = {'prefix': "我想让你扮演一个'某某'app的智能AI助理，用中文回答用户的提问，其中类似于DJXXXX是订单号，"
                               "用户和ai的聊天记录如下:{chat_history}"}

    agent_executor = initialize_agent(
        tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, agent_keyword=agent_keyword
    )

    run = asyncio.create_task(agent_executor.arun(
        {"input": "用户提问：" + question, "chat_history": chat_history}))
    async for token in handler.aiter():
        print(token, end='')
        yield token
    await run


class OrderSearch(BaseTool):
    name = "订单查询"
    description = "查询订单信息，入参是订单编号，返回值是订单信息,如果用户没有提供订单信息,清先告诉用户提供订单号"

    def _run(
            self,
            order_code: str,
    ) -> str:
        print('=======order_code=', order_code)
        return '姓名：张三，年龄：20，性别：男，订单状态：已发货'


class ExpressChange(BaseTool):
    name = "物流修改"
    description = "物流修改，入参是订单编号order_code和新的物流地址new_address组成的字符串" \
                  "，返回值是物流修改的客服工单号，使用本工具前请确保获取到了订单编号和物流地址，如果有缺失请先告诉用户提供必要信息"

    def _run(
            self,
            content: Any,
    ) -> str:
        print('物流修改input:', content)
        return 'x999999'


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


if __name__ == "__main__":
    import uvicorn

    port = 80
    uvicorn.run(app, port=port)
