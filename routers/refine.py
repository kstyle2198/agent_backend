import logging
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import List, Dict, Optional, Union, TypedDict, Annotated

# 로거 설정
logger = logging.getLogger("refine")

llm = ChatGroq(model_name="gemma2-9b-it", temperature=0,max_tokens=3000,) 

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class RefineState(TypedDict):
    """
    """
    question: str
    chat_history: list
    refined_query: str  

def format_query(docs: list) -> str:
    """Format documents for context"""
    try:
        return "\n\n".join(doc["content"].strip().replace("\n", "") for doc in docs[-8:])
    except:
        return ""

from langchain_core.messages import HumanMessage
def refined_question(state:RefineState):
    question = state['question']
    chat_history = state['chat_history']
    format_chat_history = format_query(chat_history)

    logger.info(f"Refining the question: {question}")
    try:
        prompt = f"""아래 질문 내용이 chat_history와 관련이 있는 경우, chat_history를 참고하여 질문을 정제해 주세요.
        chat_history의 내용이 없거나, 내용이 질문과 상관이 없는 경우, 질문 자체의 표현만 정제해주세요.
        질문에 대해 답하지 말고, 표현만 정제해주세요.
        <question>
        {question}

        <chat_history>
        {format_chat_history}
        """
        refined_question_content = llm.invoke([HumanMessage(content=prompt)]).content
        print(f">>>The Original Question is {question}")
        print(f">>>The Refined Question is {refined_question_content}.")
        return {"refined_query": refined_question_content}
    except Exception as e:
        logger.exception("Error occurred during Refine")
        raise

def query_refiner(state):
    refine_builder = StateGraph(state)
    refine_builder.add_node("refine_agent", refined_question)

    refine_builder.add_edge(START, "refine_agent")
    refine_builder.add_edge("refine_agent", END) 
    return refine_builder.compile()

refine_graph = query_refiner(RefineState)

class RefineRequest(BaseModel):
    question: str
    chat_history: list = []

class RefineResponse(BaseModel):
    refined_query: str


from fastapi import APIRouter, HTTPException
refine = APIRouter(prefix="/refine")

@refine.post("/", response_model=RefineResponse, tags=["Agent"])
async def refine_question(req: RefineRequest):
    logger.info(f"Refine the query: {req.question}")
    try:
        # 초기 상태 구성
        state = {
            "question": req.question,
            "chat_history": req.chat_history
            }

        # 그래프 실행
        result = refine_graph.invoke(state)
        logger.info(f"Refining Query completed. Generate {len(result)} Characters.")
        return RefineResponse(refined_query=result["refined_query"])

    except Exception as e:
        logger.error("Query Refinery failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
