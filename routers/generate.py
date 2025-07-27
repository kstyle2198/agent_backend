import logging
from typing import List, Dict, Optional, Union, TypedDict, Annotated
from langchain_groq import ChatGroq
from pydantic import BaseModel
import operator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 로거 설정
logger = logging.getLogger("generate")

llm = ChatGroq(model_name="gemma2-9b-it", temperature=0,max_tokens=3000,) 
llm_think = ChatGroq(model_name="qwen/qwen3-32b", temperature=0,max_tokens=3000,) 


class OverAllState(TypedDict):
    """
    Represents the overall state of our graph.
    """
    question: str
    questions: Annotated[list, operator.add] # 질문 누적
    context: Annotated[list, operator.add] # 검색 결과 누적
    rerank_context: Annotated[list, operator.add] # 리랭크된 검색 결과 누적
    top_scores: Annotated[list, operator.add] # 리랭크 점수 누적
    top_k: Dict = {'doc':4, "web": 2}
    rerank_k: int = 3
    rerank_threshold: float = 0.01
    generations: Annotated[list, operator.add] # 응답 결과 누적


from typing import TypedDict, Annotated, List, Dict, Optional
from pydantic import BaseModel
import re
import json
import asyncio
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
import operator

## 현재 일반 어시스턴트 관점에서 작성됨..
SYSTEM_PROMPT = """You are a Smart AI Assistant. 
Use the following pieces of retrieved contexts or web search results to answer the question.
Think step by step and generate logical and reasonable answer in Korean Language.
Nonetheless, If you don't know the answer, just say that you don't know. 
Question: {question} 
Context: {context} 
Answer:"""

class GenerationRequest(BaseModel):
    question: str
    questions: Annotated[List[str], operator.add]
    documents: Annotated[List[Dict], operator.add]
    session_id: str

class GenerationResponse(BaseModel):
    answer: str

def format_docs(docs: List[str]) -> str:
    """Format documents for context"""
    return "\n\n".join(doc["page_content"] for doc in docs)

def build_question_history(state: OverAllState) -> str:
    """Build question history from state"""
    if not state.get('generations'):
        return state['question']
    
    history = []
    for q, a in zip(state['questions'][-4:], state['generations'][-4:]):
        a_without_think = re.sub(r"<think>.*?</think>", "", a, flags=re.DOTALL).strip()
        history.append(f"question: {q}, answer: {a_without_think}")
    
    history.append(f"final question is {state['question']}")
    return "\n".join(history)

async def generate_agent(state: OverAllState) -> Dict[str, List[str]]:
    """Generate response based on question and context"""
    logger.info(f"Start Generate Agent")
    try:
        question = build_question_history(state)
        context = format_docs(state['context'])
        
        prompt = ChatPromptTemplate.from_messages([("human", SYSTEM_PROMPT),])
        rag_chain = prompt | llm_think | StrOutputParser()
        
        generation = await rag_chain.ainvoke({"context": context, "question": question})
        logger.info(f"Generation completed. Generate {len(generation)} Characters.")
        return {"generations": [generation]}
    except Exception as e:
        logger.exception("Error occurred during generate agent")
        raise


def create_gen_graph() -> StateGraph:
    """Create and configure the generation graph"""
    rag_builder = StateGraph(OverAllState)
    rag_builder.add_node("generate_agent", generate_agent)
    
    rag_builder.add_edge(START, "generate_agent")
    rag_builder.add_edge("generate_agent", END)
    
    memory = MemorySaver()
    return rag_builder.compile(checkpointer=memory)

gen_graph = create_gen_graph()

def sse_format(payload: Dict) -> str:
    """Format data for SSE"""
    return f"data: {json.dumps(payload,  ensure_ascii=False)}\n\n"

async def generate_stream(state: Dict, thread_id: str):
    """Generate streaming response"""
    thinking_started = False
    
    async for msg, metadata in gen_graph.astream(state, stream_mode="messages", config={"thread_id": thread_id}):
        if metadata["langgraph_node"] != "generate_agent":
            continue

        if msg.content == "<think>":
            yield sse_format({
                "content": msg.content,
                "type": "thinking_start",
                "thinking": True
                })
            thinking_started = True
        elif msg.content == "</think>":
            yield sse_format({
                "content": msg.content,
                "type": "thinking_end",
                "thinking": False
                })
            thinking_started = False
        elif msg.content and thinking_started:
            yield sse_format({
                "content": msg.content,
                "type": "thinking",
                "thinking": True
                })
        elif msg.content and not thinking_started:
            yield sse_format({
                "content": msg.content,
                "type": "answer",
                "thinking": False
                })
        else: pass


from fastapi import APIRouter, HTTPException
generate = APIRouter(prefix="/generate")

@generate.post("/", response_model=GenerationResponse, tags=["Agent"])
async def generate_content(request: GenerationRequest):
    """Endpoint for generating responses"""
    logger.info(f"Generation API called for the query: {request.question}")
    try:
        thread_id = f"thread-{request.session_id}"
        
        state = {
            "question": request.question,
            "questions": request.questions,
            "documents": request.documents,
            }
        
        return StreamingResponse(generate_stream(state, thread_id), media_type="text/event-stream")
    except Exception as e:
        logger.error("Generation failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

