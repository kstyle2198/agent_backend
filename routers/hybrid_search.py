import uuid
import logging
import heapq
import operator
from FlagEmbedding import FlagReranker
from typing import List, Dict, Optional, Union, TypedDict, Annotated
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from langchain_ollama import OllamaEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END

from dotenv import load_dotenv
load_dotenv(override=True)

# 로거 설정
logger = logging.getLogger("hybrid_search")

reranking_model_path = "D:/LLMs/bge-reranker-v2-m3" # 실제 경로로 변경해야 합니다.
reranker = FlagReranker(model_name_or_path=reranking_model_path, 
                        use_fp16=True,
                        batch_size=512,
                        max_length=2048,
                        normalize=True)


def load_elastic_vectorstore(index_names: Union[str, List[str]]):
    logger.info(f"Load Elastic VectorStore")
    try:
        # 단일 문자열인 경우 리스트로 변환
        if isinstance(index_names, str):
            index_names = [index_names]
        
        vector_store = ElasticsearchStore(
            index_name=index_names, 
            embedding=OllamaEmbeddings(
                base_url="http://localhost:11434", 
                model="bge-m3:latest"
            ), 
            es_url="http://localhost:9200",
            es_user="Kstyle",
            es_password="12345",
        )
        return vector_store
    except Exception as e:
        logger.exception("Error occurred during Elastic VectorStore Loading")
        raise

index_names = ["ship_safety"]
vector_store = load_elastic_vectorstore(index_names=index_names)

def reranking(query: str, docs: list, min_score: float = 0.5, top_k: int = 3):
    """
    Reranks documents based on a query.
    """
    logger.info(f"Start ReRanking")
    try:
        inputs = [[query.lower(), doc["page_content"].lower()] for doc in docs]
        scores = reranker.compute_score(inputs)
        if not isinstance(scores, list):
            scores = [scores]
        print(f"---- original scores: {scores}")
        # Filter scores by threshold and keep index
        filtered_scores = [(score, idx) for idx, score in enumerate(scores) if score >= min_score]
        # Get top_k using heapq (more efficient than sorting full list)
        top_scores = heapq.nlargest(top_k, filtered_scores, key=lambda x: x[0])
        # Get document objects from top indices
        reranked_docs = [docs[idx] for _, idx in top_scores]
        logger.info(f"ReRanking completed. Found {len(reranked_docs)} results.")
        return top_scores, reranked_docs
    except Exception as e:
        logger.exception("Error occurred during ReRanking")
        raise


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


def retrieve_agent(state: OverAllState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, 
        that contains retrieved documents
    """
    logger.info(f"Start Retrieve Agent")
    try:
        question = state['question']
        top_k_doc = state['top_k']['doc']
        
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"fetch_k": 10, "k":top_k_doc},
            )
        documents = retriever.invoke(question)
        documents = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]

        logger.info(f"Retrieving Docs completed. Found {len(documents)} results.")
        # 검색 결과를 context에 추가
        return {"context": [documents]}
    except Exception as e:
        logger.exception("Error occurred during retrieve agent")
        raise

def websearch_agent(state: OverAllState):
    """ Retrieve docs from web search """
    logger.info(f"Start WebSearch Agent")
    try:
        question = state['question']
        top_k_web = state['top_k']['web']

        # Search
        print("---Web Search---")
        tavily_search = TavilySearch(
            max_results=top_k_web,
            include_answer=True,
            include_raw_content=True,
            include_images=True,)
        search_docs = tavily_search.invoke(question)
        contents = [f"{d['title']} \n {d['content']}" for d in search_docs['results']]
        metas = [d['url'] for d in search_docs['results']]

        documents = []
        for content, url in zip(contents, metas):
            documents.append({"page_content":content, "metadata":{'url':url}})

        logger.info(f"Web search completed. Found {len(documents)} results.")
        # 웹 검색 결과를 context에 추가
        return {"context": [documents]} 
    except Exception as e:
        logger.exception("Error occurred during web search agent")
        raise


def reranking_agent(state:OverAllState):
    """Rerank retrieved documents"""
    logger.info(f"Start Reranking Agent")
    try:
        question = state['question']
        context = state['context']
        rerank_k = state['rerank_k']
        rerank_threshold = state["rerank_threshold"]
        num_of_agents = len(list(state["top_k"].keys()))

        ## 1차원으로 검색 문서 merged : [[문석검색], [웹검색]] --> [문서검색, 웹검색]
        # 에이전트 개수만큼만 리랭킹 대상 문서로 지정 (여기서는 에이전트가 두개이고, 끝에서 2개까지 문서 대상 리랭킹)
        merged_context = [item for sublist in context[num_of_agents*-1:] for item in sublist] 
        print(f"---- merged_context: {len(merged_context)}")
        
        top_scores, documents = reranking(query=question, docs=merged_context, min_score = rerank_threshold, top_k= rerank_k)
        print(f"---- Reranking 문서개수: {len(documents)} / top_scores: {top_scores}")
        
        # 리랭크된 문서와 스코어를 각각의 리스트에 추가
        logger.info(f"Reranking Agent completed. Rerank {len(documents)} results.")
        return {"rerank_context": [documents], "top_scores": [top_scores]}
    except Exception as e:
        logger.exception("Error occurred during Reranking Agent")
        raise

def search_builder(state):
    rag_builder = StateGraph(state)
    rag_builder.add_node("websearch_agent", websearch_agent)
    rag_builder.add_node("retrieve_agent", retrieve_agent)
    rag_builder.add_node("reranking_agent", reranking_agent)

    rag_builder.add_edge(START, "websearch_agent")
    rag_builder.add_edge("websearch_agent", "reranking_agent")
    rag_builder.add_edge(START, "retrieve_agent") 
    rag_builder.add_edge("retrieve_agent", "reranking_agent")
    rag_builder.add_edge("reranking_agent", END) 

    memory = MemorySaver()
    return rag_builder.compile(checkpointer=memory)

search_graph = search_builder(OverAllState)


# Define request models
class QuestionRequest(BaseModel):
    question: str
    top_k: Dict = {'doc':3, 'web':2}
    rerank_k: Optional[int] = 3
    rerank_threshold: Optional[float] = 0.01
    session_id: str = None  # 클라이언트가 줄 수도 있고 안 줄 수도 있음

class SearchResponse(BaseModel):
    refined_question: str
    documents: List[Dict]
    scores: List[float]
    questions_history: List[str] # 누적된 질문 히스토리 추가
    search_results_history: List[List[Dict]] # 누적된 검색 결과 히스토리 추가 (각 검색 단계별 결과 리스트)
    reranked_results_history: List[List[Dict]] # 누적된 리랭크 결과 히스토리 추가 (각 리랭크 단계별 결과 리스트)
    rerank_scores_history: List[List[float]] # 누적된 리랭크 스코어 히스토리 추가 (각 리랭크 단계별 스코어 리스트)


hybrid_search = APIRouter(prefix="/hybrid_search")

@hybrid_search.post("/", response_model=SearchResponse, tags=["Agent"], operation_id="safety_search")
async def search_documents(request: QuestionRequest):
    """
    Endpoint to search documents based on a question.
    Performs retrieval from vector store and web search, then reranks results.
    """
    logger.info(f"Hybrid search API called")
    try:
        # ✅ session_id 자동 생성 또는 재사용
        session_id = request.session_id or str(uuid.uuid4())
        thread_id = f"thread-{session_id}"
        
        # Initialize state with default values and empty lists for accumulation
        state = {
            "question": request.question,
            "top_k": request.top_k,
            "rerank_k": request.rerank_k,
            "rerank_threshold": request.rerank_threshold,
            "questions": [], # 여기에 첫 질문이 refined_question에서 추가됨
            "context": [],
            "rerank_context": [],
            "top_scores": [],
            "generations": [], # /search에서는 사용하지 않지만, state 정의에 포함
        }
        
        # Execute the search graph
        # ainvoke는 최종 상태를 반환합니다.
        search_result = await search_graph.ainvoke(state, config={"thread_id": thread_id})
        
        # Prepare response
        documents = []
        # rerank_context는 리스트의 리스트이므로, 가장 마지막 리랭크 결과를 가져옵니다.
        if search_result["rerank_context"]:
            documents = search_result["rerank_context"][-1] 

        scores = []
        # top_scores도 리스트의 리스트이므로, 가장 마지막 리랭크 스코어를 가져옵니다.
        if search_result["top_scores"]:
            scores = [score[0] for score in search_result["top_scores"][-1]]
        
        return SearchResponse(
            refined_question = search_result["question"],
            documents=documents,
            scores=scores,
            questions_history=search_result["questions"], # 누적된 질문 히스토리
            search_results_history=search_result["context"], # 누적된 검색 결과 히스토리
            reranked_results_history=search_result["rerank_context"], # 누적된 리랭크 결과 히스토리
            rerank_scores_history=[[s[0] for s in score_list] for score_list in search_result["top_scores"]] # 누적된 리랭크 스코어 히스토리
        )
    
    except Exception as e:
        logger.error("Hybrid search failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")