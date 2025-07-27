import logging
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# 로거 설정
logger = logging.getLogger("wiki_search")

def parse_paper_info(text: str) -> dict:
    lines = text.strip().split('\n')
    result = {}
    current_key = None
    summary_lines = []

    for line in lines:
        if line.startswith('Page:'):
            result['Page'] = line[len('Page:'):].strip()
            current_key = None
        elif line.startswith('Summary:'):
            current_key = 'Summary'
            summary_lines.append(line[len('Summary:'):].strip())
        elif current_key == 'Summary':
            summary_lines.append(line.strip())

    if summary_lines:
        result['Summary'] = ' '.join(summary_lines)

    return result


def do_wiki_search(question:str):
    logger.info(f"Performing wiki search for question: {question}")
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="ko"))
    wiki_results = wikipedia.run(question)
    wiki_results = wiki_results.split("\n\n")
    wiki_results = [wiki_result for wiki_result in wiki_results if wiki_result != '']
    refined_results = []
    for d in wiki_results:
        refined_d = parse_paper_info(d)
        refined_results.append(refined_d)
    logger.info(f"Wiki search completed. Found {len(refined_results)} results.")
    return refined_results


from pydantic import BaseModel
class SearchRequest(BaseModel):
    query: str


from fastapi import APIRouter, HTTPException
wiki_search = APIRouter(prefix="/wiki_search")

@wiki_search.post("/", tags=["Search"], operation_id="wiki_search")
def do_search(request: SearchRequest):
    logger.info(f"Wiki search API called with query: {request.query}")
    try:
        results = do_wiki_search(request.query)
        return {"results": results}
    except Exception as e:
        logger.error("Wiki search failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



