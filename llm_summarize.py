# llm_summarize.py
from typing import List, Dict

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .crawling import crawl_stores_in_threads


load_dotenv()  # .env에서 OPENAI_API_KEY 로드


# --- 1) 구조화된 Output 모델 ---
class ReviewExtraction(BaseModel):
    main_menu: List[str] = Field(
        ...,
        description="가게에서 많이 언급되는 대표 메뉴 키워드들 (예: 소금빵, 아메리카노, 고구마라떼)",
    )
    atmosphere: List[str] = Field(
        ...,
        description="가게 분위기, 경험, 매장 특징 키워드들 (예: 아늑한, 감성적인, 좌석이 넓은)",
    )
    recommended_for: List[str] = Field(
        ...,
        description="어떤 유형의 사람이 방문하면 좋은지 (예: 연인과 함께, 친구와 수다, 반려견과 함께)",
    )


# --- 2) LLM + 구조화 출력 준비 ---
base_model = ChatOpenAI(
    model_name="gpt-4o-mini",        # 빠르고 저렴한 모델
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.2,
)

model = base_model.with_structured_output(ReviewExtraction)


# --- 3) 프롬프트 ---
prompt = PromptTemplate.from_template(
    """
너는 한국어 네이버 블로그 리뷰를 분석해서
가게의 대표 메뉴, 분위기, 추천 대상을 키워드로만 뽑는 역할을 한다.

아래 리뷰 텍스트를 보고,
각 항목당 3~4개의 핵심 키워드를 한국어로만 추출해라.

- main_menu: 자주 언급되는 메뉴 이름
- atmosphere: 매장의 분위기/경험/특징
- recommended_for: 어떤 사람이 방문하면 좋을지 (ex. 연인, 친구, 반려견과 함께 등)

반드시 키워드 위주의 짧은 표현만 사용해라.

리뷰 텍스트:
----------------
{text}
----------------
"""
)

# --- 4) LCEL 체인 ---
summarize_chain = prompt | model


# --- 5) 단일 가게용 함수 ---
def extract_review_keywords(input_text: str) -> ReviewExtraction:
    if not input_text.strip():
        return ReviewExtraction(main_menu=[], atmosphere=[], recommended_for=[])
    result: ReviewExtraction = summarize_chain.invoke({"text": input_text})
    return result


# --- 6) 여러 가게 batch 처리 ---
def extract_review_keywords_batch(
    store_to_text: Dict[str, str]
) -> Dict[str, ReviewExtraction]:
    store_names = list(store_to_text.keys())
    inputs = [{"text": store_to_text[name]} for name in store_names]

    results: List[ReviewExtraction] = summarize_chain.batch(inputs)

    store_to_result: Dict[str, ReviewExtraction] = {}
    for name, res in zip(store_names, results):
        if isinstance(res, ReviewExtraction):
            store_to_result[name] = res
        else:
            store_to_result[name] = ReviewExtraction(
                main_menu=[], atmosphere=[], recommended_for=[]
            )

    return store_to_result


# --- 7) 전체 파이프라인: 3개 가게 크롤링 + 요약 ---
if __name__ == "__main__":
    stores = [
        "카피로우 일산밤리단길카페점",
        "몽키 일산 밤리단길카페 본점",
        "뒷북서재",
    ]

    # 1) 병렬 크롤링 (가게당 Chrome 하나, ThreadPool)
    store_to_text = crawl_stores_in_threads(stores, max_workers=3)

    # 2) LLM batch 요약
    print("\n=== LLM 요약 (batch) 시작 ===")
    batch_result = extract_review_keywords_batch(store_to_text)

    # 3) 결과 출력
    for store, info in batch_result.items():
        print(f"\n######## {store} ########")
        print("대표 메뉴:", info.main_menu)
        print("분위기:", info.atmosphere)
        print("추천 대상:", info.recommended_for)
