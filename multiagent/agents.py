import streamlit as st 
from crewai import Agent
from textwrap import dedent
from .tools import query_chroma
from langchain_openai import ChatOpenAI
from langchain_core.agents import AgentAction

def streamlit_callback(step_output):
    st.markdown("---")
    for step in step_output:
        if len(step) == 2:
            agent_action, context_from_agent = step
            if isinstance(agent_action, AgentAction):
                st.markdown(f"# Action")
                if agent_action.tool is not None:
                    st.markdown(f"## Tool")
                    st.markdown(f"{agent_action.tool}")
                if agent_action.tool_input is not None:
                    st.markdown(f"## Tool_input")
                    st.markdown(f"{agent_action.tool_input}")
                if agent_action.log is not None:
                    st.markdown(f"## Log")
                    st.markdown(f"{agent_action.log}")
            if isinstance(context_from_agent, str):
                st.markdown(f"# RAG Context")
                st.markdown(f"{context_from_agent}")   

class HomeworkCorrectionAgents:
    def __init__(self, temperature, model):
        self.temperature = temperature
        if model != "crewAI-llama3":
            self.llm = ChatOpenAI(
                model=model,
                temperature=self.temperature,
            )
        else:        
            self.llm = ChatOpenAI(
                model="crewAI-llama3",
                base_url="http://localhost:11434/v1",
                api_key="NA"
            )
    
    def program_manager_agent(self):
        return Agent(
            role='導師',
            goal='協調底下的教科書分析師、作業批改員和錯題本創建者，確保學生能有最佳的學習體驗',
            backstory=dedent("""
            作為導師，你的職責是監督整個教育集團，從開始到最終報告給學生的報告。你將與各個團隊成員協調，確保順暢的溝通，並確保專案符合學生的期望和要求。
            將學生問題分配交給教科書分析師，請它們分析學生問題，並依照高中教科書，指出問題和答案相關或是所屬的章節。
            將學生的問題批改交給作業批改員，請它們批改作業並彙整錯誤概念的解釋。
            將學生的錯誤概念給錯題整理者，請它們根據學生的錯誤概念搜尋相關的大學入學考試題目。
            """),
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            memory=True,
            max_iter=4
        )

    def textbook_analyst_agent(self):
        return Agent(
            role='教科書分析師',
            goal='分析學生問題，並依照教科書內容，指出相關章節',
            backstory=dedent("""
            作為教科書分析師，你的職責是收集和分析高中教科書的內容。你的分析將有助於確定學生的錯誤答案所屬的章節，
            並概述該章節需要的必要知識點和學習重點，並回傳給報告撰寫員整理訊息。
            """),
            tools=[query_chroma],
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            memory=True,
            max_iter=3
        )

    def homework_grader_agent(self):
        return Agent(
            role='作業批改員',
            goal='批改作業並彙整錯誤概念的解釋',
            backstory=dedent("""
            作為作業批改員，你的職責是批改學生的問題與答案，分析學生問題，並彙整他們錯誤概念的解釋，並回傳給報告撰寫員整理訊息。
            """),
            tools=[query_chroma],
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            memory=True,
            max_iter=5
        )

    def error_book_creator_agent(self):
        return Agent(
            role='錯題整理者',
            goal='根據學生的錯誤概念搜尋相關的考試、或是習作題目',
            backstory=dedent("""
            作為錯題整理者，你的職責是根據學生的錯誤概念搜尋相關的考試或是習作題目，並回傳給報告撰寫員整理訊息。
            """),
            tools=[query_chroma],
            verbose=True,
            memory=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            max_iter=3
        )
    
    def report_writer_agent(self):
        return Agent(
            role='報告撰寫員',
            goal='根據教科書分析師、作業批改員和錯題整理者的結果，撰寫最終報告',
            backstory=dedent("""
            作為報告撰寫員，你的職責是審查並整合教科書分析師、作業批改員和錯題整理者創建的結果。以他們的內容準備一份由 markdown 撰寫的報告，提供對學生作業的總體評估，教科書章節與概念，並包含相近錯誤題型。
            """),
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            allow_delegation=False,
            memory=True,
            # max_iter=4
        )
        
        
class CodeStudioAgents:
    def __init__(self, temperature, model):
        self.temperature = temperature
        if model != "crewAI-llama3":
            self.llm = ChatOpenAI(
                model=model,
                temperature=self.temperature,
            )
        else:        
            self.llm = ChatOpenAI(
                model="crewAI-llama3",
                base_url="http://localhost:11434/v1",
                api_key="NA"
            )
    
    def program_manager_agent(self):
        return Agent(
            role='管理員',
            goal='協調底下的問題分析師、程式設計師，確保解答能符合要求',
            backstory=dedent("""
            作為管理員，你的職責是監督整個集團，從開始到最終報告。你將與各個團隊成員協調，確保順暢的溝通，並確保專案符合問題的期望和要求。
            將問題分配交給問題分析師，請它們分析學生問題，並依照資料，整理出問題和答案相應的觀念。
            將問題分析師產生的問題描述與解題觀念交給程式設計師，請它們根據問題描述與觀念產生正確的程式碼。
            """),
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            memory=True,
            max_iter=3
        )

    def problem_analyst_agent(self):
        return Agent(
            role='問題分析師',
            goal='負責分析學生輸入的程式問題敘述，並深入研究問題的核心觀念和解決策略',
            backstory=dedent("""
            作為問題分析師，你的職責是將學生的程式問題敘述轉換為清晰的任務描述，同時深入分析問題的核心觀念和解決策略。
            你的分析將提供給程式設計師來撰寫程式，並回傳給報告撰寫員整理訊息。
            """),
            # tools=[query_chroma],
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            memory=True,
            max_iter=5
        )
    
    def program_design_agent(self):
        return Agent(
            role='程式設計師',
            goal='根據問題分析師提供的任務描述和核心觀念，設計和編寫程式碼，解決學生提出的程式問題',
            backstory=dedent("""
            作為程式設計師，你的職責是根問題分析師提供的任務描述與核心觀念，設計與撰寫程式碼，需要滿足問題指定的格式，並回傳給報告撰寫員整理訊息。
            """),
            # tools=[query_chroma],
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            memory=True,
            max_iter=5
        )
    
    def report_writer_agent(self):
        return Agent(
            role='報告撰寫員',
            goal='根據問題分析師、程式設計師的結果，撰寫最終報告',
            backstory=dedent("""
            作為報告撰寫員，你的職責是審查並整合問題分析師、程式設計師創建的結果。以他們的內容準備一份最終報告，提供對解題有幫助的觀念與正確的程式碼。
            """),
            verbose=True,
            llm=self.llm,
            step_callback=streamlit_callback,
            # allow_delegation=False,
            memory=True,
            max_iter=5
        )