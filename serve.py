import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from multiagent import CrewHomeworkCorrection, CrewCodeStudio
import os

load_dotenv() 

def main():
    with st.sidebar:
        option = st.sidebar.selectbox(
            '選擇一個區塊',
            ('AI Question Solving System', 'GPT4o-ChatBox', 'Leetcode Solver', 'Thesis Helper')
        )
        
    if option == "AI Question Solving System":
        
        st.title("🦉Senior High Solving System")
        
        with st.sidebar:
            st.header("Enter your question and answer：")
            with st.form("my_form"):
                model = st.selectbox(
                    "Select model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "crewAI-llama3"])
                question = st.text_input(
                    "Enter your question：", placeholder="光的三原色是？ A. 紅、綠、藍 B. 紅、黃、藍 C. 紅、綠、黃 D. 紅、綠、黑")
                answer = st.text_input(
                    "Enter your answer：", placeholder="(A)")
                temperature = st.select_slider(
                    "Temperature", options=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], value=0)
                # top_k = st.slider(
                #     "Top K Similar Data", min_value=1, max_value=20, value=3)
                submitted = st.form_submit_button("Solve it!")
                
        if submitted:
            with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
                with st.container(height=500, border=False):
                    teachers = CrewHomeworkCorrection(model, question, answer, temperature)
                    result = teachers.run()
                status.update(
                    label="✅ 已經完成解答!",
                    state="complete", 
                    expanded=False
                )

            st.markdown(result)
            st.download_button(
                label="Download Report",
                data=result,
                file_name="report.md",
            )
    
    if option == "GPT4o-ChatBox":
        st.title("GPT-4o")  

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-4o"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
        
    if option == "Leetcode Solver":
        st.title("Leetcode Solver")  

        with st.sidebar:
            st.header("Enter your question and answer:")
            with st.form("my_form"):
                model = st.selectbox(
                    "Select model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "crewAI-llama3"])
                leetcode_question = st.text_input(
                    "Enter your leetcode problem:", placeholder="Leetcode 1. Two Sum: https://leetcode.com/problems/two-sum/")
                temperature = st.select_slider(
                    "Temperature", options=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], value=0)
                # top_k = st.slider(
                #     "Top K Similar Data", min_value=1, max_value=20, value=3)
                submitted = st.form_submit_button("Solve it!")
                
        if submitted:
            with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
                with st.container(height=500, border=False):
                    studio = CrewCodeStudio(model, leetcode_question, temperature)
                    result = studio.run()
                status.update(
                    label="✅ 已經完成解答!",
                    state="complete", 
                    expanded=False
                )
            st.markdown(result)
        
        
if __name__ == "__main__":
    main()