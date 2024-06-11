from dotenv import load_dotenv
from crewai import Crew, Process
from multiagent.tasks import HomeworkCorrectionTasks, CodeStudioTasks
from multiagent.agents import HomeworkCorrectionAgents, CodeStudioAgents
from crewai_tools import WebsiteSearchTool
import streamlit as st

load_dotenv()

class CrewHomeworkCorrection:
    def __init__(self, model, student_question, student_answer, temperature):
        self.model = model
        self.output_placeholders = st.empty()
        self.student_question = student_question
        self.student_answer = student_answer
        self.temperature = temperature
            
    def run(self):
        # Create Tasks and Agents
        agents = HomeworkCorrectionAgents(self.temperature, self.model)
        tasks = HomeworkCorrectionTasks()
        
        # Create Agents
        self.report_writer_agent = agents.report_writer_agent()
        self.program_manager_agent = agents.program_manager_agent()
        self.homework_grader_agent = agents.homework_grader_agent()
        self.textbook_analyst_agent = agents.textbook_analyst_agent()
        self.error_book_creator_agent = agents.error_book_creator_agent()
        
        # Create Tasks
        self.project_initiation = tasks.project_initiation_task(
            self.program_manager_agent, 
            self.student_answer, 
            self.student_question)
        
        self.textbook_analysis = tasks.textbook_analysis_task(
            self.textbook_analyst_agent, 
            self.student_answer, 
            self.student_question)
        
        self.homework_grading = tasks.homework_grading_task(
            self.homework_grader_agent, 
            self.student_answer, 
            self.student_question)
        
        self.error_book_creation = tasks.error_book_creation_task(
            self.error_book_creator_agent, 
            self.student_answer, 
            self.student_question)
        
        self.final_report = tasks.final_report_task(
            self.report_writer_agent, 
            self.student_answer, 
            self.student_question)
        
        # Set Context
        self.error_book_creation.context = [
            self.textbook_analysis
        ]
        
        self.final_report.context = [
            self.textbook_analysis, 
            self.homework_grading, 
            self.error_book_creation
        ]
        
        # Create Crew responsible for Homework Correction
        crew = Crew(
            agents=[
                self.program_manager_agent,
                self.textbook_analyst_agent,
                self.homework_grader_agent,
                self.error_book_creator_agent,
                self.report_writer_agent
            ],
            tasks=[
                self.project_initiation,
                self.textbook_analysis,
                self.homework_grading,
                self.error_book_creation,
                self.final_report
            ],
            verbose=True,
            process = Process.sequential
        )

        result = crew.kickoff()
        self.output_placeholders.markdown(result)
        return result

class CrewCodeStudio:
    def __init__(self, model, leetcode_question, temperature):
        self.leetcode_question = leetcode_question
        self.output_placeholders = st.empty()
        self.model = model
        self.temperature = temperature
            
    def run(self):
        # Create Tasks and Agents
        agents = CodeStudioAgents(self.temperature, self.model)
        tasks = CodeStudioTasks()
        
        # Create Agents
        self.report_writer_agent = agents.report_writer_agent()
        self.program_manager_agent = agents.program_manager_agent()
        self.problem_analyst_agent = agents.problem_analyst_agent()
        self.program_design_agent = agents.program_design_agent()
        
        # Create Tasks
        self.project_initiation = tasks.project_initiation_task(
            self.program_manager_agent, 
            self.leetcode_question)
        
        self.problem_analysis = tasks.problem_analysis_task(
            self.problem_analyst_agent, 
            self.leetcode_question)
        
        self.program_design = tasks.program_design_task(
            self.program_design_agent, 
            self.leetcode_question)
        
        self.final_report = tasks.final_report_task(
            self.report_writer_agent, 
            self.leetcode_question)
        
        # Set Context
        self.program_design.context = [
            self.problem_analysis
        ]

        self.final_report.context = [
            self.problem_analysis,  
            self.program_design
        ]
        
        # Create Crew responsible for Code Generation
        crew = Crew(
            agents=[
                self.program_manager_agent,
                self.problem_analyst_agent,
                self.program_design_agent,
                self.report_writer_agent
            ],
            tasks=[
                self.project_initiation,
                self.problem_analysis,
                self.program_design,
                self.final_report
            ],
            verbose=True,
            process = Process.sequential
        )

        result = crew.kickoff()
        self.output_placeholders.markdown(result)
        return result


if __name__ == "__main__":
    crew_homework_correction = CrewHomeworkCorrection("光的三原色是? A. 紅、綠、藍 B. 紅、黃、藍 C. 紅、綠、黃 D. 紅、綠、黑", "(A)")
    crew_homework_correction.run()
