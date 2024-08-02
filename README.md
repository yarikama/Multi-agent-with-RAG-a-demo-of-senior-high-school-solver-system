# Multi-Agent AI System With RAG
View the Result: [https://youtu.be/KbyVo1ZbzXU](https://youtu.be/KbyVo1ZbzXU)
# Introduction

## Preface

In addition to the **Code** **Interview Solver (Leetcode)**, we have also utilized Multiagent and RAG to develop a **Paper Reading System (Arxiv Searching)** and a **High School Student Problem-Solving System**. However, to facilitate various experiments and present them in our report, we have chosen the **Code Interview Solver** as the benchmark and the main focus of our report. This is because, apart from the differences in the agent's character and the tools employed, the core techniques used are largely similar across all systems. Therefore, we will not elaborate on the other systems here; more information can be found on our [](https://github.com/yuhansun33/capstone_final_project_crewai)[**Github**](https://github.com/yarikama/CrewAI-RAG-----a-demo-of-leetcode-solver) and in the **Results** section.

## Motivation

An LLM may require us to manually assign multiple tasks to it step by step for it to achieve our desired goal. If we **combine the power of multiple LLMs**, perhaps they can help us complete many automated tasks and items that need judgment. We want to leverage this to create a **Code Interview Solver(LeetCode)**. On the other hand, we want to enhance the agent's ability to retrieve data and provide problem-related concepts and answers. Therefore, we use **RAG (Retrieval Augmented Generation)** to enable our agent to provide us with the source of the data when accessing relevant information, such as what data structure the problem needs, or what programming techniques we can attempt. This could provide users with a more complete experience.

# Related work

- CrewAI's Template 
[https://github.com/joaomdmoura/crewAI-examples](https://github.com/joaomdmoura/crewAI-examples)
- Streamlit also provides Templates
[https://streamlit.io/gallery](https://streamlit.io/gallery)
- Basic tutorial for RAG (using Llama)
Including loading, slicing, embedding, retrieval, and generation, etc.
[https://www.youtube.com/watch?v=2TJxpyO3ei4&t=836s](https://www.youtube.com/watch?v=2TJxpyO3ei4&t=836s)[https://python.langchain.com/v0.1/docs/use_cases/question_answering/](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)
- Creating a Callback Function for CrewAI and displaying it on Streamlit (partial reference)
[https://www.youtube.com/watch?v=nKG_kbQUDDE&t=112s](https://www.youtube.com/watch?v=nKG_kbQUDDE&t=112s)
- Build Local Llama3
[https://www.ollama.com/](https://www.ollama.com/)
- LangChain, an LLM toolkit
[https://www.langchain.com/](https://www.langchain.com/)

# Dataset

The materials here are used for RAG, we will talk about how we preprocess these materials at Main Approach Section. 

## Leetcode Solving System (Online)

[https://www.cnblogs.com/grandyang](https://www.cnblogs.com/grandyang)

[https://zxi.mytechroad.com/blog/](https://zxi.mytechroad.com/blog/)

[https://github.com/wisdompeak/LeetCode](https://github.com/wisdompeak/LeetCode)

[https://leetcode.ca/](https://leetcode.ca/)

[https://github.com/lzl124631x/LeetCode](https://github.com/lzl124631x/LeetCode)

[https://github.com/TheExplainthis/LeetCodeJourney?tab=readme-ov-file](https://github.com/TheExplainthis/LeetCodeJourney?tab=readme-ov-file)

[https://ithelp.ithome.com.tw/users/20119871/ironman/2210](https://ithelp.ithome.com.tw/users/20119871/ironman/2210)

### Textbooks For Algorithm & Data Sturcture:

ALGORITHMS INTRODUCTION TO THIRD EDITION

Data Structures and Algorithms

## Senior High Solving System

高中課本 From 授課橘

[https://teach-orange.com/](https://teach-orange.com/)

10 年份學測、10 年份指考

[https://www.ceec.edu.tw/](https://www.ceec.edu.tw/)

翰林10 年份學測、10 年份指考解析

# Baseline

We compare our multiagent system with a single LLM and also make comparisons with and without RAG (Retrieval-Augmented Generation).

# Evaluation Metric

## Our System (GPT4o with RAG) WIN!!

Leetcode 
Easy: 9 correct, 1 incorrect (runtime error)
Medium: 8 correct, 2 incorrect
Hard: 1 correct, 4 incorrect
It is observed that the incorrectly answered questions are those with less available information on the internet.

---

北區模擬考 112, 化學選擇題
Non-calculation related: 7 correct, 0 incorrect
Calculation-focused: 4 correct, 3 incorrect
It is hypothesized that questions emphasizing mathematical calculations are more prone to errors.

---

## Our System (GPT4o without RAG)

Leetcode 
Easy: 10 correct, 0 incorrect (win!!)
Medium: 7 correct, 3 incorrect (different questions compared to the ones with RAG)
Hard: 0 correct, 5 incorrect
It is observed that the incorrectly answered questions are those with less available information on the internet.

---

北區模擬考 112, 化學選擇題
Non-calculation related: 5 correct, 2 incorrect
Calculation-focused: 4 correct, 3 incorrect
It is hypothesized that questions emphasizing mathematical calculations are more prone to errors.

---

## Single LLM (GPT4o)

Leetcode 
Easy: 10 correct, 0 incorrect (win!!)
Medium: 6 correct, 4 incorrect
Hard: 0 correct, 5 incorrect

---

北區模擬考 112, 化學選擇題
Non-calculation related: 3 correct, 4 incorrect (no prompt engineering)
Calculation-focused: 2 correct, 5 incorrect

---

# Main Approach

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%201.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/7cfa58b1-4f2f-4b0f-935b-b26de4f6b617.png)

## Multi-Agent - Crewai

Overview

CrewAI is a tool that enables developers to define different agents, tasks, tools, and workflows:

- Agent: Developers can customize an agent's personality, purpose, language model, and other parameters. For example, a meticulous Unit Test engineer with the goal of finding errors in code, using the GPT-4o language model.
- Task: Developers can specify which agents participate in a task and what objectives they should achieve. For instance, debugging a piece of code.
- Tool: Developers can write their own functions and add descriptions such as arguments and return values. This allows agents to use these tools at appropriate times.
- Workflow: After defining tasks, agents, and tools, developers can use a workflow to package them together. This informs the agents about the process flow and the order of tasks.

### In Our Case

We designed roles by CrewAI such as manager, problem analyst, program designer, and report writer. Through dialogue replay, we can observe the communication and learning process between different agents, gaining insights into how LLMs formulate ideas, communicate with each other, and complete these tasks.

### Configuration

We use the OpenAI API key to connect to OpenAI's LLM, which serves as the LLM for this Multi-agent system. It can be configured to use models including `GPT-4`, `GPT-4-turbo`, `GPT-4o`, `GPT-3.5-turbo`, and `llama3:8b`. After our experiments, we ultimately decided to use **GPT-4o**. Detailed comparisons of the experimental performance will be explained in the following paragraphs.

![圖片2.png](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/4e310f43-a857-4699-82da-0c9f96fd6d3a.png)

### Communication and Steps:

Each LLM model plays a specific professional role, and the output of each step automatically becomes the input for the next step, forming an efficient and automated product analysis process.

---

**Step 1: Project Initiation**

- **Agent:** Manager
- **Task**: Initiate team projects by planning the work of other agents and fulfilling problem needs. Develop a project plan that includes analysis requirements and task assignments.
- **Output**: A detailed project plan with task assignments for each team member, instructing them to provide content that is helpful to users.

---

**Step 2: Problem Clarification**

- **Agent:** Problem Analyst
- **Task**: Review the origin problem, apprehending, and transform it into a clearer form with some usable hints to help the code generator simply design a requirement satisfying program, also making users understand the algorithm easily.
- **Output**: A clearer version of the origin problem and concepts the problem involving, which is beneficial for both program designer and users.

---

**Step 3: Code Generation**

- **Agent:** Program Designer
- **Task**: Generate the desired code fulfiling all requirements of the problem. Besides basic requirements, the code is also expected to perform well in time consuming, space utilizing or concisenes aspects.
- **Output**: A great code that solves the problem users encountered, which is the most important part for a code interview solver.

---

**Step 4: Report Generation**

- **Agent:** *Report Writer*
- **Task:** Review and integrate the results created by the problem analyst and program designer. Utilize their findings to compile a final report that offers a comprehensive analysis of the root problem, along with a variable programming code for user reference.
- **Output:** A comprehensive final report, written in markdown, after reviewing the agents above, and overall suggestions for user. The report should be concise while retaining important information.

## Retrieval Augmented Generation - LangChain, Chroma

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%202.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%203.png)

### Overview

RAG (Retrieval-Augmented Generation) is often employed when computational resources are limited or when the fine-tuned model is too large. By using RAG, we can ensure that when users make inquiries, we provide them with information from reliable sources. Moreover, in addition to relevant data, we also aim to offer users the specific information they truly need, rather than simply providing related content. 

Based on our observations, Multiagent systems can **effectively address this issue**, as agents can first identify the information that users genuinely require. For instance, in our high school student problem-solving system, the agent first determines the concept that the inquirer is asking about based on its own character. It then **breaks down the concept into smaller parts** using a top-down approach and searches for answers in the Vector Database. Finally, the agent consolidates the information. 

![圖片2.png](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/%25E5%259C%2596%25E7%2589%25872.png)

On the other hand, we have divided the data into three parts and applied them to different agents, namely: 

- Leetcode, algorithm, data structure, python document

 This approach prevents each agent from being unable to access its own specialized data.

### Indexing

**Load**
To accurately read data, ***LangChain*** provides loaders for different data formats, such as HTML, PDF, and Markdown. Initially, we used LangChain's PDF loader, but it did not provide a UTF-8 decoder, which caused our exam questions to become garbled. Later, we switched to `pdfplumber`, which yielded results without errors.

**Slice**
After loading the data, it needs to be sliced into individual chunks. The size of these chunks can be determined by the ourselves. In our case, we use `RecursiveCharacterTextSplitter` to slice the context into chunks and specify the `chunk_size` as **1000 tokens**, with a `chunk_overlap` of **100 tokens**. As a result, our questions are usually contained within a single, complete chunk.

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%204.png)

Ref: [https://python.langchain.com/v0.1/docs/use_cases/question_answering/](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)

**Embed**
Finally, to store these chunks in a vector database, we must first convert them into individual vectors, enabling the database to perform searches. Initially, we used Meta's Llama-embedding locally, but the results were poor. The data found by RAG often differed entirely from our expectations. Subsequently, we switched to OpenAI's `text-embedding-3-large`, and the retrieved data matched our expectations.

### Retrieval & Generation

**Retrieval**

The retrieval process also uses the embedding function mentioned earlier. First, the question is embedded into a vector, and then the vector database is searched for relevant data. It's important to note that the question may not always be the one provided by the student. As mentioned before, agents have the ability to identify the problem and break it down into smaller sub-questions for RAG. The k most similar data points found are then added to the prompt.

**Generation**

After the retrieval process, the LLM processes the modified prompt and provides the final answer. In our system, we allow the agents to decide the maximum number of data points (k) to retrieve. If no value is specified, the default value of k is set to 3-5. Additionally, in the callback, we can see what data RAG has retrieved, its source, and the question used for the retrieval.

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%205.png)

Ref: [https://python.langchain.com/v0.1/docs/use_cases/question_answering/](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)

## Graphic User Interface - Streamlit

### Overview

Streamlit is a library that facilitates the visualization of data science projects. It can also be used in conjunction with LangChain to create chatbots. We have attempted to use Streamlit to visualize our Multiagent System and provide features such as displaying callbacks, downloading Markdown files, and switching to other systems like the ***Code Problem Solver, Senior High Solving System, Paper Reading System***, ***GPT-4o Chatbot***, and more. This eliminates the need for users to operate solely through the Command Line Interface and allows them to directly connect to and use our server.

**Our Graphic Interface:**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%206.png)

### Callback Function

In CrewAI's callback, we define our own callback function and display its thought process, inter-agent communication, tools used during the process, and various other parameters in Markdown format on Streamlit.

### Configuration

We provide users with the ability to input their questions and, optionally, what they believe to be the correct answer. Additionally, we offer various settings, such as selecting the model, adjusting the temperature, choosing the k value, and more.

# Experiments

## Change The Embedding Function in RAG

*(Only tested in Senior High Solving System in case that Leetcode is hard to have semantic meaning… )*

**Embedding Evaluation Report**

This report aims to evaluate the performance of different embedding methods on the college entrance examination dataset. We used the following embedding methods:

1. OpenAI's text-embedding-3-large model
2. LlamaCpp's Embedding model
3. GPT4All's Embedding model

We used several sentences related to the college entrance examination as the evaluation dataset.

**Evaluation Methods**

- Semantic Similarity Test:
    - Calculate the cosine similarity between sentence pairs to evaluate the ability of the embedding to capture semantic similarity.
    - Expect sentence pairs with semantic similarity to have higher cosine similarity.
    - example
    
    ![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/cb130668-6177-457d-b840-2d857160615b.png)
    

- Vector Arithmetic Test:
    - Perform addition and subtraction operations on word vectors, such as "mathematics - algebra + geometry", to evaluate the ability of the embedding in semantic arithmetic.
    - Expect the arithmetic results to be semantically close to the expected meaning.
    - example
        
        ![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%207.png)
        

- Discrimination Ability Test:
    - Evaluate the ability of the embedding to distinguish similar but different concepts, such as "The Chinese subject includes reading comprehension and writing tests" and "The English subject includes reading comprehension and writing tests".
    - Expect the embedding to effectively discriminate between these concepts.
    - example
    
    ![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%208.png)
    

- Stability Test:
    - Perform embedding on the same word multiple times to evaluate the stability of the obtained vectors.
    - Expect the vectors obtained from multiple embeddings to be highly similar.
    Evaluation Results
    The following is an analysis of the results for each evaluation:
    - example
    
    ![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%209.png)
    
1. The embedding methods of OpenAI, and GPT4All perform exceptionally well on the college entrance examination dataset, with good performance in semantic similarity, vector arithmetic, downstream task performance, and discrimination ability.
2. LlamaCpp's embedding method performs relatively poorly on the college entrance examination dataset, with room for improvement in semantic capture, concept discrimination, and stability.
Based on the evaluation results, we recommend prioritizing the use of OpenAI or GPT4All's embedding methods for tasks related to the college entrance examination.

## LLM Model Configuration

Among the models used in CrewAI, we tried GPT-4, GPT-4o, GPT-3.5-turbo, and Llama3. The comparison metrics included running time and the structure and completeness of the generated reports. From the chart, we can see that GPT-3.5-turbo has the shortest running time, but its report performance is the worst. It can be observed that the process of communication and prompting between agents did not run very well, so we didn’t consider this model.

In addition, the running times of GPT-4o and Llama3 are close, and they are much faster than GPT-4. Among them, GPT-4o performs the best, and after multiple attempts, GPT-4o also has the most stable report performance, demonstrating that GPT-4o has better ability in mutual communication and collaborative prompting. Moreover, it is particularly stable in processing Chinese sentences. Therefore, we ultimately chose GPT-4o as our model.

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2010.png)

# Difficulties We Faced

## Excessive API Usage

In this test, we spent nearly **$90 USD (approximately NT$2,700) in tota**l, for three main reasons:

1. CrewAI's process is iterative. Each agent can repeatedly perform prompt engineering or request other agents to work on areas it deems insufficient until it is satisfied, before finishing the chain. As a result, the number of tokens used becomes very large.
2. Using RAG significantly increases the context size, consuming a considerable number of tokens. This is especially true since we have three agents, all utilizing RAG. Consequently, we reduced the value of k in the top-k retrieval to mitigate this issue.
3. The database embedding also requires API usage. As the amount of data being embedded grows, the cost increases linearly.

Solutions:

- Reduce the amount of data retrieved by RAG
- Set an upper limit on the number of iterations for each agent
- Introduce memory for agents to avoid repeated inquiries

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2011.png)

## Agent Workflow Convergence Issues

As mentioned earlier, agents will iteratively work on areas they are not satisfied with, especially **for less intelligent agents (llama3:8b).** This can exacerbate the issue of workflows not converging or increase the frequency of such occurrences. 

Fortunately, CrewAI allows users to set a **maximum number of iterations** (`max_iter`) and a **minimum response time** (`max_rpm`) for agents. However, this may result in slightly lower quality results (the best result from the current iteration is returned). For GPT-4o, though, the quality of each response is very good, so the difference is not significant.

It is important to note that since we ultimately use the "report writer" agent to output the final markdown report, it **cannot have limitations set**. In our tests, when limitations were placed on the report writer, it was highly likely that it would fail to provide the final result. However, setting limitations on other agents does not cause similar issues.

## Conflicts between CrewAI's Asynchronous Execution and Streamlit's Callback

To allow users to observe the thinking process and interactions between agents in our GUI, we used th**e Callback feature provided by the collaboration between CrewAI and LangChain,** and displayed the information using Streamlit. However, initially, we repeatedly encountered **thread errors,** spending a considerable amount of time debugging. Eventually, we discovered that setting CrewAI's `async_execute` to True caused the thread errors. 

This is because Streamlit only runs the server on the main thread, so when agents use other threads, the Callback function cannot send their messages (thoughts and interaction records) back to Streamlit, resulting in errors. Additionally, we found that async often led to errors. Ultimately, we chose to disable async in CrewAI. Although this significantly slows down the process, it provides more complete information and prevents unexpected errors.

## Context Window Too Small

Ref: [https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard](https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard)

Due to the reasons that multiagents perform prompt engineering with each other and use RAG, our system prompts **often become bulky.** As a result, there are times when the context window is not large enough to accommodate the prompts. To address this issue, we have adopted the same approach of reducing the amount of data extracted by RAG. Additionally, we **ensure that a single agent is not responsible for too many tasks or an excessive number of tools**. We also choose models with larger context windows, such as GPT-4o.

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2012.png)

## Embedding Method for RAG

Initially, to utilize a free, local embedding solution, we used`Llama-cpp`to call `llama3:8b` as the embedding method for RAG. However, the results were poor, and we had to set a very large value for k to find truly relevant data, which appeared later in the sequence. This led us to question whether our method was incorrect. However, after switching the embedding function to OpenAI's `text-embedding-large`, the issue was resolved.

# Results

<aside>
💡 ——
The Leetcode Solving Demo is on E3
You Can See Our Demo video For Senior High Question Solving System : [https://youtu.be/KbyVo1ZbzXU](https://youtu.be/KbyVo1ZbzXU)
——

</aside>

## GUI Overview

The image on the right shows the GUI interface we created using Streamlit. The functions that can be changed are shown in the three images below. First, you can select the type of problem you want to solve, such as the question solving system we used this time, or a chat box connected to GPT-4o, a code generator, a thesis helper, etc. You can also freely choose the LLM model to run. Finally, input the question you want to ask and the answer.

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2013.png)

**System Type**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2014.png)

**Model Type**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2015.png)

**Enter Question & Temperature**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2016.png)

**Running Log**

After entering the leetcode question, each agent will start to complete their respective tasks and coordinate with each other. The image below shows a detailed record, including what tools did they use, the mentor's task assignment, the communication process between agents, and the process of the LLM using RAG to obtain exam questions and their sources. All these agents' actions and coordination processes can be clearly displayed.

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2017.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2018.png)

**Task Assignment**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2019.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2020.png)

**Communication Process**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2021.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2022.png)

**Task Working Process Detail**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2023.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2024.png)

**RAG Sources**

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2025.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2026.png)

## Comprehensive Report

In the final report, we can see a comprehensive structured report, including the chapters and concepts covered in the student's question, grading results, and error question set, as shown in the screenshot here. Through the above concept organization, concept explanation, and error review process, students can further engage in comprehensive learning and review. Also, these informetion is generated in markdown file, it can be easily downloaded and import into Notion or other editor.

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2027.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2028.png)

![Untitled](Multi-Agent%20AI%20System%20With%20RAG%2092141a18f18447f39da8e2ded1b7f202/Untitled%2029.png)

# Analysis

Based on the results we have achieved, since the report will mainly focus on qualitative aspects, several factors will primarily affect the completeness of our report. First is the agent collaboration and task allocation. Our description of how each agent allocates tasks and coordinates with each other upon receiving the question and answer will directly affect the communication process between agents and the process of the LLM using RAG to obtain exam questions and sources. Therefore, we need to define the tasks of each agent very clearly and in detail, which is also the part we continuously test and modify. Moreover, the collaboration between agents and what information they exchange heavily relies on the performance of the LLM model. From the previous experiments, we know that different models do indeed affect the quality of communication between agents, which further influences the structure and completeness of the report. Therefore, these two aspects are also core parts of the results.

# Conclusion

In this project, we have collaborated among multiple LLMs using CrewAI. By assigning specific roles to each agent and allowing them to prompt each other, we have achieved better results than relying on a single model. The integration of RAG and vector databases has enhanced the accuracy and reliability of the generated reports by providing a reliable knowledge base. Through experimentation, we have identified GPT-4o as the most efficient and effective model for this application. The combination of multiple LLMs can also be adapted to various fields by modifying the agent roles, task settings, and the content of the RAG sources. 

For example, in academic research domain, the multi-agent LLM approach can be adapted to assist researchers and students in reading papers and synthesizing information. By assigning roles such as literature search assistant, paper summarizer, concept extractor, synthesis agent, and report generator, and integrating RAG with relevant databases, the system can help researchers and students navigate vast amounts of literature, generate comprehensive reports, and accelerate scientific discovery.

Therefore, the flexibility and adaptability of this multi-agent LLMs approach with CrewAI, and RAG, make it a powerful tool for solving complex problems across various domains. By carefully designing agent roles, task settings, and RAG content to suit the specific needs of each industry, this approach has the potential to revolutionize the way we work and make decisions in the future.

# What We Have Learned from the Project

Multiagent systems can be used to automate many tasks. When combined with RAG, they can provide data sources and offer agents longer-term memory capabilities without the need for fine-tuning data. This allows for the completion of automation projects at a low cost. In our project, we successfully implemented a Senior High Student Solving System and a Code Problem Solver, among others. It is important to provide precise instructions when setting up agents and tasks, and to be very mindful of API usage. Apart from that, the performance of Multiagent systems and RAG is excellent, akin to having a personal work studio. Beyond the examples in this project, they have other applications, such as web crawling while impersonating human users. The range of uses is vast, and the design process is very convenient. While smaller models have the ability to handle some basic tasks, they are prone to hallucinations and can easily mess up complex tasks.

# Future Work

Currently, we have implemented parts such as the Leetcode Solving System, Senior High Question Solving, and more. Multiagent has other applications that enhance automation efficiency, such as AI crawlers disguised as real people, Search The Web, automatic email draft reply robots, stock analysis, and so on. In addition to this, when building Multiagent, we can try to reduce the iterations of its discussions. The current methods for reducing iterations include adding Memory and setting a maximum iteration limit, which are relatively rigid approaches. We will see if CrewAI supports more capabilities that won't forcibly cut off conversations in the future.
For the RAG (Retrieval-Augmented Generation) part, we can make more variations, such as Self-RAG, which uses algorithms to evaluate the amount of RAG usage, allowing the Agent to assess how much data it should retrieve. Furthermore, we can use Web Search to enable the Agent to choose the most recent few pieces of data to search for, to prevent the Context from being too large and avoid including too much unnecessary information.
Regarding Streamlit, we can try the Asynchronous functionality, but this feature is still in the experimental stage, so we haven't implemented it yet. The process we use is Sequential to avoid thread errors.

# Code

[https://github.com/yarikama/CrewAI-RAG-----a-demo-of-leetcode-solver](https://github.com/yarikama/CrewAI-RAG-----a-demo-of-leetcode-solver)

# References

[https://arxiv.org/abs/2402.01680](https://arxiv.org/abs/2402.01680)

[https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

[https://arxiv.org/abs/2310.05421](https://arxiv.org/abs/2310.05421)

[https://www.langchain.com/](https://www.langchain.com/)

[https://www.crewai.com/](https://www.crewai.com/)

[https://streamlit.io/](https://streamlit.io/)

[https://www.langchain.com/langsmith](https://www.langchain.com/langsmith)
