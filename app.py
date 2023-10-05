import os
from dotenv import load_dotenv
import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.chains.openai_functions import (
    create_extraction_chain
)

import pandas as pd


SYSTEM_MESSAGE_BP_ANALYST= "You are a skillful UX Researcher who can generate insightful User Journey Maps. Your journey maps are always backed by real data and user interviews."

PROMPT_BP_ANALYST = """
Your task is to generate a User Journey Map for a typical day of the worker in the format specified below for a given job description.

Instructions:
- For the User Journey Map, imagine the day of the worker, how it can go from the start and till the end.
- Don't copy the job description word for word, the job description is only given for the reference and context.

Format of the task flow:
Stage: <Stage name>
- Task: <Task #1>
- Task: <Task #2>

Job position: {job_title}

Job description:
###
{job_description}
###

The task flow in markdown format:
"""

SYSTEM_MESSAGE_USE_CASE_FINDER = "You are an AI consultant who helps companies find use cases for Generative AI, Large Language Models, and other GPTs in their processes to make business more effective and customers more productive."

PROMPT_USE_CASE_FINDER = """
Instructions:
- I'll give you a task flow of the process of one of the job roles in my company. 
- You need to focus on analyzing a specific task that I'll specify.
- Take into account the pain point related to the task if specified.
- Important to focus on Generative AI, LLMs and GPTs use cases specifically and not on AI and Machine Learning overall
- If you can't come up with a Generative AI, Large Language Models, and other GPTs use case for a particular task of the flow, find a subtask of the specified task that will be suitable for that and describe an AI use case for that.
- If there is no suitable subtask, indicate clearly that you can't find a use case.

Job position: {job_title}

The task flow:
###
{task_flow}
###

The specific task to focus on from the flow:
{task}

Pain point related to the task:
{pain_point}

Format of the response:
**Generative AI use case**: <name of the Generative AI, LLMs, GPTs use case>

**Description**: <description of the AI use case solution>

**Data required**: <data required for Generative AI, LLMs, GPTs use case, formatted as list>

**Explanation**: <explain in maximum of 2-3 sentences why you picked this particular Gen AI, LLMs, GPTs use case>

Don't use words like 'automated' or 'assistant' for Generative AI use case name.
Take a deep breath and think step-by-step about why you chose a particular use case before coming up with the result and how Gen AI, LLMs, GPTs can assist a human in making the decision.
Think thouroghly if there is really an AI use case.
"""

# **Generative AI use case**: <name of the Generative AI, LLMs, GPTs use case>


def generate_task_flow(job_title, job_description):
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(SYSTEM_MESSAGE_BP_ANALYST)
            ),
            HumanMessagePromptTemplate.from_template(PROMPT_BP_ANALYST),
        ]
    )

    model09_gpt4 = ChatOpenAI(temperature=0.9, model="gpt-4")
    model09 = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo")


    task_flow_chain = template | model09_gpt4

    response = task_flow_chain.invoke({"job_title": job_title, "job_description": job_description})

    task_flow = response.content

    return task_flow

def extract_task_flow(task_flow):
    json_schema = {
        "title": "Task flow",
        "description": "The tasks that a worker does during the day",
        "type": "object",
        "properties": {
            "stage": {
                "title": "Stage", 
                "description": "A broad categorization or phase of activities that represents a specific part or segment of a workflow or daily routine.", 
                "type": "string"
            },
            "task": {
                "title": "Task", 
                "description": "A specific activity or action that needs to be executed within its respective stage. It is more granular and provides explicit details about what needs to be done.", 
                "type": "string"
            },
            "pain_point": {
                "title": "Pain", 
                "description": "A specific activity or action that needs to be executed within its respective stage. It is more granular and provides explicit details about what needs to be done.", 
                "type": "string"
            }
        },
        "required": ["stage", "task"],
    }

    model_functions = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    chain = create_extraction_chain(json_schema, model_functions)

    r = chain.invoke(task_flow)

    task_flow_extracted = r['text']

    return task_flow_extracted


def find_ai_use_cases(task_flow_extracted, job_title, task_flow):
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=SYSTEM_MESSAGE_USE_CASE_FINDER
            ),
            HumanMessagePromptTemplate.from_template(PROMPT_USE_CASE_FINDER),
        ]
    )

    partial_prompt = template.partial(job_title=job_title, task_flow=task_flow);

    model05 = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")

    usecase_chain = partial_prompt | model05

    just_tasks = [{'task': item['task'], 'pain_point': item.get("pain_point", "No pain point")} for item in task_flow_extracted]

    response_batch = usecase_chain.batch(just_tasks)

    return response_batch

# Function to extract relevant information from AIMessage objects and convert to DataFrame
def aimessages_to_dataframe(ai_messages, extracted_json):
    data = []

    for idx, message in enumerate(ai_messages):
        # This will match the AIMessage to the respective stage and task based on index
        # stage = extracted_json[idx]['stage']
        task = extracted_json[idx]['task']

        use_case = message.content
        
        data.append([task, use_case])

    df = pd.DataFrame(data, columns=["Task", "AI Use Case"])

    return df


def map_aimessages_to_tasks(ai_messages, extracted_json):
    mapped_data = []

    for ai_message, task_data in zip(ai_messages, extracted_json):
        mapped_entry = {
            "Stage": task_data["stage"],
            "Task": task_data["task"],
            "AI Use Case": ai_message.content
        }
        mapped_data.append(mapped_entry)

    return mapped_data

def display_mapped_data(mapped_data):
    with st.container():
        col_header1, col_header2 = st.columns([2, 5])
        with col_header1:
            st.markdown("**Task**")
        with col_header2:
            st.markdown("**Generative AI Use Case**")
        add_grey_line()

    for entry in mapped_data:
        with st.container():
            col1, col2 = st.columns([2, 5])
            
            # Task column
            with col1:
                st.markdown(entry['Task'])
            
            # AI Use Case column
            with col2:
                st.markdown(entry['AI Use Case'])

        add_grey_line()

def add_grey_line():
    st.markdown("<hr style='border:1px solid #F0F0F0; margin-top:0px; margin-bottom:1rem;'>", unsafe_allow_html=True)




def app():
    st.title("Generative AI Use Case Finder")
    st.markdown("This app helps you find Generative AI use cases for your job title and tasks.")

    tab1, tab2 = st.tabs(["Direct Task Flow Input", "Job Title & Description"])

    with tab1:
        job_title = st.text_input("Job Title", key="job_title_flow1")
        
        task_flow = st.text_area("Tasks Map", height=200)

        # Button to generate task flow
        if st.button('Find AI Use Cases'):
            # Check if job title or description are empty
            if not job_title or not task_flow:
                st.error("Please enter both a task flow before proceeding.")
            else:
                st.markdown(f"### AI use cases for {job_title}")
                with st.spinner('Extracting tasks from tasks map'):
                    extracted_json = extract_task_flow(task_flow)
                    # st.markdown("### Extracted JSON:")
                    # with st.expander("See Extracted JSON"):
                    #     st.json(extracted_json)

                    # Convert JSON to DataFrame and display
                with st.spinner('Finding AI Use Cases'):
                    ai_usecases = find_ai_use_cases(extracted_json, job_title, task_flow)
                    mapped_tasks = map_aimessages_to_tasks(ai_usecases, extracted_json)
                    display_mapped_data(mapped_tasks)

    
    with tab2:
        # Input fields
        job_title = st.text_input("Job Title", key="job_title_flow2")
        job_description = st.text_area("Job Description", height=300)

        # Button to generate task flow
        if st.button('Generate task flow'):
            # Check if job title or description are empty
            if not job_title or not job_description:
                st.error("Please enter both a job title and description before generating task flow.")
            else:
                with st.spinner('Generating task flow...'):
                    task_flow = generate_task_flow(job_title, job_description)
                    st.markdown("### Generated Task Flow:")
                    with st.expander("See Generated Task Flow"):
                        st.write(task_flow)

                with st.spinner('Extracting JSON from task flow...'):
                    extracted_json = extract_task_flow(task_flow)
                    st.markdown("### Extracted JSON:")
                    with st.expander("See Extracted JSON"):
                        st.json(extracted_json)

                    # Convert JSON to DataFrame and display
                with st.spinner('Finding AI Use Cases'):
                    ai_usecases = find_ai_use_cases(extracted_json, job_title, task_flow)
                    df = aimessages_to_dataframe(ai_usecases, extracted_json)
                    st.dataframe(df)
        
   

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "AI Use Case Finder"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "ls__bba3daa7b1fe4d8db9c3f6e263e1db22"
    app()
