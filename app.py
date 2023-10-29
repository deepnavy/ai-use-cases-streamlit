import os
from dotenv import load_dotenv
import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from langchain.chains.openai_functions import (
    create_extraction_chain
)

from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from langsmith import Client

import pandas as pd

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["AI Use Case Finder"],
)

client = Client()

st.set_page_config(
    page_title="Generative AI Use Case Ideator",
    page_icon="🤖",
    menu_items={
        'Get Help': 'https://twitter.com/deepnavy',
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


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

PROMPT_USE_CASE_FINDER = """Instructions:
- I'll give you a task flow of the process of one of the job roles in my company. 
- You need to focus on analyzing a specific task that I'll specify.
- Take into account the pain point related to the task if specified.
- Important to focus on Generative AI, LLMs and GPTs use cases specifically and not on AI and Machine Learning overall.
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
#### <name of the Generative AI, LLMs, GPTs use case>

**Description**: <description of the AI use case solution>

**Data required**: <data required for Generative AI, LLMs, GPTs use case, formatted as list>

**Explanation**: <explain in maximum of 2-3 sentences why you picked this particular Gen AI, LLMs, GPTs use case>

Don't use words like 'automated' or 'assistant' for Generative AI use case name.
Take a deep breath and think step-by-step about why you chose a particular use case before coming up with the result and how Gen AI, LLMs, GPTs can assist a human in making the decision.
Think thouroghly if there is really an AI use case.
"""

demo_job_title = "Performance Fitness Coach"

demo_task_map = """Stage: Client onboarding

Task: Send a welcome email to the client
Pain point: We want to make it more personalized, but since the email sending process is manual, we end up spending a lot of time going through various surveys and notes related to the client before sending it.

Stage: Start of the program

Task: Prepare an exercise plan
Pain point: Even though we have a library of exercise plans, it's better for the coach to personally choose and modify them to suit the client's needs. They should review the client's information, which can be time-consuming.

Task: Set up first training session
Pain point: Even though we use calendly integrated into our app to schedule the time for sessions, sometimes coaches struggle with starting the conversation in the chat, which makes the communication very slow.

Stage: During the program

Task: Send a memo outlining the things to take into account during the next solo training sessions.
Pain point: Coaches spend a lot of time recalling what they said during the session and then preparing the email itself. Since they have many back-to-back sessions, creating these memos often leads to overtime.
"""

task_flow_help_text = """Task Map is a simplified version of a User Journey Map that only includes Stages, Tasks, and related Pain Points.
    
Don't worry about the specific structure; the system is intelligent enough to extract the elements from the Task Map provided in a free form."""

task_flow_placeholder_text = """Enter stages, tasks, and pain points for the specified role.

Example:
Stage: Client onboarding

Task: Send a welcome email to the client
Pain point: We want to make it more personalized, but since the email sending process is manual,
"""

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
                "type": "string",
                "enum": [
                    "Day",
                    "Morning",
                    "Afternoon",
                    "Evening",
                    "Night",
                    "Planning",
                    "Organizing",
                    "Working",
                    "Finishing",
                    "Reviewing",
                    "Meeting",
                    "Awareness",
                    "Consideration",
                    "Decision",
                    "Retention",
                    "Advocacy",
                    "Onboarding"
                ]
            },
            "task": {
                "title": "Task", 
                "description": "A specific activity or action that needs to be executed. It is more granular and provides explicit details about what needs to be done.", 
                "type": "string"
            },
            "pain_point": {
                "title": "Pain", 
                "description": "A challenge or difficulty that the worker encounters while executing the task or during the workflow. It highlights areas that need attention or improvement.", 
                "type": "string"
            }
        },
        "required": ["task"],
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

    model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    usecase_chain = partial_prompt | model

    just_tasks = [{'task': item['task'], 'pain_point': item.get("pain_point", "Not identified")} for item in task_flow_extracted]

    response_batch = usecase_chain.batch(just_tasks, config=runnable_config)

    st.session_state['run_ids'] = [str(run.id) for run in run_collector.traced_runs]

    return response_batch


def map_aimessages_to_tasks(ai_messages, extracted_json):
    mapped_data = []

    for ai_message, task_data in zip(ai_messages, extracted_json):
        mapped_entry = {
            # "Stage": task_data["stage"],
            "Task": task_data["task"],
            "AI Use Case": ai_message.content
        }
        mapped_data.append(mapped_entry)

    return mapped_data

def send_feedback():
    for run_id in st.session_state.run_ids:
        if st.session_state[run_id]:
            client.create_feedback(run_id, "user_score", score=True)
    
    st.session_state.run_after_submit = True
    

def display_mapped_data(mapped_data):
    with st.form(key="submit_form", clear_on_submit=False):
        with st.container():
            col_header1, col_header2, col_header3 = st.columns([2, 5, 1])
            with col_header1:
                st.markdown("**Task**")
            with col_header2:
                st.markdown("**Generative AI Use Case**")
            with col_header3:
                st.markdown("**Rate**")
            add_grey_line()

        run_ids = st.session_state['run_ids']

        for index, entry in enumerate(mapped_data):
            with st.container():
                col1, col2, col3 = st.columns([2, 5, 1])
                
                # Task column
                with col1:
                    st.markdown(entry['Task'])
                
                # AI Use Case column
                with col2:
                    st.markdown(entry['AI Use Case'])

                with col3:
                    st.checkbox('👍', key=run_ids[index])

            add_grey_line()
    
        st.form_submit_button(label="Submit ratings", type="primary", on_click=send_feedback, use_container_width=True)

    # st.write(mapped_data)

def display_mapped_data_after_submit(mapped_data, job_title):

    st.markdown(f"### AI use cases for {job_title}")

    with st.form(key="fake_form", clear_on_submit=False):
        with st.container():
            col_header1, col_header2 = st.columns([2, 5])
            with col_header1:
                st.markdown("**Task**")
            with col_header2:
                st.markdown("**Generative AI Use Case**")
            add_grey_line()

        for index, entry in enumerate(mapped_data):
            with st.container():
                col1, col2 = st.columns([2, 5])
                
                # Task column
                with col1:
                    st.markdown(entry['Task'])
                
                # AI Use Case column
                with col2:
                    st.markdown(entry['AI Use Case'])

            add_grey_line()

        st.success('Thank you for submitting the ratings!', icon="✅")
        
        st.form_submit_button(label="Submit ratings", type="primary", use_container_width=True, disabled=True)




def add_grey_line():
    st.markdown("<hr style='border:1px solid #F0F0F0; margin-top:0px; margin-bottom:1rem;'>", unsafe_allow_html=True)


def app():

    st.markdown("""
        <style>
            h4 {
                padding-top: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)


    st.session_state['run_ids'] = []

    if 'demo_job_title' not in st.session_state:
        st.session_state.demo_job_title = ""

    if 'demo_task_map' not in st.session_state:   
        st.session_state.demo_task_map = ""

    if 'mapped_tasks' not in st.session_state:   
        st.session_state.mapped_tasks = []

    if 'run_after_submit' not in st.session_state:   
        st.session_state.run_after_submit = False
    
    st.title("Generative AI Use Case Ideator")

    with st.container():

        # Split the container into two columns
        col1, col2 = st.columns([3, 1])

        # Place the input and textarea in the first column
        with col1:
            st.markdown("This app can help you brainstorm how your business can be automated with Generative AI by providing a Role and a Task Map for this role.")
            st.caption("Need some inspiration or guidance? Click 'Use example data' to get started.")


        # Place the button in the second column
        with col2:
            if st.button("Use example data"):
                st.session_state.demo_job_title = demo_job_title
                st.session_state.demo_task_map = demo_task_map
                st.rerun()

    # with tab1:
    job_title = st.text_input("Role", value=st.session_state.demo_job_title, key="job_title_flow1", placeholder="Enter a role")
    
    task_flow = st.text_area("Tasks Map", value=st.session_state.demo_task_map, height=200, help=task_flow_help_text, placeholder=task_flow_placeholder_text)

    # if 'mapped_tasks' in st.session_state and st.session_state.mapped_tasks:
    #     display_mapped_data(st.session_state.mapped_tasks)

    # st.button('Find AI Use Cases', type="primary", use_container_width=True, on_click=find_ai_use_cases_flow, args=(job_title, task_flow))
    
    # Button to generate task flow
    if st.button('Find AI Use Cases', type="primary", use_container_width=True):
        st.session_state.run_after_submit = False
        st.session_state.mapped_tasks = []
        # Check if job title or description are empty
        if not job_title or not task_flow:
            st.error("Please enter both a task flow before proceeding.")
        else:
            st.markdown(f"### AI use cases for {job_title}")
            st.caption("Help us improve response quality by checking relevant use cases and hitting 'Submit ratings' at the bottom of the form.")
            with st.spinner('Extracting tasks from tasks map'):
                extracted_json = extract_task_flow(task_flow)
                # with st.expander("See Extracted JSON"):
                #     st.json(extracted_json)

                # Convert JSON to DataFrame and display
            with st.spinner('Finding AI Use Cases'):
                ai_usecases = find_ai_use_cases(extracted_json, job_title, task_flow)
                mapped_tasks = map_aimessages_to_tasks(ai_usecases, extracted_json)
                display_mapped_data(mapped_tasks)
                st.session_state.mapped_tasks = mapped_tasks

    if 'mapped_tasks' in st.session_state and st.session_state.mapped_tasks and st.session_state.run_after_submit:
        display_mapped_data_after_submit(st.session_state.mapped_tasks, job_title)




   

if __name__ == "__main__":
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "AI Use Case Finder"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "ls__bba3daa7b1fe4d8db9c3f6e263e1db22"
    app()
