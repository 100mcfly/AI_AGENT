from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI


load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)


@tool
def add_task(task, dsc):
    """Add task to the to-do list. Use this when the user wants to add or create tasks"""
    todoist.add_task(content=task, description=dsc)

@tool
def show_tasks():
    """Show all tasks in the inbox folder of Todoist when the user wants to see the list or their tasks. """
    results_paginator = todoist.get_tasks()
    tasks = []
    for task_list in results_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks


tools = [add_task, show_tasks]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.3

)


system_prompt = ("""You are Zee Mariko-Sama, a helpful assistant. You will help the user add, complete and show tasks. "
                 If the user asks to show the list or tasks, print out tasks in a bullet list. """)

prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

# chain = prompt | llm | StrOutputParser()

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# response = chain.invoke({"input":user_prompt})

history = []
while True:
    user_prompt = input("Enter task, Description: ")
    response = agent_executor.invoke({"input": user_prompt, "history": history})
    print(response["output"])
    history.append(HumanMessage(content=user_prompt))
    history.append(AIMessage(content=response["output"]))

    if user_prompt == "exit":
        break