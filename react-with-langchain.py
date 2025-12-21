import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI as _ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
import os

class ChatOpenAI(_ChatOpenAI):
    def __init__(self, model=os.getenv('CHINA_AI_MODEL'), temperature=0.1):
        m,t,k = model, temperature, os.getenv('CHINA_AI_AUTH_KEY')
        super().__init__(model=m, temperature=t, base_url=os.getenv('CHINA_AI_BASE_URL'), api_key=k)

llm = ChatOpenAI()
tools = load_tools(["ddg-search"])

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt_template = PromptTemplate.from_template(template)

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=False)

with st.sidebar:
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.title("💬 Chatbot")

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])