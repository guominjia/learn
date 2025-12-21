from langchain_openai import ChatOpenAI as _ChatOpenAI
import os

class ChatOpenAI(_ChatOpenAI):
    def __init__(self, model=os.getenv('CHINA_AI_MODEL'), temperature=0.1):
        m,t,k = model, temperature, os.getenv('CHINA_AI_AUTH_KEY')
        super().__init__(model=m, temperature=t, base_url=os.getenv('CHINA_AI_BASE_URL'), api_key=k)

import streamlit as st

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI()
tools = load_tools(["ddg-search"])
llm_with_tools = llm.bind_tools(tools)
chain = prompt | llm_with_tools

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory when the Chain is called.
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)