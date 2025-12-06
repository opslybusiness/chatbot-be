import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import warnings
warnings.filterwarnings("ignore")
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",convert_system_message_to_human=True)

from chat_memory import PostgresChatMessageHistory

def get_session_history(session_id: str):
    return PostgresChatMessageHistory(session_id)


config = {"configurable": {"session_id": "firstchat"}}
model_with_memory=RunnableWithMessageHistory(model,get_session_history)
model_with_memory.invoke([HumanMessage(content="Hi! I'm ahmad")],config=config).content
config = {"configurable": {"session_id": "secondtchat"}}

model_with_memory.invoke([HumanMessage(content="what is my name?")],config=config).content
config = {"configurable": {"session_id": "firstchat"}}
model_with_memory.invoke([HumanMessage(content="what is my name?")],config=config).content


prompt=ChatPromptTemplate.from_messages(
    [
      ("system","You are a helpful customer support bot. Answer all questions in conversation style and not too long.",),
      MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | model
chain.invoke({"messages": ["hi! I'm bob"]})
chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]}).content
model_with_memory=RunnableWithMessageHistory(chain,get_session_history)
config = {"configurable": {"session_id": "thirdchat"}}
response=model_with_memory.invoke(
    [HumanMessage(content="Hi! I'm Jim"),
     ],config=config
)
print(response.content)

response=model_with_memory.invoke(
    [HumanMessage(content="hi what is my name"),
     ],config=config
)
print(response.content)


response=model_with_memory.invoke(
    [HumanMessage(content="what is my name?"),
     ],config=config
)
print(response.content)


trimmer = trim_messages(
    max_tokens=60,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no hadsgjdja sghdgahsbdja sdgashgdvasndjavhdh asdhavsdhas nd problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

model.get_num_tokens_from_messages(messages)
trimmer.invoke(messages)
trimmed_message = trimmer.invoke(messages)
model.get_num_tokens_from_messages(trimmed_message)

prompt=ChatPromptTemplate.from_messages(
    [
      ("system","You are a helpful assistant. Answer all questions to the best of your ability.",),
      MessagesPlaceholder(variable_name="messages"),
    ]
)


from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)

response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "English",
    }
)
response.content


response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what math problem did i ask")],
        "language": "English",
    }
)
response.content
 
