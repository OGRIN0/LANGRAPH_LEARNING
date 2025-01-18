from pydantic import BaseModel, Field  
from typing import Annotated, Literal, Sequence
from langchain import hub 
from langchain_core.messages import BaseMessage, HumanMessage 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import PromptTemplate 
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.simple import Tool
from langgraph.graph import END, StateGraph, START 
from langgraph.prebuilt import ToolNode 
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model_name="Gemma2-9b-It")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

try:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
except Exception as e:
    print(f"Error in document loading: {e}")
    docs_list = []

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=5)
doc_splits = text_splitter.split_documents(docs_list)

VectorStore = Chroma.from_documents(documents=doc_splits, collection_name="rag-chroma", embedding=embeddings)
retriever = VectorStore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retriever_blog_posts",
    "Search and return information about Lilian weng blog posts on LLM agents, prompt engineering and LLM Fine tuning"
)

tools = [retriever_tool]

class grade(BaseModel):
    binary_score: Literal["yes", "no"]  # Proper typing for the grading score

class AgentState:
    messages: Annotated[Sequence[BaseMessage], add_messages]

def AI_Assistant(state: AgentState):
    print("--CALL AGENT--")
    messages = state['messages']
    llm_with_tool = llm.bind_tools(tools)
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

def rewrite(state: AgentState):
    print("--TRANSFORM QUERY--")
    messages = state['messages']
    question = messages[0].content 
    msg = [
        HumanMessage(content=f""" \n
        Look at the input try to reason about the underlying semantic intent/meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question:""")
    ]
    response = llm.invoke(msg)
    return {"messages": [response]} 

def generate(state: AgentState):
    try:
        print("--GENERATE--")

        if 'messages' not in state or not isinstance(state['messages'], list) or not state['messages']:
            raise ValueError("Invalid 'state': 'messages' key is missing or not properly structured.")

        messages = state['messages']
        question = getattr(messages[0], 'content', None)
        if question is None:
            raise ValueError("First message does not have a 'content' attribute.")

        last_message = messages[-1]
        docs = getattr(last_message, 'content', None)
        if docs is None:
            raise ValueError("Last message does not have a 'content' attribute.")

        if 'hub' not in globals():
            raise NameError("'hub' is not defined.")
        if 'llm' not in globals():
            raise NameError("'llm' is not defined.")
        
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | llm | StrOutputParser()

        response = rag_chain.invoke({"context": docs, "question": question})
        
        if not isinstance(response, dict) or 'messages' not in response:
            raise ValueError("Invalid response structure from 'rag_chain.invoke'.")

        return {"messages": [response]}

    except Exception as e:
        print(f"Error in generate function: {e}")
        return {"error": str(e)}

def grade_documents(state: AgentState):
    llm_with_structure_op = llm.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing the relevance of a retrieved document to the user's question. 
        Here is the retrieved document: {context}
        Here is the user's question: {question}
        If the document contains keyword(s) or semantic meaning related to the user's question, grade it. 
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the user's question.""",
        input_variables=["context", "question"]
    )

    chain = prompt | llm_with_structure_op 

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content 
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score
    
    if score == "yes":
        print("--DECISION: DOCS RELEVANT--")
        return "generator"
    else:
        print("--DECISION: DOCS NOT RELEVANT--")
        return "rewriter"

workflow = StateGraph(AgentState)
workflow.add_node("ai_assistant", AI_Assistant)
retrive = ToolNode([retriever_tool])
workflow.add_node("retriver", retrive)
workflow.add_node("rewriter", rewrite)
workflow.add_node("generator", generate)

def tools_condition(state):
    if "tools" in state.get("ai_assistant", {}):
        return "tools"
    return END

workflow.add_edge(START, "ai_assistant")
workflow.add_conditional_edges(
    "ai_assistant", 
    tools_condition,  
    {
        "tools": "retriver",
        END: END,
    }
)

existing_branches = workflow.branches.get("retriver", {})
if "grade_documents" not in existing_branches:
    workflow.add_conditional_edges(
        "retriver",
        grade_documents,
        {"rewriter": "rewriter", "generator": "generator"}
    )
else:
    print("Branch 'grade_documents' already exists for the node 'retriver'.")

workflow.add_edge("generator", END)
workflow.add_edge("rewriter", "ai_assistant")

app = workflow.compile()

app.invoke({"messages":["LLM extends beyond generating"]})
