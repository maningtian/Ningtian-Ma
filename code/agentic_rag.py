import os
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langgraph.graph import END, StateGraph

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MODEL_ID = "meta/llama-3.1-70b-instruct"

class WebSearcher():
    def __init__(self, k):
        self.web_searcher = TavilySearchResults(k=k)
    
    def search(self, question):
        docs = self.web_searcher.invoke({'query': question})
        contents = ''
        urls = []
        for doc in docs:
            contents += doc['content'] + '\n'
            urls.append(doc['url'])
        
        return contents.strip(), urls

def build_rag_pipeline():
    docs = []
    folder_path = os.path.join(BASE_PATH, 'data/docs')
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                text = f.read()
                docs.append(Document(page_content=text, metadata={'source': filename}))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250,
        chunk_overlap=0,
    )
    doc_splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=NVIDIAEmbeddings(model='NV-Embed-QA'),
    )
    retriever = vectorstore.as_retriever()

    # ASSISTANT MODEL
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a personalized assistant for 
        financial advising and investment management question-answering tasks. Use the following context from 
        retrieved documents to formulate a detailed and accurate answer without preluding that you used the provided 
        context. Also, use the following statements from the user context to develop a deep understanding of the user's 
        mindset for tailoring relevant financial plans that match the user's investor personality. Be helpful and 
        confident and clearly state your answer. Refrain from saying anything along the lines of "Please note that 
        this is a general recommendation and not personalized investment advice. It's essential to consult with a 
        financial advisor or conduct your own research before making any investment decisions." If the question asks 
        about predicting a specific stock, at the end of your response, return a JSON with three keys. The first key 
        is `symbol` which is the trading symbol of the stock. The second key is `action` which is your choice of `buy`, 
        `hold`, or `sell`. The last key is 'days' which is an integer value representing the number of days to predict 
        this stock for. If the question did not ask about predicting a specific stock, return this JSON with "None" for 
        all values. Do not include any preamble or headline before or explanation after you return this JSON.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        User Context: {user_context}
        Retrieved Documents: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context", "user_context"], 
    )
    llm = ChatNVIDIA(model=MODEL_ID, temperature=0.5)
    global rag_chain  
    rag_chain = prompt | llm | StrOutputParser()

    # QUESTION ROUTER
    question_router_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the JSON with a single key 'datasource' and no preamble
        or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    llm_router = ChatNVIDIA(model=MODEL_ID, temperature=0)
    question_router = question_router_prompt | llm_router | JsonOutputParser()

    # RETRIEVAL GRADER
    retrieval_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. 
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: {document} 
        Here is the user question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    llm_grader = ChatNVIDIA(model=MODEL_ID, temperature=0)
    retrieval_grader = retrieval_grader_prompt | llm_grader | JsonOutputParser()

    # HALLUCINATION GRADER
    hallucination_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        {documents} 
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )
    hallucination_grader = hallucination_grader_prompt | llm_grader | JsonOutputParser()

    # ANSWER GRADER
    answer_grader_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        {generation} 
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    answer_grader = answer_grader_prompt | llm_grader | JsonOutputParser()

    # STATE
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
            urls: list of urls
            user_context: user context
        """
        question: str
        generation: str
        web_search: str
        documents: List[Document]
        urls: List[str]
        user_context: str

    # NODES
    def web_search(state):
        print("---WEB SEARCH---")
        question = state["question"]
        documents = state.get("documents", [])
        user_context = state.get("user_context")

        # Web search
        web_searcher = WebSearcher(k=3)
        contents, urls = web_searcher.search(question)
        web_results = Document(page_content=contents)
        documents.append(web_results)
        return {
            "documents": documents,
            "question": question,
            "urls": urls,
            "user_context": user_context
        }

    def retrieve(state):
        print("---RETRIEVE---")
        question = state["question"]
        user_context = state.get("user_context")

        # Retrieval
        documents = retriever.invoke(question)
        return {
            "documents": documents,
            "question": question,
            "user_context": user_context
        }

    def grade_documents(state):
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        user_context = state.get("user_context")

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score["score"]
            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
            "user_context": user_context
        }

    def generate(state):
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        urls = state.get("urls", [])
        user_context = state.get("user_context")  

        # RAG generation with user context
        generation = rag_chain.invoke({
            "context": documents,
            "question": question,
            "user_context": user_context  
        })
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "urls": urls,
            "user_context": user_context
        }

    # CONDITIONAL EDGE
    def route_question(state):
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = question_router.invoke({"question": question})
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "retrieve"

    def decide_to_generate(state):
        print("---ASSESS GRADED DOCUMENTS---")
        web_search = state.get("web_search", "No")
        if web_search == "Yes":
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "websearch"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def grade_generation_v_documents_and_question(state):
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        user_context = state.get("user_context")

        score = hallucination_grader.invoke({
            "documents": documents,
            "generation": generation
        })
        grade = score["score"]

        if grade.lower() == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({
                "question": question,
                "generation": generation
            })
            grade = score["score"]
            if grade.lower() == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    # Build the workflow
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("websearch", web_search)  
    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("grade_documents", grade_documents)  
    workflow.add_node("generate", generate)  

    # Build graph
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    return workflow

def ask(rag_agents, question, user_context):
    inputs = {"question": question, "user_context": user_context}
    failure_count = 0
    value = {}
    for out in rag_agents.stream(inputs):
        for key, val in out.items():
            value[key] = val
            if key == 'generate':
                failure_count += 1
            if failure_count > 2:
                return "I am sorry. I am having trouble with your request. Please try again.", [], 1
    return value["generation"], value.get("urls", []), 0