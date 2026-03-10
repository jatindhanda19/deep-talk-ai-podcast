""" This module  defines the podcast generation workflow using a state-based graph
    It handles:
    1. Retrieving relevant content from the uploaded PDF
    2. Generating a natural podcast script using LLM
    3. Supporting different modes: Q&A, Auto , Debate

   we use the Langgraph Stategraph to define nodes and edges, making
   the workFlow easy
"""
# Library Import
import os
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph
from typing import Any, TypedDict
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

#Ensure the Groq API key is set
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY  is not set in your .env file")

#Podcast State Schema
#Define the structure of the data that flow through the graph
class PodcastState(TypedDict):
    question:str
    retriever: Any
    vectorstore: Any
    mode: str
    context : str
    script : str

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, api_key=GROQ_API_KEY)

#Node Function
def retrieve_node(state: PodcastState) -> dict:
    """ Retrieves relevant document chunks of text from Vectorstore
        Q&A mode : fetch only relevant chunks for the users's question
        Auto mode: fetch multiple key section for the document for full coverage
    """
    question = state.get("question","").strip()
    mode = state.get("mode", "Auto")

    retriever = state["retriever"]
    vectorstore = state.get("vectorstore")

    docs = []

    if  mode == "Q&A" and question:
        docs = retriever.invoke(question)
    else:
        if vectorstore is None:
            raise ValueError("Retriever does not have an associated vectorstore for Auto mode.")
        
        # Predefined queries to cover major section of document
        queries= [
                "introduction overview background",
                "main topics key concepts definitions",
                "important findings results details",
                "examples use cases applications",
                "conclusion summary implications"
            ]
        
        seen , docs = set(),[]
        for q in queries:
            for doc in vectorstore.similarity_search(q, k=4):
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    docs.append(doc) 
           
    # Combine all retrieved chunk into a single context string
    context = "\n\n".join([doc.page_content for doc in docs])
    print("DEBUG vectorstore:", vectorstore)

    return{"context":context}
    
def generate_node(state:PodcastState) -> dict:
    """Generates a podcast script using the LLM
       The fuction adapts to different modes:
       -Q&A: Host asks question, Expert answer strictly from document.
       -Auto: Generates a full podcast episode covering main themes
       -Debate: Produces a debate-style episode with two experts
    """
    mode = state["mode"]
    context = state["context"]
    question = state.get("question","").strip()

    if not context.strip():
        return {"script": "Error: No content was retrieved from the document."}

    # common rules : strictly use only document
    RULES = """
            IMPORTANT RULES:
            1. Use ONLY the document context provided below.
            2. Do NOT add outside knowledge or general facts.
            3. If the answer is not in the document, say "the document does not cover this."
            4. Every sentence must be traceable to the document context.
         """
    # Contruct Prompt based on mode
    # Q&A mode
    if mode == "Q&A":
        prompt = f""" You are producing a podcast episode.
                 {RULES}

        Context from the document:
        {context}
  
        The host asks the following question:
        {question}

        Create a natural, engaging podcast conversation following this format exactly:
        
        Host: asks the question clearly
        Expert: gives a direct answer using only the document context
        Host: asks a thoughtful follow-up based on the document
        Expert: explains more deeply using only the document context
        Host: wraps up with a summary question
        Expert: gives a concise closing answer from the document
        
        Write only dialogue. Do not add any preamble or metadata.Stick strictly to the document
"""
    # Auto mode
    elif mode== "Auto":
        prompt = f"""You are producing a podcast episode
        {RULES}

        Context from the document:
        {context}
      
        Generate a podcast episode discussing key themes from this document.

        follow this format exactly:
        Title: a catchy episode title based on the document's main theme
        Host: introduces the episode and topic using the document
        Expert: gives an overview of the subject using the document
        Host: asks about most important point in document
        Expert: explains with detail and example from the document
        Host: asks about real-world implication of this information
        Expert: discusses practical impact using evidence from the document
        Host: closes the episode with a summary and call to action based on the document

        Write only dialogue. Do not add any preamble or metadata.  Stick strictly to the document

        """

       # Debate Mode
    elif mode == "Debate":
         prompt = f""" You are producing a debate-style podcast episode.
            {RULES}

         Context from the document:
         {context}

         Create a debate-style podcast discussion based on this content.

         Follows this format exactly:
         Host: introduces the topic and two experts using the document
         Expert_A: presents the main argument supporting the core idea from the document
         Expert_B: challenges the idea with a counter-point that is also from the document
         Host: asks both to clarify a specific point from the document
         Expert_A: defends with evidence directly from the document
         Expert_B: provides a strong counter using another point from the document
         Host: summarizes both sides using only what the document says and closes

         Write only dialogue. Do not add any preamble or metadata.
     """
    else:
        raise ValueError(f"Unknown Podcast mode:'{mode}'.Choose from 'Q&A', 'Auto' , 'Debate'")
   
    # Invoke LLm
    response = llm.invoke(prompt)

    #Extract text from LLM response
    content = response.content
    if isinstance(content, list):
        script_text ="".join(block.get("text","")if isinstance(block, dict) else str(block) for block in content)
    else:
        script_text = str(content)

    script_text = script_text.strip()
    print(f"[DEBUG] Script type: {type(response.content).__name__}, length: {len(script_text)} chars")

    return {"script": script_text}
   

# Graph Builder
def build_graph():
    """ Creates the Langgraph StateGraph workflow
        -"retriever" node fetch document context
        -"generate" node produces the podcast script
    """

    graph = StateGraph(PodcastState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve","generate")

    return graph.compile()


if __name__ == "__main__":
    from rag_engine import build_vectorstore

    vectorstore = build_vectorstore("ERP PPT.pdf")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    graph = build_graph()

    result = graph.invoke({
        "question": "What is the main purpose of this document?",
        "retriever": retriever,
        "vectorstore": vectorstore,
        "mode": "Q&A",
        "context": "",
        "script": ""
    })

    print(result["script"])