"""
This module evaluates the quality of the generated podcast script
using RAGAS framework

RAGAS helps measure how well the Retrieval-Augmented Generation(RAG)
pipeline performs by analyzing:

1.Faithfulness - Whether the generated answer stays grounded in the retrived context
2.Answer Relevancy - Whether the generated answer actually addresses the question

These metrics help us understand how reliable the podcast generation pipeline
is when working with document-based knowledge.
"""
import logging
import os
import warnings
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

#Load environment variable
load_dotenv()

logger = logging.getLogger(__name__)

#Suppress noisy deprecation warnings from libraries
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Get Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#LLM used by RAGAS to judge answer quqlity
_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
)

#Embedding model used for answer relevancy comaparison
_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

#Attach models to RAGAS metrics
faithfulness.llm    = _llm
answer_relevancy.llm = _llm
answer_relevancy.embeddings = _embeddings


def evaluate_rag(question: str, script: str, context: str) -> dict:
    """
    Evaluates the podcast script using RAGAS.
    The function checks:
    -How faithful the generated script is to the source document
    -How relevant the script is to the user question
    """
    try:
        #If the user didn't ask a question (Auto mode)
        #use a generic evaluation question instead
        effective_question = (
           question.strip() if question and question.strip()
           else "What are the main themes of the document?"
        )
        logger.info("[EVAL] question=%s | script=%d chars | context=%d chars",
                    effective_question[:60], len(script), len(context))
        
        # Prepare data in the format required by RAGAS
        data = {
            "question":  [effective_question],
            "answer":    [script],
            "contexts":  [[context]],
        }

        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm = _llm,
        )

        # Convert result to a simple dictionary
        scores = result.to_pandas().to_dict(orient="records")[0]
        return {
          "faithfulness":     float(scores.get("faithfulness",     0) or 0),
          "answer_relevancy": float(scores.get("answer_relevancy", 0) or 0)
        }
    
    except Exception as e:
        #If evalaution fails, log the error and return default values
        logger.warning(f"Evaluation failed: {e}")
        return {
          "faithfulness": 0.0,
          "answer_relevancy": 0.0,
          "error": str(e)
        }


if __name__ == "__main__":
    test_question = "What is RAG?"
    test_script   = "Host: What is RAG?\nExpert: RAG stands for Retrieval Augmented Generation."
    test_context  = "RAG combines document retrieval with language model generation."
    print(evaluate_rag(test_question, test_script, test_context))