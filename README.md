# 🎙️ DeepTalk – AI Podcast Generator from Documents

> **Transform any PDF document into a conversational podcast using AI.**

DeepTalk is an AI-powered podcast generation system that converts uploaded PDF documents into engaging host–expert style dialogues, synthesized into multi-voice audio with background music. It combines **LLMs**, **vector databases**, **LangGraph workflows**, and **text-to-speech synthesis** into a fully automated podcast pipeline.
---

## 🎥 Demo Video
Watch the project demo here: 
https://github.com/user-attachments/assets/e3d92c8d-406a-465c-b7b7-ec3fa4346ebe
---

## 🚀 Features

| Feature | Description |
|---|---|
| 📄 PDF Upload | Upload any PDF document as the podcast source |
| 🔍 RAG Pipeline | Automatically index and retrieve document knowledge |
| 🎙️ Podcast Modes | Auto, Question & Answer, and Debate modes |
| 🗣️ Multi-Voice Audio | Convert scripts into multi-speaker TTS audio |
| 🎵 Background Music | Mix background music into the final podcast |
| 🔗 LangGraph Workflow | Orchestrated AI agent pipeline |

---

## 🧠 System Architecture

```
┌──────────────────┐
│   User Uploads   │
│      PDF         │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  Document Loader │
│    (PyPDF)       │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  Text Chunking   │
│ Recursive Split  │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Embedding Model  │
│ BAAI/bge-small   │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Vector Database  │
│    ChromaDB      │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│    Retriever     │
│  (MMR Search)    │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│   LangGraph      │
│ Workflow Engine  │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  LLM Generation  │
│  Podcast Script  │
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│  MeloTTS Voices  │
│ Multi-speaker TTS│
└─────────┬────────┘
          │
          ▼
┌──────────────────┐
│ Background Music │
│     Mixing       │
└─────────┬────────┘
          │
          ▼
    🎧 Final Podcast Audio
```

---

## ⚙️ Tech Stack

### 🤖 AI / ML
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [LangGraph](https://langchain-ai.github.io/langgraph/) – Agent workflow engine
- [Groq LLM](https://groq.com/) – Fast LLM inference
- [HuggingFace Embeddings](https://huggingface.co/) – Sentence embeddings
- [Sentence Transformers](https://www.sbert.net/) – Semantic similarity

### 🗄️ Vector Database
- [ChromaDB](https://www.trychroma.com/) – Local vector store for RAG

### 📝 Text Processing
- [PyPDF](https://pypi.org/project/pypdf/) – PDF loading
- `RecursiveCharacterTextSplitter` – Document chunking

### 🔊 Audio Generation
- [MeloTTS](https://github.com/myshell-ai/MeloTTS) – Multi-speaker text-to-speech
- [Pydub](https://github.com/jiaaro/pydub) – Audio mixing and post-processing

### 🖥️ Frontend
- [Streamlit](https://streamlit.io/) – Web application interface

---

## 📂 Project Structure

```
Podcast-AI/
│
├── app.py                  # Streamlit application interface
├── rag_engine.py           # PDF loading, chunking, embeddings & ChromaDB
├── langgraph_flow.py       # LangGraph workflow for retrieval & script generation
├── tts_engine.py           # Multi-speaker podcast audio synthesis
├── audio_engine.py         # Background music mixing
├── evaluation.py           # RAG evaluation using RAGAS
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🔄 Pipeline Workflow

### 1️⃣ Document Indexing
The uploaded PDF is processed using `PyPDFLoader` and `RecursiveCharacterTextSplitter`, dividing large documents into smaller semantic chunks for retrieval.

### 2️⃣ Embedding Generation
Each chunk is converted into vector embeddings using **BAAI/bge-small-en-v1.5**, enabling semantic search within the document.

### 3️⃣ Vector Storage
Embeddings are stored in a **Chroma** vector database for efficient similarity-based retrieval.

### 4️⃣ Retrieval-Augmented Generation (RAG)
When a podcast is requested:
- User query is received
- Relevant chunks are retrieved via **MMR Search**
- Context is injected into the LLM prompt
- The model generates a structured podcast dialogue

### 5️⃣ Podcast Script Generation
**LangGraph** orchestrates the workflow:
```
retrieve_node → generate_node
```
The LLM produces structured dialogue:
```
Host:   [Question or topic introduction]
Expert: [Detailed explanation based on document]
```

### 6️⃣ Text-to-Speech Conversion
The script is synthesized using **MeloTTS**, with distinct voices assigned to the Host and Expert speakers.

### 7️⃣ Audio Post-Processing
Final audio is enhanced by mixing in background music using **Pydub**, producing the finished podcast.

---

## 🎙️ Podcast Modes

| Mode | Description |
|---|---|
| 🤖 **Auto Mode** | Automatically generates a full topic overview |
| ❓ **Q&A Mode** | Host asks questions, Expert provides answers |
| ⚔️ **Debate Mode** | Host and Expert take opposing viewpoints |

---

## 📊 RAG Evaluation

The system includes evaluation using [RAGAS](https://docs.ragas.io/) metrics:

| Metric | Score |
|---|---|
| Faithfulness | 0.57 |
| Answer Relevancy | 0.25 |

> These metrics reflect how well the generated answers align with retrieved document content.

---

## 📌 Example Output

```
Host:   What is the main purpose of this document?

Expert: The document explains several business processes,
        including recruitment, insurance, and procurement.

Host:   Can you explain the recruitment process?

Expert: The recruitment process includes defining job roles,
        writing job descriptions, advertising positions,
        and selecting qualified candidates.
```
🎧 **Output:** Final synthesized podcast audio file

---

## 🖥️ Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/podcast-ai.git
cd podcast-ai
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
```
Activate it:
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

- [ ] Real-time podcast streaming
- [ ] Multi-agent podcast generation
- [ ] More realistic dialogue generation
- [ ] Voice emotion and tone control
- [ ] Cloud deployment (AWS / GCP / Azure)

---

⚠️ Deployment Status

Currently, the project is not deployed on Hugging Face Spaces or other cloud platforms due to some dependency compatibility issues related to the environment and certain libraries used in the system.

The application runs successfully in a local environment, but deployment on some hosted platforms requires additional configuration or dependency adjustments.

👨‍💻 Author
Jatin Dhanda
AI / Machine Learning Enthusiast — Focused on building LLM-powered intelligent systems

---

