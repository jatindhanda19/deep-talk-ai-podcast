"""
This  is main streamlit appliaction for DeepTalk
The app allow user to:
1.Upload a PDF document
2.Generate a podcast-style conversation from document
3.Convert the script into multi-voice audio
4.Optionally add background music
5.Evaluate the generated podcast using RAG metrics

The interface is designed to be simple:
Upload → Generate Script → Convert to Audio → Evaluate.
"""
import os
import logging
import streamlit as st

from rag_engine import build_vectorstore
from langgraph_flow import build_graph
from tts_engine import generate_multi_voice_audio
from audio_engine import add_background
from evaluation import evaluate_rag

#Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Streamlit Page Configuration
st.set_page_config(
    page_title="DeepTalk",
    page_icon="🎙️",
    layout=  "centered"
)

#Cache the LangGraph pipeline
@st.cache_resource(show_spinner="Loading pipeline...")
def get_graph():
    """Compile the langGraph pipeline only once
    Streamlit would otherwise rebuild the graph on every interaction,
    so caching improves performance significantly
    """
    return build_graph()

#Helper Function
def parse_script(script) -> list[tuple[str,str]]:
    """Convert the generated script into structured (speaker,text) pairs.
       This helps format the script nicely for display in UI
    """
    if isinstance(script, list):
        script = "\n".join(str(item) for item in script)

    segments = []

    for line in script.strip().split("\n"):
        line = line.strip()
        if not line or line.lower().startswith("title"):
            continue

        if ":" in line:
            speaker, text = line.split(":", 1)
            speaker, text  = speaker.strip(), text.strip()
            if speaker and text:
                segments.append((speaker.strip(),text.strip()))

    return segments

def speaker_prefix(speaker:str) -> str:
   """Add a small emoji indicator for each speaker role"""
   s = speaker.lower()
   if s == "host":     return "🎙️"
   if s == "expert_a": return "🔵"
   if s == "expert_b": return "🟠"
   return "👤"
       

# Header
st.title("🎙️ DeepTalk")
st.caption("Transform any PDF into a multi-Voice podcast")
st.divider()

#Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.divider()
    
    # Choose podcast generation mode
    mode = st.selectbox("🎚️ Podcast Mode",
                        ["Auto","Debate","Q&A"],
                        help="Auto: AI picks key themes. Debate: Two experts argue. Q&A: Host asks questions based on content."
                        )
    
    #Toggle background music
    bg_music = st.toggle(
        "🎵 Background Music",
        value = True,
        help="Mix soft background music under the voices"
    )
    st.divider()

#PDF upload Section
st.subheader("📄 Upload your PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

question =""
# In Q&A mode we allow the user to ask question
if mode == "Q&A":
    st.divider()

    question = st.text_input("❓Your Question",
                             placeholder="E.g. What are the key trends in renewable energy?")

#Document Indexing  
if uploaded_file:

    stable_dir = "indexed_pdfs"
    os.makedirs(stable_dir, exist_ok=True)

    stable_path = os.path.join(stable_dir, uploaded_file.name)

    file_changed = st.session_state.get("last_uploaded", "") != uploaded_file.name

    if file_changed:

        file_byte =uploaded_file.getvalue()

        if not file_byte:
            st.error("Uploaded file is empty.")
            st.stop()

        #Save the Uploded file locally
        with open(stable_path, "wb") as f:
            f.write(file_byte)

        # Build vectorstore index
        with st.spinner("📚 Indexing document..."):

            try:
                vs = build_vectorstore(stable_path)

                retriever = None

                if hasattr(vs, "as_retriever"):
                    retriever = vs.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k":4, "fetch_k": 20, "lambda_mult": 0.5}
                    )

                else:
                    retriever = vs
                #Store in session
                st.session_state.vectorstore = vs
                st.session_state.retriever = retriever
                st.session_state.last_uploaded = uploaded_file.name

            except Exception as e:
                st.error(f"Error during document indexing: {e}")
                st.stop()
        st.success("✅ Document indexed successfully!")
    else:
        st.info("📎 Using cached index.")    
               
st.divider()

#Podcast Generation Section
st.subheader("🎬 Generating  Podcast")

if st.button("Generate Podcast", type="primary", use_container_width=True):

    if not uploaded_file:
        st.warning("Please upload a PDF first.")
        st.stop()

    if mode == "Q&A" and not question.strip():
        st.warning("Please enter a question for Q&A mode.")
        st.stop()    

    if "retriever" not in st.session_state:
        st.warning("Document not indexed. Please re-upload the PDF.")
        st.stop()

    # script generation
    with st.spinner("✍️ Generating podcast script..."):
        try:
            result = get_graph().invoke({
                "question": question,
                "retriever": st.session_state.retriever,
                "vectorstore": st.session_state.vectorstore,
                "mode": mode,
                "context": "",
                "script": ""
            })

        except Exception as e:
            st.error(f"Error during script generation: {e}")
            st.stop()

    script = result.get("script", "")
    context = result.get("context", "")

    if not script:
        st.error("Failed to generate podcast script.")
        st.stop()

    st.session_state.last_script = script
    st.session_state.last_context = context
    
    segments = parse_script(script)

    st.divider()

    st.subheader("📝Script")

    display = ""
    for speaker, text in segments:
        prefix = speaker_prefix(speaker)
        display += f"{prefix} {speaker}: {text}\n\n"
        
    st.text_area(
        label="Script",
        value=display,
        height=300,
        label_visibility="collapsed"
        )
    st.divider()
   
    #Text-to-speech Generation 
    st.subheader("🔊 Audio")

    with st.spinner("🎤 Generating audio (it might take minute)..."):
            try:
                voice_file = generate_multi_voice_audio(script)
            except Exception as e:
                st.error(f"Error during TTS generation: {e}")
                st.stop()
    
    #optional background music
    final_file = voice_file

    if bg_music:
            with st.spinner("🎵 Mixing background music..."):
                try:
                    final_file = add_background(voice_file, "podcast background music.mp3")
                except Exception as e:
                    st.error(f"Error during audio mixing: {e}")
                    st.stop()        

    st.audio(final_file)
    
    #Download button
    with open(final_file, "rb") as f:
            st.download_button(
                label="📥 Download Podcast",
                data=f,
                file_name="deeptalk_podcast.mp3",
                mime="audio/mpeg",
                use_container_width=True
            )
    st.divider()
    
    #RAG Evaluation Metrics
    st.subheader("📊 Evaluation")

    with st.spinner("Evaluating podcast quality..."):
            
            try:
                scores = evaluate_rag(question, script, context)
                faith =round(float(scores.get("faithfulness", 0)), 2)
                relev = round(float(scores.get("answer_relevancy", 0)), 2)

                c1, c2 = st.columns(2)
                c1.metric(
                    label="📏 Faithfulness",
                    value=f"{faith}/1.0",
                    help="How well the podcast content is grounded in the source document."
                )
                c2.metric(
                    label="📊 Answer Relevancy",
                    value=f"{relev}/1.0",
                    help="How relevant the podcast answers are to the user's question."
                )

            except Exception as e:
                st.error(f"Error during evaluation: {e}")
                st.stop()

#Display previous script if available
elif "last_script" in st.session_state:
    segments = parse_script(st.session_state.last_script)
    st.divider()
    st.subheader("📝 Script")

    display = ""

    for speaker, text in segments:
        prefix = speaker_prefix(speaker)
        display += f"{prefix} {speaker}: {text}\n\n"

    st.text_area(
        label="Script",
        value=display,
        height=300,
        label_visibility="collapsed"
    )
    