import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


# ============================================================
# AI UNIVERSITY STUDY ASSISTANT
# Fast local version for classroom / teacher demonstration
# ============================================================

APP_TITLE = "📚 AI University Study Assistant"

LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fast PDF settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
QA_TOP_K = 3

# Short generation limits keep the live demo responsive
QA_MAX_TOKENS = 300
SUMMARY_MAX_TOKENS = 300
NOTES_MAX_TOKENS = 350
QUIZ_MAX_TOKENS = 500


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="AI University Study Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# LOCAL AI
# ============================================================

# ONLINE AI
@st.cache_resource
def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=QA_MAX_TOKENS,
        api_key=st.secrets["GROQ_API_KEY"],
    )


@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# PDF PROCESSING + VECTOR DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_and_vectorize(file_bytes, file_name, file_hash):
    """
    Load the PDF, split it into chunks and store the chunks
    in a local Chroma vector database.

    Streamlit caches this function, so the same PDF is not
    embedded again every time the page reruns.
    """

    temp_path = f"uploaded_{file_hash[:12]}.pdf"

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

  
    # Remove empty PDF pages
    
    
   
    pages = [
        page
        for page in pages
        if page.page_content and page.page_content.strip()
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)
    # Add human-friendly page numbers
    for chunk in chunks:
        page_number = chunk.metadata.get("page", 0)
        chunk.metadata["page_number"] = int(page_number) + 1
        chunk.metadata["source_file"] = file_name

    embeddings = get_embedding()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"study_assistant_{file_hash[:16]}",
    )

    return vectorstore, len(pages), len(chunks)

# ============================================================
# DOCUMENT HELPERS
# ============================================================

def retrieve_documents(vectorstore, question, k=QA_TOP_K):
    """Retrieve only the most relevant chunks for a question."""
    return vectorstore.similarity_search(question, k=k)


def format_sources(docs):
    """Return unique page references."""
    pages = []

    for doc in docs:
        page = doc.metadata.get("page_number", "?")
        if page not in pages:
            pages.append(page)

    if not pages:
        return "No source pages found."

    return ", ".join(f"Page {page}" for page in pages)



def get_representative_content(vectorstore):
    """Return all document chunks for full-document processing."""
    data = vectorstore.get(include=["documents"])
    documents = data.get("documents", [])

    if not documents:
        return "No document content found."

    return "\n\n".join(
        document for document in documents if document and document.strip()
    )


# ============================================================
# ANSWER QUESTIONS
# ============================================================

def answer_question(vectorstore, question):
    docs = retrieve_documents(
        vectorstore,
        question,
        k=QA_TOP_K,
    )

    if not docs:
        return (
            "I don't know based on the uploaded document.",
            "No source pages found.",
        )

    context_parts = []

    for doc in docs:
        page = doc.metadata.get("page_number", "?")
        context_parts.append(
            f"[Page {page}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an AI University Study Assistant.

Answer the student's question using ONLY the retrieved information
from the uploaded PDF.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not contained in the retrieved text, say:
   "I don't know based on the uploaded document."
4. Give a clear answer suitable for a university student.
5. Mention the relevant page number when possible.
6. Keep the answer concise and organized.

STUDENT QUESTION:
{question}

RETRIEVED PDF CONTENT:
{context}

ANSWER:
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    return response.content, format_sources(docs)


# ============================================================
# FAST SUMMARY
# ============================================================


def generate_summary(vectorstore):
    data = vectorstore.get(include=["documents"])
    documents = [
        doc for doc in data.get("documents", [])
        if doc and doc.strip()
    ]

    if not documents:
        return "Unable to retrieve document content."

    llm = get_llm()
    partial_summaries = []

    for index, document in enumerate(documents, start=1):
        prompt = f"""
Summarize this section of a university study PDF.

Rules:
- Use only the supplied text.
- Include important concepts, definitions, and facts.
- Use simple, student-friendly language.
- Do not invent information.
- Keep this section summary concise.

PDF SECTION:
{document}
"""
        response = llm.invoke(prompt)
        partial_summaries.append(response.content)

    final_prompt = f"""
Create a comprehensive university study summary of the PDF
using all the section summaries below.

Requirements:
- Organize the result into 5 to 12 clear bullet points.
- Include the most important concepts from across the document.
- Use simple, student-friendly language.
- Do not add outside knowledge.
- Avoid unnecessary repetition.

SECTION SUMMARIES:
{"".join(chr(10) + summary for summary in partial_summaries)}
"""
    response = llm.invoke(final_prompt)
    return response.content


# ============================================================
# FAST STUDY NOTES
# ============================================================


def generate_notes(vectorstore):
    data = vectorstore.get(include=["documents"])
    documents = [
        doc for doc in data.get("documents", [])
        if doc and doc.strip()
    ]

    if not documents:
        return "Unable to retrieve document content."

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=NOTES_MAX_TOKENS,
        api_key=st.secrets["GROQ_API_KEY"],
    )

    section_notes = []

    for document in documents:
        prompt = f"""
Create concise exam-oriented study notes from this PDF section.

Requirements:
- Use headings and bullet points.
- Include important definitions, concepts, and facts.
- Keep explanations short and easy to revise.
- Use only the supplied text.
- Do not invent information.

PDF SECTION:
{document}

STUDY NOTES:
"""
        response = llm.invoke(prompt)
        section_notes.append(response.content)

    final_prompt = f"""
Combine the following section notes into comprehensive study notes
for the entire PDF.

Requirements:
- Use clear headings and bullet points.
- Preserve important definitions, concepts, and facts.
- Remove unnecessary repetition.
- Use simple, student-friendly language.
- Do not add outside knowledge.

SECTION NOTES:
{"".join(chr(10) + notes for notes in section_notes)}
"""
    response = llm.invoke(final_prompt)
    return response.content
   


# ============================================================
# FAST QUIZ
# ============================================================


def generate_quiz(vectorstore):
    data = vectorstore.get(include=["documents"])
    documents = [
        doc for doc in data.get("documents", [])
        if doc and doc.strip()
    ]

    if not documents:
        return "Unable to retrieve document content."

    llm = get_llm()
    question_sets = []

    for document in documents:
        prompt = f"""
You are a university quiz generator.

Create up to 2 multiple-choice questions from this PDF section.
If the section does not contain enough information, create fewer.

Rules:
- Use only the supplied text.
- Each question must have four options: A, B, C, D.
- Mark the correct answer.
- Keep questions and options short.
- Do not invent information.

Use this format:

1. Question
A. Option
B. Option
C. Option
D. Option
Correct Answer: B

PDF SECTION:
{document}

QUIZ:
"""
        response = llm.invoke(prompt)
        if response.content.strip():
            question_sets.append(response.content.strip())

    if not question_sets:
        return "Unable to generate quiz questions from this PDF."

    final_prompt = f"""
Create a quiz with EXACTLY 5 multiple-choice questions using
the section question sets below.

Rules:
- Choose questions covering different important topics.
- Each question must have four options: A, B, C, D.
- Include exactly one correct answer for each question.
- Mark each correct answer.
- Use only information in the supplied question sets.
- Do not invent facts.
- Output exactly 5 questions in the same numbered format.

SECTION QUESTION SETS:
{"".join(chr(10) + questions for questions in question_sets)}

FINAL QUIZ:
"""
    response = llm.invoke(final_prompt)
    return response.content

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "notes" not in st.session_state:
    st.session_state.notes = None

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Study Assistant")

    st.success(
        "🔒 100% Local AI\n\n"
        "No OpenAI API key is required."
    )

    st.write("**AI Model:**")
    st.code(LLM_MODEL)

    st.write("**Embedding Model:**")
    st.code(EMBEDDING_MODEL)

    st.divider()

    uploaded_file = st.file_uploader(
        "📄 Upload your university PDF",
        type=["pdf"],
    )

    st.divider()

    

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(APP_TITLE)

st.write(
    "A local AI-powered study assistant that reads your PDF, "
    "answers questions, creates summaries, generates study notes, "
    "and creates practice MCQs."
)


# ============================================================
# NO PDF
# ============================================================

if uploaded_file is None:

    st.info("👉 Upload a PDF from the sidebar to begin.")

    st.markdown(
        """
### 🚀 What this application can do

**💬 Ask Questions**  
Ask questions about your uploaded university material.

**📝 Generate Summary**  
Create a concise summary from the PDF.

**📖 Generate Study Notes**  
Turn material into organized exam notes.

**❓ Generate Quiz**  
Create 5 practice multiple-choice questions.

**🔎 Source References**  
See which PDF pages were used for answers.

### 🔒 100% Local AI

The application uses an AI model to help students study university materials.

Upload a PDF to generate answers, summaries, study notes, and quizzes.

"""
    )

    st.stop()


# ============================================================
# PROCESS PDF
# ============================================================

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()

# Clear generated results when a different PDF is uploaded
if st.session_state.file_hash != file_hash:
    st.session_state.file_hash = file_hash
    st.session_state.chat_history = []
    st.session_state.summary = None
    st.session_state.notes = None
    st.session_state.quiz = None


with st.spinner("📄 Preparing your PDF..."):

    try:
        vectorstore, page_count, chunk_count = load_and_vectorize(
            file_bytes,
            uploaded_file.name,
            file_hash,
        )

    except Exception as e:
        st.error("Could not process the PDF.")
        st.code(str(e))
        st.stop()


st.success(
    f"✅ Document ready! "
    f"{page_count} pages processed into "
    f"{chunk_count} searchable chunks."
)


# ============================================================
# TABS
# ============================================================

tab_questions, tab_summary, tab_notes, tab_quiz = st.tabs(
    [
        "💬 Ask Questions",
        "📝 Summary",
        "📖 Study Notes",
        "❓ Quiz",
    ]
)


# ============================================================
# ASK QUESTIONS
# ============================================================

with tab_questions:

    st.subheader("💬 Ask Questions")

    st.caption(
        "Ask anything that is directly related to the uploaded PDF."
    )

    for item in st.session_state.chat_history:

        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.write(item["answer"])

            if item.get("sources"):
                st.caption(
                    f"🔎 Sources: {item['sources']}"
                )

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("🤖 Finding the answer..."):

                try:
                    answer, sources = answer_question(
                        vectorstore,
                        question,
                    )

                    st.write(answer)
                    st.caption(
                        f"🔎 Sources: {sources}"
                    )

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer,
                            "sources": sources,
                        }
                    )

                except Exception as e:
                    st.error(
                        "The AI could not answer the question."
                    )
                    st.code(str(e))


# ============================================================
# SUMMARY
# ============================================================

with tab_summary:

    st.subheader("📝 AI Summary")

    
            st.write(
                "Generate a comprehensive summary of your entire PDF."
            )

    if st.button(
        "📝 Generate Summary",
        use_container_width=True,
    ):

        with st.spinner(
            "Creating a summary from your PDF..."

            try:
                st.session_state.summary = generate_summary(
                    vectorstore
                )

            except Exception as e:
                st.error(f"Could not generate the summary: {e}")

    if st.session_state.summary:
        st.markdown("### Summary")
        st.markdown(st.session_state.summary)


# ============================================================
# STUDY NOTES
# ============================================================

with tab_notes:

    st.subheader("📖 Study Notes")

    st.write(
        "Generate concise exam-oriented notes from the PDF."
    )

    if st.button(
        "📖 Generate Study Notes",
        use_container_width=True,
    ):

        with st.spinner(
           "Creating study notes from your PDF..."

            try:
                st.session_state.notes = generate_notes(
                    vectorstore
                )

            except Exception as e:
                st.error(
                    "Could not generate study notes."
                )
                st.code(str(e))

    if st.session_state.notes:
        st.markdown("### Study Notes")
        st.markdown(st.session_state.notes)


# ============================================================
# QUIZ
# ============================================================

with tab_quiz:

    st.subheader("❓ Practice Quiz")

    st.write(
        "Generate 5 multiple-choice questions from the PDF."
    )

    if st.button(
        "❓ Generate 5 MCQs",
        use_container_width=True,
    ):

        with st.spinner(
            "Creating a quiz from your PDF..."

            
            try:
                st.session_state.quiz = generate_quiz(
                    vectorstore
                )

            except Exception as e:
                st.error(f"Could not generate the quiz: {e}")
    

   
    if st.session_state.quiz:
        st.markdown("### Practice Quiz")
        questions = st.session_state.quiz.split("\n\n")
        for i, question in enumerate(questions):
            if question.strip():
                st.markdown(f"#### Question {i + 1}")
                st.markdown(question)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "📚 AI University Study Assistant | "
    "AI-powered study tools | "
   "PDF summaries, notes and quizzes"
)
