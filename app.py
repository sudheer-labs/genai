import psycopg2
from youtube_transcript_api import YouTubeTranscriptApi
import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
import re
from dotenv import load_dotenv
import os
from openai import OpenAI
from pgvector.psycopg2 import register_vector
from fastapi import FastAPI
from pydantic import BaseModel
import requests


load_dotenv()

app = FastAPI(title="Youtube Transcript RAG API")
nltk.download("punkt")
#nltk.download("punkt_tab")

# Use sent_tokenize of NLTK for breaking the transcript into sentences
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_TOKENS = 512 # BERT's max token limit (accounting for [CLS] and [SEP])


# -------------------------------------
# Pydantic Models for API
#--------------------------------------
class YouTubeRequest(BaseModel):
    youtube_link: str

class QuestionRequest(BaseModel):
    question: str

# -------------------------------------
# API Endpoints
# -------------------------------------

@app.post("/ingest")
def ingest_video(request: YouTubeRequest):
    video_id = request.youtube_link.split("v=")[-1].split("&")[0]
    chunks = retrieve_and_chunk(request.youtube_link)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    for chunk, embedding in zip(chunks, embeddings):
        cursor.execute(
            """
            INSERT INTO youtube_chunks
            (video_id, chunk_text, start_time, end_time, duration, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                video_id,
                chunk["text"],
                chunk["start"],
                chunk["end"],
                chunk["duration"],
                chunk["token_count"],
                embedding
            )
        )

    conn.commit()
    return {"message": "Video ingested successfully"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    query_embedding = generate_embeddings([request.question])[0]

    cursor.execute(
        """
        SELECT chunk_text
        FROM youtube_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT 5
        """,
        (query_embedding,)
    )

    results = [r[0] for r in cursor.fetchall()]
    context = "\n\n".join(results)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using the provided context only."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{request.question}"}
        ],
        temperature=0.2
    )

    return {"answer": response.choices[0].message.content}

# -------------------------------------
# Helper Functions
# -------------------------------------

def retrieve_and_chunk(youtube_link: str):
    video_id = youtube_link.split("v=")[-1].split("&")[0]
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    sentences_with_time = []
    buffer_text = ""
    sentence_start = None
    sentence_end = None

    # Step 1: Build time-aligned sentences
    for snippet in transcript:
        clean_text = re.sub(r"\[.*?\]", "", snippet.text).strip()
        if not clean_text:
            continue

        if sentence_start is None:
            sentence_start = snippet.start

        buffer_text += " " + clean_text
        sentence_end = snippet.start + snippet.duration

        sentences = sent_tokenize(buffer_text)

        while len(sentences) > 1:
            finalized = sentences.pop(0)

            sentences_with_time.append({
                "text": finalized.strip(),
                "start": sentence_start,
                "end": sentence_end,
                "duration": sentence_end - sentence_start
            })

            buffer_text = " ".join(sentences)
            sentence_start = snippet.start
            sentences = sent_tokenize(buffer_text)

    if buffer_text.strip():
        sentences_with_time.append({
            "text": buffer_text.strip(),
            "start": sentence_start,
            "end": sentence_end,
            "duration": sentence_end - sentence_start
        })

    # Step 2: Chunk sentences under 512 BERT tokens
    chunks = []
    current_chunk = []
    current_token_count = 0
    chunk_start = None
    chunk_end = None

    for sentence in sentences_with_time:
        tokens = tokenizer.tokenize(sentence["text"])
        token_count = len(tokens)

        if current_token_count + token_count > MAX_TOKENS:
            # finalize current chunk
            chunks.append({
                "text": " ".join([s["text"] for s in current_chunk]),
                "start": chunk_start,
                "end": chunk_end,
                "duration": chunk_end - chunk_start,
                "token_count": current_token_count
            })

            # reset
            current_chunk = []
            current_token_count = 0
            chunk_start = None

        # add sentence to chunk
        if chunk_start is None:
            chunk_start = sentence["start"]

        current_chunk.append(sentence)
        current_token_count += token_count
        chunk_end = sentence["end"]

    # finalize last chunk
    if current_chunk:
        chunks.append({
            "text": " ".join([s["text"] for s in current_chunk]),
            "start": chunk_start,
            "end": chunk_end,
            "duration": chunk_end - chunk_start,
            "token_count": current_token_count
        })

    return chunks
    

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

register_vector(conn)
cursor = conn.cursor()


def generate_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def store_chunks(youtube_link):
    video_id = youtube_link.split("v=")[-1].split("&")[0]
    chunks = retrieve_and_chunk(youtube_link)

    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    for chunk, embedding in zip(chunks, embeddings):
        cursor.execute(
            """
            INSERT INTO youtube_chunks
            (video_id, chunk_text, start_time, end_time, duration, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                video_id,
                chunk["text"],
                chunk["start"],
                chunk["end"],
                chunk["duration"],
                chunk["token_count"],
                embedding
            )
        )

    conn.commit()
    print("Chunks stored successfully!")

def search_similar(query, top_k=5):
    query_embedding = generate_embeddings([query])[0]

    cursor.execute(
        """
        SELECT chunk_text
        FROM youtube_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, top_k)
    )

    results = cursor.fetchall()
    return [r[0] for r in results]


def ask_question(query):
    relevant_chunks = search_similar(query)

    context = "\n\n".join(relevant_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer strictly using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# -------------------------------------
# UI Function using Gradio
# -------------------------------------

import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from gradio.routes import mount_gradio_app

# -------------------------------------
# Allow CORS
# -------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------
# Gradio Functions (calling your fast api endpoints)
# -------------------------------------

def gradio_ingest(youtube_link):
    response = requests.post(
        "http://localhost:8000/ingest",
        json={"youtube_link": youtube_link}
    )
    return response.json()["message"]



def gradio_chat(message, history):
    answer = ask_question(message)

    history = history or []
    history.append(
        {"role": "user", "content": message}
    )

    history.append(
        {"role": "assistant", "content": answer}
    )

    return history


# -------------------------------------
# Gradio UI Layout
# -------------------------------------

with gr.Blocks() as gradio_app:
    gr.Markdown("# YouTube Transcript RAG Chatbot")

    with gr.Row():
        youtube_link = gr.Textbox(label="Enter YouTube URL")
        ingest_btn = gr.Button("Ingest Video")

    ingest_output = gr.Textbox(label="Ingestion Status")

    ingest_btn.click(
        gradio_ingest,
        inputs=youtube_link,
        outputs=ingest_output
    )

    gr.Markdown("## Ask Questions")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something about the video...")
    clear = gr.Button("Clear")

    msg.submit(
        gradio_chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    )

    clear.click(lambda: None, None, chatbot, queue=False)


# -------------------------------------
# Mount Gradio into FastAPI
# -------------------------------------

app = mount_gradio_app(app, gradio_app, path="/gradio")


# Also try semantic chunking using tiktoken
# https://www.youtube.com/watch?v=E2DEHOEbzks
# if __name__ == "__main__":
    # youtube_link = "https://www.youtube.com/watch?v=02YLwsCKUww" # "https://www.youtube.com/watch?v=02YLwsCKUww"
    # db_connection()
    # Step 1: Store chunks
    #store_chunks(youtube_link)

    # Step 2: Ask question
    #answer = ask_question("What is the main topic discussed?")
    #print(answer)
