import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize Gemini chat model and embeddings
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=os.getenv("GOOGLE_API_KEY"))

# Get transcript from YouTube
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([t['text'] for t in transcript_list])
        return transcript
    except Exception as e:
        return f"Could not retrieve transcript: {e}"

# Chunk and embed transcript
def build_vectorstore(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(texts, embedding)
    return vectorstore

# RAG retrieval and response generation
def generate_response_with_rag(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    system_template = "You are a helpful assistant that answers questions based on the following video transcript:\n\n{context}"
    user_template = "{question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template),
    ]).invoke({"context": context, "question": question})

    response = model.invoke(prompt)
    return response.content

# Main CLI loop
def main():
    video_id = input("Enter YouTube video ID: ").strip()
    question = input("Ask your question based on the video transcript: ").strip()

    transcript = get_transcript(video_id)
    if transcript.startswith("Could not"):
        print(transcript)
        return

    print("\nTranscript fetched successfully!")

    vectorstore = build_vectorstore(transcript)
    answer = generate_response_with_rag(vectorstore, question)

    print("\nAnswer from AI:")
    print(answer)

if __name__ == "__main__":
    main()
