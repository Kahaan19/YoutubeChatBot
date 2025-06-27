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
    # Perform similarity search with scores
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=6)

    # Print the scores for debugging/insight
    print("\n🔍 Retrieved Documents with Similarity Scores:")
    for i, (doc, score) in enumerate(docs_with_scores):
        print(f"[{i+1}] Score: {score:.4f}\nSnippet: {doc.page_content[:100]}...\n")

    # Optional: Filter by a minimum score threshold (e.g., 0.7 is a decent start)
    filtered_docs = [doc for doc, score in docs_with_scores if score >= 0.7]

    # Use all if nothing meets threshold
    if not filtered_docs:
        print("⚠️ No documents met the threshold. Using top results anyway.")
        filtered_docs = [doc for doc, _ in docs_with_scores]

    # Prepare context for the prompt
    context = "\n\n".join([doc.page_content for doc in filtered_docs])

    # Prompt templates
    system_template = "You are a helpful assistant that answers questions based on the following video transcript:\n\n{context}"
    user_template = "{question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template),
    ]).invoke({"context": context, "question": question})

    # Generate response using Gemini
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
