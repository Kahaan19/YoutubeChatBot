import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # updated import per deprecation warning
from langchain.schema import Document, BaseRetriever
from langchain.chains import RetrievalQA


# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize model & embeddings
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)

# Step 1: Fetch YouTube transcript
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([t['text'] for t in transcript_list])
        return transcript
    except Exception as e:
        return f"Could not retrieve transcript: {e}"

# Step 2: Build vectorstore
def build_vectorstore(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.create_documents([transcript])
    vectorstore = FAISS.from_documents(texts, embedding)
    return vectorstore

# Step 3: Custom retriever to include scores
class ScoredRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 6

    def get_relevant_documents(self, query):
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)
        documents = []
        for doc, score in results:
            doc.metadata["score"] = score
            documents.append(doc)
        return documents

# Step 4: Generate response with chaining
def generate_response_with_chain(vectorstore, question, k=6):
    retriever = ScoredRetriever(vectorstore=vectorstore, k=k)

    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",  # later customize if needed
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]

    print("\n🔍 Retrieved Documents with Scores:")
    for i, doc in enumerate(sources):
        score = doc.metadata.get("score", "N/A")
        print(f"\n📄 Doc {i+1} (Score: {score:.4f}):")
        print(doc.page_content[:300])  # first 300 chars

    return answer

# Step 5: Main flow
def main():
    video_id = input("Enter YouTube video ID: ").strip()
    question = input("Ask your question based on the video transcript: ").strip()

    transcript = get_transcript(video_id)
    if transcript.startswith("Could not"):
        print(transcript)
        return

    print("\n✅ Transcript fetched successfully!")

    vectorstore = build_vectorstore(transcript)
    answer = generate_response_with_chain(vectorstore, question, k=6)

    print("\n💬 Answer from AI:")
    print(answer)

if __name__ == "__main__":
    main()
