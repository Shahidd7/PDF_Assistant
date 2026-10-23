import groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage,AIMessage

load_dotenv()
DB_DIR = "chroma_db"

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function = embedding,
    persist_directory = DB_DIR
)

count = vector_store._collection.count()
print("Chunks in database:", count)
if count == 0:
    raise SystemExit("Database is empty. Run database.py first.")

retriever = vector_store.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k" : 15,
        "fetch_k" : 40,
        "lambda_mult" : 0.5
    }
)

llm = ChatGroq(model_name = "openai/gpt-oss-20b",temperature = 0.2, 
               max_tokens = 1024,max_retries = 3  )

prompt = ChatPromptTemplate.from_messages([
    ("system", "Using ONLY the context, write 5-8 key points about the topic. "
     "Cite pages like [p. 12]. If the context does not cover the topic, say this topic is not covered in the given pdf."),

    ("human","Context:\n{context}\n\nQuestion: {question}")
    ])

while True :
    query = input("Ask your question: ")
    if query.lower().strip() =="exit":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([f"[Page {d.metadata['page']}]\n{d.page_content}" for d in docs])

    pages = sorted({d.metadata["page"] for d in docs})
    print("Pages retrieved:", pages)

    final_prompt = prompt.invoke(
        {
            "context" : context,
            "question" : query
        }
    )

    try:
        response = llm.invoke(final_prompt)
    except groq.RateLimitError:
        print("Rate limit hit. Wait a few seconds and try again.")
        continue
    except groq.APIConnectionError:
        print("Can't reach Groq. Check your internet.")
        continue
    except groq.APIStatusError as e:
        print(f"API error {e.status_code}.")
        continue

    print("AI:",response.content)

