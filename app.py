# ============================================================
# RESTAURANT CHATBOT WITH RAG + LANGCHAIN + GROQ + GRADIO
# ============================================================

import os
import shutil
import uuid
import gradio as gr

# Try loading dotenv for local execution
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try loading from google.colab if available, otherwise fallback to os.environ
try:
    from google.colab import userdata
    COLAB_ENV = True
except ImportError:
    userdata = None
    COLAB_ENV = False

# LangChain
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. LOAD GROQ API KEY
# ============================================================

GROQ_API_KEY = None

if userdata:
    try:
        GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    except Exception:
        pass

if not GROQ_API_KEY:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Set it as an environment variable before running the application."
    )

print("Groq API key loaded successfully.")


# ============================================================
# 2. LOCATE RESTAURANT PDF FILES
# ============================================================

BRIEFING_PDF = "restaurant_briefing.pdf"
MENU_PDF = "restaurant_menu.pdf"

if not os.path.exists(BRIEFING_PDF) or not os.path.exists(MENU_PDF):
    # Check if in Colab where user might upload files
    if COLAB_ENV:
        print()
        print("Please upload these two files:")
        print("1. restaurant_briefing.pdf")
        print("2. restaurant_menu.pdf")
        print()
        from google.colab import files
        uploaded_files = files.upload()

        briefing_uploaded = None
        menu_uploaded = None

        for filename in uploaded_files:
            lower_name = filename.lower()
            if "briefing" in lower_name:
                briefing_uploaded = filename
            elif "menu" in lower_name:
                menu_uploaded = filename

        if briefing_uploaded:
            shutil.copy(briefing_uploaded, BRIEFING_PDF)
        if menu_uploaded:
            shutil.copy(menu_uploaded, MENU_PDF)

if not os.path.exists(BRIEFING_PDF):
    raise FileNotFoundError("Restaurant briefing PDF ('restaurant_briefing.pdf') was not found.")

if not os.path.exists(MENU_PDF):
    raise FileNotFoundError("Restaurant menu PDF ('restaurant_menu.pdf') was not found.")

print("Both PDF files found and ready.")


# ============================================================
# 3. INITIALIZE GROQ MODEL
# ============================================================

# Use Groq model (supports llama-3.3-70b-versatile, openai/gpt-oss-120b, etc.)
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

model = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0
)

print(f"Groq model initialized: {MODEL_NAME}")


# ============================================================
# 4. LOAD RESTAURANT BRIEFING & MENU
# ============================================================

briefing_loader = PyPDFLoader(BRIEFING_PDF)
briefing_pages = briefing_loader.load()
briefing_text = "\n".join(page.page_content for page in briefing_pages)

print(f"Restaurant briefing loaded: {len(briefing_pages)} pages")

menu_loader = PyPDFLoader(MENU_PDF)
menu_pages = menu_loader.load()
menu_text = "\n".join(page.page_content for page in menu_pages)

print(f"Restaurant menu loaded: {len(menu_pages)} pages")


# ============================================================
# 5. TRANSLATE PORTUGUESE MENU
# ============================================================

translation_prompt = f"""
You are a professional restaurant-menu translator.

Translate the following Portuguese restaurant menu
into English.

IMPORTANT RULES:

1. Preserve all dish names.
2. Preserve all prices exactly.
3. Preserve all ingredients.
4. Preserve vegetarian labels.
5. Preserve vegan labels.
6. Preserve menu categories.
7. Do not invent information.
8. Do not remove information.
9. Do not change prices.
10. Preserve all menu information.

Portuguese Restaurant Menu:

{menu_text}
"""

translation_response = model.invoke(
    [
        {
            "role": "user",
            "content": translation_prompt
        }
    ]
)

translated_menu = translation_response.content
print("Menu translation completed.")


# ============================================================
# 6. COMBINE RESTAURANT INFORMATION & CHUNK
# ============================================================

combined_text = f"""

==================================================
RESTAURANT BRIEFING
==================================================

{briefing_text}


==================================================
ENGLISH TRANSLATED RESTAURANT MENU
==================================================

{translated_menu}

"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = splitter.create_documents([combined_text])
print(f"Created {len(documents)} text chunks.")


# ============================================================
# 7. LOAD HUGGINGFACE EMBEDDINGS & CHROMADB
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

print("HuggingFace embedding model loaded.")

chroma_path = f"./restaurant_chroma_db_{uuid.uuid4().hex[:8]}"
os.makedirs(chroma_path, exist_ok=True)

vectorstore = Chroma(
    collection_name="restaurant_collection",
    embedding_function=embeddings,
    persist_directory=chroma_path
)

vectorstore.add_documents(documents)
print(f"Documents stored in ChromaDB (count: {vectorstore._collection.count()}).")

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

print("Retriever created.")


# ============================================================
# 8. RETRIEVAL TOOL & SYSTEM PROMPT
# ============================================================

@tool
def retrieve_restaurant_info(query: str) -> str:
    """
    Retrieve relevant restaurant information from
    the restaurant briefing and translated menu.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant restaurant information was found in the restaurant documents."
    return "\n\n".join(doc.page_content for doc in docs)

print("retrieve_restaurant_info tool created.")

system_prompt = """
You are a friendly and professional AI assistant
for a Brazilian restaurant.

Your job is to answer customer questions using the
restaurant's provided documents.

IMPORTANT RULES:

1. ALWAYS use retrieve_restaurant_info for
   restaurant-specific questions.

2. Only use information found in the restaurant
   documents.

3. Never invent menu items.

4. Never invent prices.

5. Never invent ingredients.

6. Never invent restaurant services.

7. Never invent restaurant policies.

8. Never invent opening hours.

9. Never assume information that is not present
   in the documents.

10. If the information is not available, clearly say:

    "I don't have that information in the
    restaurant documents."

11. Preserve prices exactly as provided.

12. Preserve vegetarian and vegan labels exactly
    according to the restaurant documents.

13. If a customer asks about a dish that does not
    appear in the menu, explain that it was not
    found in the menu.

14. Keep responses concise and friendly.

15. Use Markdown when useful.

16. Do not use general world knowledge to answer
    restaurant-specific questions.

17. Do not fabricate information.

18. If multiple retrieved chunks contain information,
    combine them carefully without duplicating dishes.
"""


# ============================================================
# 9. CREATE LANGCHAIN AGENT
# ============================================================

# Support both create_agent and tool calling across LangChain versions
try:
    from langchain.agents import create_agent
    agent = create_agent(
        model=model,
        tools=[retrieve_restaurant_info],
        system_prompt=system_prompt
    )
    _agent_mode = "create_agent"
except Exception:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    raw_agent = create_tool_calling_agent(model, [retrieve_restaurant_info], prompt)
    agent = AgentExecutor(agent=raw_agent, tools=[retrieve_restaurant_info], verbose=False)
    _agent_mode = "tool_calling"

print("Restaurant LangChain agent created.")


# ============================================================
# 10. GRADIO CHAT FUNCTION
# ============================================================

def restaurant_chat(message, history):
    try:
        if _agent_mode == "create_agent":
            messages = []
            if history:
                for item in history:
                    if isinstance(item, dict) and item.get("role") in ["user", "assistant"]:
                        messages.append({"role": item["role"], "content": item.get("content", "")})
            messages.append({"role": "user", "content": message})
            result = agent.invoke({"messages": messages})
            return result["messages"][-1].content
        else:
            chat_history = []
            if history:
                for item in history:
                    if isinstance(item, dict):
                        role = item.get("role")
                        content = item.get("content", "")
                        if role == "user":
                            chat_history.append(HumanMessage(content=content))
                        elif role == "assistant":
                            chat_history.append(AIMessage(content=content))
            result = agent.invoke({"input": message, "chat_history": chat_history})
            return result.get("output", "")

    except Exception as e:
        print("Restaurant chatbot error:", str(e))
        return (
            "❌ Sorry, I encountered an error while processing your question.\n\n"
            f"Error: `{str(e)}`"
        )


# ============================================================
# 11. GRADIO UI SETUP & LAUNCH
# ============================================================

theme = gr.themes.Soft(
    primary_hue="red",
    secondary_hue="orange",
    neutral_hue="slate"
)

chatbot = gr.Chatbot(
    height=550,
    placeholder="""
    <div style="text-align:center">
    <h2>👋 Welcome!</h2>
    <p>I'm your AI restaurant assistant.</p>
    <p>Ask me about our menu, prices, ingredients, vegetarian dishes, vegan options and restaurant information.</p>
    </div>
    """
)

textbox = gr.Textbox(
    placeholder="Ask about our menu, prices, ingredients...",
    container=False,
    scale=7
)

demo = gr.ChatInterface(
    fn=restaurant_chat,
    chatbot=chatbot,
    textbox=textbox,
    title="🍽️ Brazilian Restaurant AI Assistant",
    description="""
### Your intelligent restaurant concierge

Ask me about:

🍽️ **Menu items**  
💰 **Prices**  
🥗 **Vegetarian dishes**  
🌱 **Vegan options**  
🧂 **Ingredients**  
📋 **Restaurant information**

**Powered by Groq + LangChain + ChromaDB + HuggingFace**
""",
    examples=[
        "What dishes are available on the menu?",
        "What is the price of Picanha na Chapa?",
        "Which dishes are vegetarian?",
        "Which dish is vegan?",
        "What are the ingredients of Moqueca de Peixe?",
        "What is the most expensive dish?",
        "Do you serve pizza?",
        "What are the restaurant's opening hours?"
    ],
    example_labels=[
        "🍽️ Show me the menu",
        "💰 Check Picanha price",
        "🥗 Vegetarian dishes",
        "🌱 Vegan dishes",
        "🧂 Moqueca ingredients",
        "💎 Most expensive dish",
        "🍕 Pizza availability",
        "🕒 Opening hours"
    ],
    api_name="restaurant_chat"
)

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("STARTING RESTAURANT AI INTERFACE")
    print("=" * 60)
    print()

    demo.launch(
        share=False,
        debug=True,
        theme=theme
    )
