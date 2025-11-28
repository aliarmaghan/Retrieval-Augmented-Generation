# AI Onboarding Chatbot (Umbrella Onboarding Assistant)

This project details the creation of an AI assistant designed to handle employee onboarding, aiming to reduce the typically long and expensive onboarding process. This chatbot provides new employees with quick, context-specific answers based on internal company regulations.

The project uses the fictitious **Umbrella Corporation** as its company context.

## 🚀 Key Features

*   **Context-Aware Responses:** The AI assistant is able to detect and use the context of the user it is talking to (e.g., name, role, responsibilities).
*   **Knowledge Retrieval (RAG):** Implements a Retrieval-Augmented Generation (RAG) architecture to retrieve relevant context from internal company policies.
*   **Policy Access:** Utilizes a PDF document containing all of the internal regulations/corporate guidelines for the company (47 pages of text).
*   **User Interface:** Provides a graphical user interface (GUI) built with Streamlit.
*   **Streaming Generation:** Responses from the Language Model (LLM) are streamed to the user interface.
*   **Observability:** Integrated with Langsmith for tracking and tracing the execution of the application's chains and runnables.

## 🛠️ Technologies Used

This project implements RAG using a variety of modern AI/ML development tools:

| Category | Technology | Purpose | Sources |
| :--- | :--- | :--- | :--- |
| **Framework** | LangChain | Core framework for building the RAG chain and managing components. | |
| **Frontend/GUI** | Streamlit | Used to create the graphical user interface for the chatbot. | |
| **Vector Database** | ChromaDB (Chroma) | Used as the vector store to hold the vectorized policy documents. | |
| **LLMs/Models** | Grok (Chat Grok) | Used as the primary language model (e.g., LAMA 3.1 8B or Mixtral 8x7b). | |
| **Embeddings** | OpenAI Embeddings | Used to embed the policy text splits into vectors. | |
| **Data Handling** | PyPDF Loader | Used to parse and load data from the PDF document. | |
| **Data Utility** | Faker | Used to generate fake employee data for contextual testing. | |
| **Tracing** | Langsmith | Used for tracking and tracing application runnables and observing the chain's operation. | |

## 🏗️ Architecture Overview (RAG Chain)

The assistant uses a LangChain Expression Language (LCEL) chain, initialized in the `Assistant` class.

1.  **Input:** The chain takes the user query (user input) as the primary input.
2.  **Parallel Context Retrieval (Runnable Parallel):** The input triggers multiple parallel operations to gather context:
    *   **Policy Retrieval:** The `vectorStore.as_retriever()` fetches relevant documents (retrieved policy information) from the Chroma vector store based on the user query.
    *   **Employee Context:** A lambda function returns the current `employee information` (user data).
    *   **Conversation History:** A lambda function returns the entire `conversation history` (managed via Streamlit session state).
    *   **User Input:** A `runnable pass-through` ensures the original user query is included.
3.  **Prompt Construction:** The gathered context (System Prompt, retrieved policy information, employee information, conversation history, and user input) is combined using a `ChatPromptTemplate`.
4.  **Generation:** The resulting prompt is sent to the Language Model (LLM), such as Chat Grok.
5.  **Output:** A `StringOutputParser` processes the LLM's response, extracting the text content. The `getResponse` method streams this output.

## ⚙️ Setup and Installation

### Prerequisites

*   Python environment (Conda is suggested for initialization/activation, e.g., `conda activate umbrella four`).
*   API keys for OpenAI (for embeddings) and Grok (for LLM).
*   API keys for Langsmith tracing (optional, but recommended).

### Step 1: Clone the Repository

The starting point for this exercise is available in the master branch of the repository (link provided in the video description), and the complete solution is under the solution tab.

### Step 2: Install Requirements

Ensure you install all necessary packages, which include `langchain` (v0.3.1), `streamlit` (v1.38), `pypdf`, `faker`, and specific LangChain integrations (OpenAI, Grok, Chroma).

```bash
pip install -r requirements.txt
```
*(Note: A specific `requirements.txt` file is used to ensure matching package versions.)*

### Step 3: Configure Environment Variables

Create a `.env` file in the root directory and populate it with your necessary API keys.

| Variable | Purpose | Notes |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | Required for OpenAI embeddings. | |
| `GROK_API_KEY` | Required for using Grok LLMs (e.g., LAMA 3.1 8B). | |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable Langsmith tracing. | |
| `LANGCHAIN_ENDPOINT` | Langsmith endpoint URL. | |
| `LANGCHAIN_API_KEY` | Your Langsmith API key. | |
| `LANGCHAIN_PROJECT` | Define the project name (e.g., `UmbrellaSolution`). | |

Load these variables using `load_dotenv` at the start of the application.

### Step 4: Data Preparation

The required policy document, `umbrella_corp_policies.pdf`, should be located in the `data/` directory.

### Step 5: Run the Application

The application is run via Streamlit by executing `app.py`:

```bash
streamlit run app.py
```
This will initialize the application, load the PDF data, parse and split the documents (e.g., into chunks of 2,000 characters with 200 overlap, although 4,000 was tested), vectorize them using OpenAI embeddings, and store them in the Chroma vector store. The application uses Streamlit's caching features (`@st.cache_data` and `@st.cache_resource`) to prevent repeated expensive operations like data generation or vector store initialization across runs.