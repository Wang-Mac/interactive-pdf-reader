from langchain_core.prompts import ChatPromptTemplate

QA_SYSTEM_TEMPLATE = """You answer questions about uploaded PDF documents.

Use only the provided context. If the context does not contain enough information,
say that you cannot determine the answer from the uploaded document.

Be concise and helpful. Cite the PDF source and page number when available.
Do not invent facts or sources.

Return only the final answer. Do not return raw source fragments, hidden reasoning,
analysis notes, or a list of instructions."""

QA_PROMPT_TEMPLATE = """Context:
{context}

Chat history:
{chat_history}

Question:
{question}

Answer in the same language as the question."""

FOLLOW_UP_PROMPT_TEMPLATE = """Given the chat history and the latest user question,
rewrite the question as a standalone search question.

Return only the rewritten question. Preserve names, concepts, and the original
language.

Chat history:
{chat_history}

Latest question:
{question}

Standalone question:"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_TEMPLATE),
        ("human", QA_PROMPT_TEMPLATE),
    ]
)

follow_up_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite the user's question for document retrieval."),
        ("human", FOLLOW_UP_PROMPT_TEMPLATE),
    ]
)
