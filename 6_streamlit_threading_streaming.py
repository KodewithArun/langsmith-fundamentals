"""
6_streamlit_threading_streaming.py

A single-file Streamlit chatbot that combines the two ideas from the
campusx "chatbot-in-langgraph" repo:

  1. Threading  -> a LangGraph checkpointer + a `thread_id` per conversation,
                   so every chat keeps its own memory.
  2. Streaming  -> `chatbot.stream(..., stream_mode="messages")` +
                   `st.write_stream(...)`, so tokens appear as they arrive.

Everything is traced to LangSmith. The `thread_id`, `session_id`, turn number
and model are attached as trace metadata, so each Streamlit turn can be
filtered and inspected in the LangSmith UI.

Run:
    streamlit run 6_streamlit_threading_streaming.py
"""

import os
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv

# Load .env first, then pin this app to its own LangSmith project BEFORE
# importing LangGraph/LangChain so the tracer picks the right project up.
load_dotenv()

PROJECT_NAME = "langgraph-streamlit-threading-streaming"
os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME
os.environ["LANGSMITH_PROJECT"] = PROJECT_NAME

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

APP_NAME = "streamlit-langgraph-threading-streaming"
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Use Markdown when it makes the "
    "answer easier to read."
)

# **************************************** LangGraph backend ************************

llm = ChatGroq(model=MODEL_NAME, temperature=0.7)


class ChatState(TypedDict):
    """Graph state: only the message list, merged with LangGraph's reducer."""

    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState) -> dict:
    """Call the LLM with the system prompt plus the full thread history."""
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_chatbot():
    """Compile the graph with a checkpointer (this is what enables threading)."""
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    # InMemorySaver keeps thread state for the lifetime of the server process.
    # Swap in SqliteSaver if you want chats to survive a server restart.
    checkpointer = InMemorySaver()

    # `name` becomes the root run name in the LangSmith trace.
    return graph.compile(
        checkpointer=checkpointer,
        name="threaded_streaming_chatbot",
    )


@st.cache_resource(show_spinner="Building the LangGraph...")
def get_chatbot():
    """Cache the compiled graph so Streamlit reruns do not wipe thread memory."""
    return build_chatbot()


# **************************************** helpers *********************************

def new_thread_id() -> str:
    return str(uuid.uuid4())


def get_thread_config(thread_id: str, turn: int, mode: str = "streaming") -> dict:
    """Build the RunnableConfig used by every graph call.

    LangSmith reads `run_name`, `tags` and `metadata` from here, which is how
    the thread/turn context ends up attached to the root trace.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"chat_turn_{turn}",
        "tags": ["streamlit", "langgraph", "threading", "streaming"],
        "metadata": {
            "app": APP_NAME,
            "ui": "streamlit",
            "mode": mode,
            "model": MODEL_NAME,
            "thread_id": thread_id,
            "session_id": st.session_state["session_id"],
            "turn": turn,
        },
    }


def load_messages(thread_id: str) -> list[BaseMessage]:
    """Read a thread's message history from the checkpointer."""
    snapshot = get_chatbot().get_state(
        {"configurable": {"thread_id": thread_id}}
    )
    return snapshot.values.get("messages", []) or []


def message_text(message: BaseMessage) -> str:
    """Normalise string or block-list message content into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def make_thread_title(text: str, limit: int = 30) -> str:
    title = " ".join(text.split())
    if not title:
        return "New conversation"
    return title if len(title) <= limit else title[:limit].rstrip() + "…"


def get_thread_title(thread_id: str) -> str:
    """Use the stored title, otherwise derive it from the first user message."""
    titles = st.session_state["thread_titles"]
    if thread_id in titles:
        return titles[thread_id]

    for message in load_messages(thread_id):
        if isinstance(message, HumanMessage):
            titles[thread_id] = make_thread_title(message_text(message))
            return titles[thread_id]

    return "New conversation"


def init_session_state() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("threads", [])
    st.session_state.setdefault("thread_titles", {})

    if "thread_id" not in st.session_state:
        thread_id = new_thread_id()
        st.session_state["thread_id"] = thread_id
        st.session_state["threads"].append(thread_id)
    elif st.session_state["thread_id"] not in st.session_state["threads"]:
        st.session_state["threads"].append(st.session_state["thread_id"])


def start_new_chat() -> None:
    thread_id = new_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["threads"].append(thread_id)
    st.session_state["thread_titles"][thread_id] = "New conversation"


def token_stream(user_input: str, config: dict):
    """Yield only assistant text chunks for `st.write_stream`."""
    for chunk, _metadata in get_chatbot().stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk):
            text = message_text(chunk)
            if text:
                yield text


def render_history(messages: list[BaseMessage]) -> None:
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue

        text = message_text(message)
        if not text:
            continue

        with st.chat_message(role):
            st.markdown(text)


def tracing_enabled() -> bool:
    value = os.getenv("LANGCHAIN_TRACING_V2") or os.getenv("LANGSMITH_TRACING") or ""
    return value.strip().lower() in {"true", "1", "yes", "on"}


# **************************************** page setup ******************************

st.set_page_config(
    page_title="LangGraph Threaded Chat",
    page_icon="💬",
    layout="wide",
)

init_session_state()
chatbot = get_chatbot()

# **************************************** sidebar *********************************

with st.sidebar:
    st.title("💬 LangGraph Chat")
    st.caption(f"Model: `{MODEL_NAME}`")

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.divider()
    st.subheader("Conversations")

    for thread_id in reversed(st.session_state["threads"]):
        is_active = thread_id == st.session_state["thread_id"]
        label = ("🟢 " if is_active else "💬 ") + get_thread_title(thread_id)
        if st.button(
            label,
            key=f"thread-{thread_id}",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state["thread_id"] = thread_id
            st.rerun()

    st.divider()
    with st.expander("🔎 Trace details", expanded=False):
        st.code(
            f"project:    {PROJECT_NAME}\n"
            f"thread_id:  {st.session_state['thread_id']}\n"
            f"session_id: {st.session_state['session_id']}\n"
            f"turns:      {sum(isinstance(m, HumanMessage) for m in load_messages(st.session_state['thread_id']))}",
            language="text",
        )
        if tracing_enabled():
            st.success("LangSmith tracing is ON")
        else:
            st.warning(
                "LangSmith tracing is OFF. Set LANGCHAIN_TRACING_V2=true "
                "and LANGCHAIN_API_KEY in .env."
            )
        st.link_button("Open LangSmith", "https://smith.langchain.com")

    st.caption(
        "Each turn is a LangSmith trace. Use `thread_id` in the trace "
        "metadata to follow a full conversation."
    )

# **************************************** main UI *********************************

st.title("LangGraph: threading + streaming")
st.caption(
    "Each conversation has its own `thread_id`. Replies stream token by token, "
    "and every run is traced in LangSmith."
)

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to `.env` and reload the app.")
    st.stop()

current_thread = st.session_state["thread_id"]
history = load_messages(current_thread)

render_history(history)

user_input = st.chat_input("Type your message...")

if user_input:
    # Show the user's turn immediately. The assistant reply is streamed below.
    with st.chat_message("user"):
        st.markdown(user_input)

    turn = sum(isinstance(m, HumanMessage) for m in history) + 1
    if st.session_state["thread_titles"].get(current_thread) in (None, "New conversation"):
        st.session_state["thread_titles"][current_thread] = make_thread_title(user_input)
    config = get_thread_config(current_thread, turn, mode="streaming")

    with st.chat_message("assistant"):
        try:
            st.write_stream(token_stream(user_input, config))
        except Exception as exc:  # surface API/config errors in the UI
            st.error(f"Request failed: {exc}")

    # Re-render from the checkpointer so the sidebar title, trace details and
    # history all reflect the state that was just persisted.
    st.rerun()
