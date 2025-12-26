import json
import logging
import re
import streamlit as st
from llm_tools import parse_filters_with_langgraph_with_trace, parse_filters_with_llm_no_tools
from test import AirbnbFilters

st.set_page_config(page_title="Airbnb MCP demo", page_icon="🏠", layout="wide")
st.markdown(
    """
    # FreeText → Filters
    """
)
st.info(
    "Turning natural language into structured search inputs using Agents\n\n"
    "Use the sidebar to describe a stay. This app shows how the agent selects tools, "
    "fills their inputs, and returns structured filters—without calling Airbnb."
)
logging.warning(
    "EXPERIMENTAL APP: Not production-ready. "
    "Uses mcp-server-airbnb with ignore_robots enabled."
)
st.divider()

# Chat state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "search_requested" not in st.session_state:
    st.session_state.search_requested = False


def add_chat_message():
    user_text = st.session_state.get("chat_input", "").strip()
    if not user_text:
        return
    st.session_state.chat_messages = [{"role": "user", "text": user_text}]
    st.session_state.search_requested = True


def clear_chat_input():
    st.session_state.chat_input = ""


def _naive_parse_filters(user_text: str) -> dict:
    text = user_text.strip()
    lower = text.lower()
    result: dict = {}

    loc_match = re.search(
        r"(?:in|around|near)\s+([a-z0-9 ,.-]+?)(?:\s+(?:for|with|under|between|from|to)\b|$)",
        lower,
    )
    if loc_match:
        result["location"] = loc_match.group(1).strip()

    adults_match = re.search(r"(?:family of|for)\s+(\d+)", lower)
    if not adults_match:
        adults_match = re.search(r"(\d+)\s+(?:adults|adult|guests|people)", lower)
    if adults_match:
        result["adults"] = int(adults_match.group(1))

    pets_match = re.search(r"(\d+)\s+pets?", lower)
    if pets_match:
        result["pets"] = int(pets_match.group(1))
    elif "pets" in lower:
        result["pets"] = 1

    max_price_match = re.search(r"(?:under|below|max)\s*\$?\s*(\d+)", lower)
    if max_price_match:
        result["maxPrice"] = int(max_price_match.group(1))

    min_price_match = re.search(r"(?:over|above|min|minimum)\s*\$?\s*(\d+)", lower)
    if min_price_match:
        result["minPrice"] = int(min_price_match.group(1))

    rating_match = re.search(
        r"(?:rating|rated|stars?)\s*(?:at least|>=|minimum|min)?\s*(\d+(?:\.\d+)?)",
        lower,
    )
    if rating_match:
        result["minRating"] = float(rating_match.group(1))

    date_match = re.search(r"(?:between|from)\s+([a-z0-9 ,]+)", lower)
    if date_match:
        result["date_text"] = date_match.group(1).strip()

    return result


with st.sidebar:
    st.header("Ask to search")
    st.radio(
        "Mode",
        ["Ask to search (tool calls enabled)", "Ask to search (without tool calls)"],
        index=0,
        key="parse_mode",
    )
    chat_container = st.container()
    if st.session_state.chat_messages:
        for msg in st.session_state.chat_messages:
            chat_container.markdown(f"**You:** {msg['text']}")
    else:
        chat_container.caption("Describe what you want (e.g., '2 adults in 94103 under $300 Oct 12-15').")

    st.text_area(
        "Describe the accommodation you are looking for",
        key="chat_input",
        height=120,
        disabled=bool(st.session_state.get("chat_input", "").strip()),
    )
    action_cols = st.columns([1, 1])
    action_cols[0].button("Search", type="primary", on_click=add_chat_message)
    action_cols[1].button("Clear", on_click=clear_chat_input)
    st.divider()

    agent_filters: dict = {}
    agent_question: str | None = None
    agent_trace: list[dict] = []
    latest_input = ""
    if st.session_state.chat_messages:
        latest_input = st.session_state.chat_messages[-1].get("text", "")
        st.session_state.latest_input = latest_input
        if st.session_state.get("parse_mode") == "Ask to search (tool calls enabled)":
            try:
                agent_result, agent_trace = parse_filters_with_langgraph_with_trace(latest_input)
                agent_filters = agent_result.filters
                agent_question = agent_result.question
            except Exception as exc:
                st.warning(f"LLM parser error: {exc}")
        else:
            try:
                agent_result = parse_filters_with_llm_no_tools(latest_input)
                agent_filters = agent_result.filters
                agent_question = agent_result.question
            except Exception as exc:
                st.warning(f"LLM parser error: {exc}")

    st.divider()
    if agent_filters:
        st.caption("Agent-parsed filters from chat")
    elif st.session_state.chat_messages:
        st.caption("Agent did not return any filters.")

    if agent_question:
        st.info(f"Agent question: {agent_question}")
if st.session_state.search_requested:
    st.session_state.search_requested = False
    args: dict = dict(agent_filters) if agent_filters else {}
    if agent_question:
        st.warning(agent_question)
        st.stop()

    if st.session_state.get("parse_mode") == "Ask to search (tool calls enabled)" and not args.get("location"):
        st.error("Add a location in your chat message so the agent can search.")
        st.stop()

    if hasattr(AirbnbFilters, "model_fields"):
        fields = AirbnbFilters.model_fields
    else:
        fields = AirbnbFilters.__fields__
    field_names = ", ".join(f"`{name}`" for name in fields.keys())
    st.markdown(f"**Airbnb MCP filter fields:** {field_names}")

    if st.session_state.get("parse_mode") == "Ask to search (tool calls enabled)":
        tools = ["resolve_place_id_tool", "resolve_date_range_tool"]
        invoked = {name: False for name in tools}
        for item in agent_trace:
            name = item.get("name")
            if name in invoked:
                invoked[name] = True

        st.markdown("**Tools used by the agent:**")
        for name in tools:
            status = "YES" if invoked.get(name) else "NO"
            st.markdown(f"- `{name}` invoked: {status}")

    def _indent_text(text: str, spaces: int = 2) -> str:
        pad = " " * spaces
        return "\n".join(f"{pad}{line}" for line in text.splitlines() or [""])

    def _format_output(output: object) -> tuple[object, bool]:
        if output is None:
            return None, False
        if isinstance(output, (dict, list)):
            return output, True
        if isinstance(output, str):
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                return output, False
            if isinstance(parsed, (dict, list)):
                return parsed, True
            return output, False
        return str(output), False

    if st.session_state.get("parse_mode") == "Ask to search (tool calls enabled)":
        if agent_trace:
            st.markdown("**Tool inputs and outputs:**")
            for idx, item in enumerate(agent_trace, start=1):
                tool_name = item.get("name") or f"tool_{idx}"
                with st.expander(f"{idx}. {tool_name}"):
                    st.markdown("**Input:**")
                    st.json(item.get("input") or {})
                    st.markdown("**Output:**")
                    output, is_json = _format_output(item.get("output"))
                    if output is None:
                        st.caption("No output captured.")
                    elif is_json:
                        st.json(output)
                    else:
                        st.text(_indent_text(str(output)))
        else:
            st.caption("No tool calls captured for this request.")

    st.markdown("**Parsed filters (tool args):**")
    st.json(args)


st.divider()
st.warning(
    "⚠️ Experimental Demo – Not for Production Use\n\n"
    "This application is a personal prototype built for educational and research purposes only. "
    "It is not affiliated with, endorsed by, or supported by Airbnb. "
    "Data access may bypass robots.txt."
)
