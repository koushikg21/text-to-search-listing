import html
import json
import logging
import streamlit as st
from setup import build_listing_url, call_mcp_tool, extract_links
from test import parse_filters_with_langgraph
import pandas as pd

st.set_page_config(page_title="Text2Stay Agent", page_icon="🏠", layout="wide")
st.caption("An experimental agent that turns natural language into structured Airbnb search filters using MCP + tool calling.")

logging.warning(
    "EXPERIMENTAL APP: Not production-ready. "
    "Uses mcp-server-airbnb with ignore_robots enabled."
)
st.warning(
    "⚠️ Experimental Demo – Not for Production Use\n\n"
    "This application is a personal prototype built for educational and research purposes only. "
    "It is not affiliated with, endorsed by, or supported by Airbnb. "
)

st.markdown(
    """
    <style>
    /* Reduce dataframe font size */
    .stDataFrame div[data-testid="stDataFrame"] {
        font-size: 13px;
    }

    .listing-card {
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 16px;
        padding: 16px;
        background: #fff;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
        min-height: 220px;
    }

    .listing-card h3 {
        font-size: 18px;
        margin: 0 0 8px 0;
    }

    .listing-meta {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 10px 0 8px 0;
        font-size: 13px;
        color: #2e2e2e;
    }

    .listing-pill {
        background: #f3f4f6;
        border-radius: 999px;
        padding: 4px 10px;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 6px 0 12px 0;
    }

    .summary-card {
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 14px;
        padding: 12px 14px;
        background: #f8fafc;
    }

    .summary-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
    }

    .summary-value {
        font-size: 16px;
        font-weight: 600;
        margin-top: 4px;
        color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Chat state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "search_requested" not in st.session_state:
    st.session_state.search_requested = False


def add_chat_message():
    user_text = st.session_state.get("chat_input", "").strip()
    if not user_text:
        return
    st.session_state.query_history.append(user_text)
    st.session_state.chat_messages = [
        {"role": "user", "text": entry} for entry in st.session_state.query_history
    ]
    st.session_state.search_requested = True
    st.session_state.chat_input = ""


def clear_chat_input():
    st.session_state.chat_input = ""
    st.session_state.query_history = []
    st.session_state.chat_messages = []
    st.session_state.search_requested = False


with st.sidebar:
    st.header("Ask to search")
    chat_container = st.container()
    st.text_area(
        "Describe the accommodation you are looking for",
        key="chat_input",
        height=160,
    )
    if st.session_state.query_history:
        combined_query = " ".join(st.session_state.query_history)
        chat_container.markdown(f"**Current query:** {combined_query}")
    else:
        chat_container.caption("Describe what you want (e.g., '2 adults in 94103 under $300 Oct 12-15').")

   
    action_cols = st.columns([1, 1])
    action_cols[0].button("Search", type="primary", on_click=add_chat_message)
    action_cols[1].button("Clear", on_click=clear_chat_input)
    st.divider()

    ignore_robots = True
    agent_filters: dict = {}
    agent_question: str | None = None
    agent_trace: list[str] = []
    latest_input = ""
    if st.session_state.query_history:
        latest_input = " ".join(st.session_state.query_history)
        try:
            agent_result = parse_filters_with_langgraph(latest_input)
            agent_filters = agent_result.filters
            agent_question = agent_result.question
           # agent_trace = agent_result.get("tool_trace", []) or []
        except Exception as exc:
            st.warning(f"LLM parser error: {exc}")

    if agent_filters:
        st.caption("Agent-parsed filters from chat")
        st.json(agent_filters)
    elif st.session_state.query_history:
        st.caption("Agent did not return any filters.")

    if agent_question:
        st.info(f"Agent question: {agent_question}")
    if agent_trace:
        st.caption("Tool calls")


if st.session_state.search_requested:
    st.session_state.search_requested = False
    args: dict = dict(agent_filters) if agent_filters else {}
    if agent_question:
        st.warning(agent_question)
        st.stop()

    if not args.get("location"):
        st.error("Add a location in your chat message so the agent can search.")
        st.stop()

#    mcp_args = {k: v for k, v in args.items() if k != "minRating"}
#    st.write(mcp_args)
    with st.spinner("Starting MCP server via npx and calling the tool..."):
        response = call_mcp_tool("airbnb_search", args, ignore_robots)
    raw_text = response["content"][0]["text"]
    data = json.loads(raw_text)
    #st.write(num_listings = len(response["searchResults"]))
    parsed = extract_links(response)
    if parsed:
        listings = parsed.get("listings") or []
        min_rating = args.get("minRating")
        min_rating_value = None
        if min_rating is not None:
            try:
                min_rating_value = float(min_rating)
            except (TypeError, ValueError):
                min_rating_value = None
        if min_rating_value is not None:
            listings = [
                item for item in listings
                if item.get("rating_value") is not None and item.get("rating_value") >= min_rating_value
            ]
        if listings:
            location_text = args.get("location", "N/A")
            date_text = f"{args.get('checkin', 'Any')} → {args.get('checkout', 'Any')}"
            price_min = args.get("minPrice")
            price_max = args.get("maxPrice")
            if price_min or price_max:
                price_text = f"${price_min or 'Any'}–${price_max or 'Any'}"
            else:
                price_text = "Any"
            st.markdown(
                f"""
                <div class="summary-grid">
                    <div class="summary-card">
                        <div class="summary-label">Total Results</div>
                        <div class="summary-value">{len(listings)}</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Location</div>
                        <div class="summary-value">{location_text}</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Dates</div>
                        <div class="summary-value">{date_text}</div>
                    </div>
                    <div class="summary-card">
                        <div class="summary-label">Price Filter</div>
                        <div class="summary-value">{price_text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        rows = []

        st.subheader("Search results")

        cols_per_row = 3
        for idx in range(0, len(listings), cols_per_row):
            row = listings[idx: idx + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, listing in zip(cols, row):
                name = html.escape(listing.get("name") or "Listing")
                url = build_listing_url(listing.get("url"), args) or "#"
                url = html.escape(url, quote=True)
                rating = html.escape(listing.get("rating_text") or "No rating yet")
                price = html.escape(listing.get("price_text") or "Price unavailable")
                with col:
                    st.markdown(
                        f"""
                        <div class="listing-card">
                            <h3><a href="{url}" target="_blank">{name}</a></h3>
                            <div class="listing-meta">
                                <span class="listing-pill">⭐ {rating}</span>
                                <span class="listing-pill">💰 {price}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.write("")



        details_url = parsed.get("listing_details_url")
        if details_url:
            st.subheader("Listing details link")
            st.markdown(f"- [Listing details]({details_url})")
