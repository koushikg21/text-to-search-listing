"""
Core MCP wiring helpers so the Streamlit UI in app.py can stay focused on UI/agent logic.
"""

import anyio
import json
import shlex
import shutil
from mcp import ClientSession, StdioServerParameters, stdio_client
import pgeocode

# Default command string used to start the Airbnb MCP server.
DEFAULT_MCP_COMMAND_STR = "npx -y @openbnb/mcp-server-airbnb"


def build_command(command_str: str, ignore_robots: bool) -> list[str]:
    """
    Convert the user-provided command string into an argv list, add ignore flag, and
    ensure the executable exists if it's a simple binary like npx.
    """
    parts = shlex.split(command_str.strip())
    if not parts:
        raise ValueError("Command cannot be empty")

    exe = parts[0]
    exe_path = shutil.which(exe)
    if exe_path is None and "/" not in exe and "\\" not in exe:
        # exe not on PATH and not a path; signal early
        raise FileNotFoundError(f"Executable '{exe}' not found in PATH")

    if ignore_robots:
        parts.append("--ignore-robots-txt")
    return parts


def resolve_zip(zip_code: str) -> str | None:
    """
    Convert a US ZIP code to a human-readable location string the Airbnb search accepts.
    Example: "94103" -> "San Francisco, CA 94103".
    """
    nomi = pgeocode.Nominatim("us")
    rec = nomi.query_postal_code(zip_code)
    if rec is None or rec.place_name is None or rec.state_name is None:
        return None
    return f"{rec.place_name}, {rec.state_name} {zip_code}"


async def _run_mcp_tool(tool_name: str, arguments: dict, ignore_robots: bool) -> dict:
    """
    Spawn the MCP server over stdio, initialize a session, and call the requested tool.
    """
    cmd = build_command(DEFAULT_MCP_COMMAND_STR, ignore_robots)
    server_params = StdioServerParameters(command=cmd[0], args=cmd[1:])

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            available = {tool.name for tool in tools.tools}

            if tool_name not in available:
                return {
                    "error": f"Tool '{tool_name}' is not exposed by the server",
                    "available_tools": sorted(available),
                }

            result = await session.call_tool(tool_name, arguments)
            return result.model_dump(mode="json")


def call_mcp_tool(tool_name: str, arguments: dict, ignore_robots: bool) -> dict:
    try:
        return anyio.run(_run_mcp_tool, tool_name, arguments, ignore_robots)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # Stream errors back to the UI
        return {"error": str(exc)}


def get_airbnb_page(base_arguments: dict, ignore_robots: bool, cursor=None):
    """
    Fetch a single page of Airbnb listings from the MCP server.
    """
    arguments = dict(base_arguments)
    if cursor:
        arguments["cursor"] = cursor  # only include if not first page

    response = call_mcp_tool("airbnb_search", arguments, ignore_robots=ignore_robots)

    # Safety check and unwrap structure
    if "content" in response and response["content"]:
        try:
            data = json.loads(response["content"][0]["text"])
            return data
        except Exception as e:
            raise ValueError(f"Failed to parse MCP response: {e}")
    else:
        raise ValueError("No valid content in MCP response")


def fetch_all_pages(base_arguments: dict, ignore_robots: bool):
    """
    Walk pagination using the provided base search arguments.
    """
    all_listings = []
    cursor = None
    page = 1

    while True:
        data = get_airbnb_page(base_arguments, ignore_robots, cursor)
        listings = data.get("searchResults", [])
        all_listings.extend(listings)

        pagination = data.get("paginationInfo", {})
        cursor = pagination.get("nextPageCursor")
        if not cursor:
            break
        page += 1

    return all_listings


def extract_links(resp: dict) -> dict:
    """
    Pull out structured listing info from the MCP tool response.
    """
    listings: list[dict] = []
    listing_details_url: str | None = None
    search_url: str | None = None
    for item in resp.get("content", []):
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            payload = json.loads(item.get("text", ""))
        except Exception:
            continue

        # Search results
        search_url = search_url or payload.get("searchUrl") or payload.get("search_url")
        for result in payload.get("searchResults", []) or []:
            if not isinstance(result, dict):
                continue
            url = result.get("url")
            if not url:
                continue
            structured = result.get("structuredContent")
            primary_line = None
            secondary_line = None
            if isinstance(structured, dict):
                primary = structured.get("primaryLine")
                if isinstance(primary, dict):
                    primary_line = primary.get("body")
                secondary = structured.get("secondaryLine")
                if isinstance(secondary, dict):
                    secondary_line = secondary.get("body")
            label = (
                result.get("description")
                or primary_line
                or url
            )
            listing = {
                "id": result.get("id"),
                "name": result.get("demandStayListing", {})
                .get("description", {})
                .get("name", {})
                .get("localizedStringWithTranslationPreference"),
                "label": label,
                "url": url,
                "price_text": result.get("structuredDisplayPrice", {})
                .get("primaryLine", {})
                .get("accessibilityLabel"),
                "price_details": result.get("structuredDisplayPrice", {})
                .get("explanationData", {})
                .get("priceDetails"),
                "rating_text": result.get("avgRatingA11yLabel"),
                "rating_value": _extract_rating(result.get("avgRatingA11yLabel")),
                "primary_line": primary_line,
                "secondary_line": secondary_line,
                "badge": result.get("badges"),
            }
            listings.append(listing)

        # Listing details
        listing_url = payload.get("listingUrl") or payload.get("listing_url")
        if listing_url:
            listing_details_url = listing_url

    return {
        "listings": listings,
        "listing_details_url": listing_details_url,
        "search_url": search_url,
        "result_count": len(listings),
    }


def _extract_rating(text: str | None) -> float | None:
    if not text:
        return None
    import re

    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def build_listing_url(base_url: str | None, args: dict | None = None) -> str | None:
    """
    Append check-in/out and guest params to a listing URL if provided.
    """
    if not base_url:
        return None
    from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

    parts = urlparse(base_url)
    query = dict(parse_qsl(parts.query))
    params = args or {}
    if params.get("checkin"):
        query["check_in"] = params["checkin"]
    if params.get("checkout"):
        query["check_out"] = params["checkout"]
    if params.get("adults"):
        query["adults"] = str(params["adults"])
    if params.get("children"):
        query["children"] = str(params["children"])
    if params.get("infants"):
        query["infants"] = str(params["infants"])
    if params.get("pets"):
        query["pets"] = str(params["pets"])
    return urlunparse(parts._replace(query=urlencode(query)))
