from __future__ import annotations

import json
import logging
import re
import warnings
from typing import Any

import yt_dlp
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pytube import Search
from youtube_transcript_api import YouTubeTranscriptApi


warnings.filterwarnings("ignore")

pytube_logger = logging.getLogger("pytube")
pytube_logger.setLevel(logging.ERROR)

yt_dlp_logger = logging.getLogger("yt_dlp")
yt_dlp_logger.setLevel(logging.ERROR)


@tool
def extract_video_id(url: str) -> str:
    """
    Extract the 11-character YouTube video ID from a URL.
    """
    pattern = r"(?:v=|be/|embed/|shorts/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetch the transcript of a YouTube video by video ID.
    """
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=[language])
        return " ".join(snippet.text for snippet in transcript.snippets)
    except Exception as exc:
        return f"Error: {exc}"


@tool
def search_youtube(query: str) -> list[dict[str, str]] | str:
    """
    Search YouTube for videos matching the query.
    """
    try:
        results = Search(query).results
        return [
            {
                "title": video.title,
                "video_id": video.video_id,
                "url": f"https://youtu.be/{video.video_id}",
            }
            for video in results
        ]
    except Exception as exc:
        return f"Error: {exc}"


@tool
def get_full_metadata(url: str) -> dict[str, Any]:
    """
    Extract YouTube metadata from a video URL.
    """
    with yt_dlp.YoutubeDL({"quiet": True, "logger": yt_dlp_logger}) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "title": info.get("title"),
            "views": info.get("view_count"),
            "duration": info.get("duration"),
            "channel": info.get("uploader"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),
            "chapters": info.get("chapters", []),
        }


@tool
def get_thumbnails(url: str) -> list[dict[str, Any]]:
    """
    Get YouTube thumbnails for a video URL.
    """
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "logger": yt_dlp_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            thumbnails: list[dict[str, Any]] = []
            for thumbnail in info.get("thumbnails", []):
                if "url" not in thumbnail:
                    continue
                thumbnails.append(
                    {
                        "url": thumbnail["url"],
                        "width": thumbnail.get("width"),
                        "height": thumbnail.get("height"),
                        "resolution": (
                            f"{thumbnail.get('width', '')}x{thumbnail.get('height', '')}"
                        ).strip("x"),
                    }
                )
            return thumbnails
    except Exception as exc:
        return [{"error": f"Failed to get thumbnails: {exc}"}]


def _build_chain(api_key: str):
    tools = [
        extract_video_id,
        fetch_transcript,
        search_youtube,
        get_full_metadata,
        get_thumbnails,
    ]
    llm = ChatOpenAI(model="gpt-4.1-nano", api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)
    tool_mapping = {tool_fn.name: tool_fn for tool_fn in tools}

    def execute_tool(tool_call: dict[str, Any]) -> ToolMessage:
        try:
            result = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
            content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as exc:
            content = f"Error: {exc}"
        return ToolMessage(content=content, tool_call_id=tool_call["id"])

    def process_tool_calls(messages: list[Any]) -> list[Any]:
        last_message = messages[-1]
        tool_messages = [
            execute_tool(tool_call) for tool_call in getattr(last_message, "tool_calls", [])
        ]
        updated_messages = messages + tool_messages
        next_ai_response = llm_with_tools.invoke(updated_messages)
        return updated_messages + [next_ai_response]

    def should_continue(messages: list[Any]) -> bool:
        last_message = messages[-1]
        return bool(getattr(last_message, "tool_calls", None))

    def recursive_chain(messages: list[Any]) -> list[Any]:
        if should_continue(messages):
            return recursive_chain(process_tool_calls(messages))
        return messages

    return (
        RunnableLambda(lambda x: [HumanMessage(content=x["query"])])
        | RunnableLambda(lambda messages: messages + [llm_with_tools.invoke(messages)])
        | RunnableLambda(recursive_chain)
    )


def run_youtube_explorer_query(query: str, api_key: str) -> dict[str, Any]:
    chain = _build_chain(api_key=api_key)
    response = chain.invoke({"query": query})
    final_message = response[-1]

    return {
        "answer": getattr(final_message, "content", ""),
        "toolCalls": [
            {
                "name": getattr(message, "name", ""),
                "content": getattr(message, "content", ""),
            }
            for message in response
            if isinstance(message, ToolMessage)
        ],
    }
