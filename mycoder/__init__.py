"""MyCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from mycoder.agent import Agent
from mycoder.llm import LLM
from mycoder.config import Config
from mycoder.model_router import ModelRouter
from mycoder.tools import ALL_TOOLS

__all__ = ["Agent", "LLM", "Config", "ModelRouter", "ALL_TOOLS", "__version__"]
