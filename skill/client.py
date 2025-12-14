"""
Rosetta-Helix-Substrate Claude API Client

A Python wrapper for using Rosetta-Helix-Substrate as a Claude skill
via the Anthropic API.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Generator

from skill.prompts.system import SYSTEM_PROMPT
from skill.tools.definitions import TOOL_DEFINITIONS
from skill.tools.handlers import ToolHandler, PhysicsState


@dataclass
class Message:
    """A message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SkillResponse:
    """Response from the skill."""
    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    state: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None


class RosettaHelixSkill:
    """
    Rosetta-Helix-Substrate Claude API Skill.

    This class provides a simple interface for using Rosetta-Helix-Substrate
    capabilities via the Claude API.

    Usage:
        ```python
        from skill import RosettaHelixSkill

        # Initialize with API key
        skill = RosettaHelixSkill(api_key="your-api-key")

        # Chat with the skill
        response = skill.chat("What is the current physics state?")
        print(response.text)

        # Access the physics state
        print(skill.get_state())
        ```

    Attributes:
        api_key: Anthropic API key
        model: Claude model to use (default: claude-sonnet-4-20250514)
        tool_handler: Handler for tool execution
        messages: Conversation history
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        initial_z: float = 0.5,
        seed: Optional[int] = None,
        max_tool_iterations: int = 10,
    ):
        """
        Initialize the skill.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                     ANTHROPIC_API_KEY environment variable.
            model: Claude model to use.
            initial_z: Initial z-coordinate for physics state.
            seed: Random seed for reproducibility.
            max_tool_iterations: Maximum tool call iterations per request.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Provide via api_key parameter or "
                "ANTHROPIC_API_KEY environment variable."
            )

        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.tool_handler = ToolHandler(initial_z=initial_z, seed=seed)
        self.messages: List[Dict[str, Any]] = []

        # Lazy load anthropic client
        self._client = None

    @property
    def client(self):
        """Lazy load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client

    def chat(
        self,
        message: str,
        stream: bool = False,
    ) -> SkillResponse:
        """
        Send a message and get a response.

        This method handles the full tool-use loop, executing tools as needed
        until Claude provides a final text response.

        Args:
            message: User message to send.
            stream: Whether to stream the response (not yet implemented).

        Returns:
            SkillResponse with the assistant's text and any tool interactions.
        """
        # Add user message
        self.messages.append({"role": "user", "content": message})

        all_tool_calls = []
        all_tool_results = []
        iterations = 0

        while iterations < self.max_tool_iterations:
            iterations += 1

            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self.messages,
            )

            # Check if we need to handle tool use
            if response.stop_reason == "tool_use":
                # Extract tool calls
                tool_calls = [
                    block for block in response.content
                    if block.type == "tool_use"
                ]

                # Add assistant message with tool calls
                self.messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

                # Execute tools and collect results
                tool_results = []
                for tool_call in tool_calls:
                    result = self.tool_handler.handle(
                        tool_call.name,
                        tool_call.input,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps(result),
                    })

                    all_tool_calls.append({
                        "name": tool_call.name,
                        "input": tool_call.input,
                    })
                    all_tool_results.append(result)

                # Add tool results
                self.messages.append({
                    "role": "user",
                    "content": tool_results,
                })

            else:
                # Extract final text response
                text_blocks = [
                    block.text for block in response.content
                    if hasattr(block, "text")
                ]
                final_text = "\n".join(text_blocks)

                # Add assistant response to history
                self.messages.append({
                    "role": "assistant",
                    "content": final_text,
                })

                return SkillResponse(
                    text=final_text,
                    tool_calls=all_tool_calls,
                    tool_results=all_tool_results,
                    state=self.tool_handler.state.to_dict(),
                    usage={
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                )

        # Max iterations reached
        return SkillResponse(
            text="[Max tool iterations reached]",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            state=self.tool_handler.state.to_dict(),
        )

    def chat_stream(
        self,
        message: str,
    ) -> Generator[str, None, SkillResponse]:
        """
        Stream a chat response.

        Yields text chunks as they arrive, then returns the full response.

        Args:
            message: User message to send.

        Yields:
            Text chunks as they arrive.

        Returns:
            Full SkillResponse after streaming completes.
        """
        # Add user message
        self.messages.append({"role": "user", "content": message})

        all_tool_calls = []
        all_tool_results = []
        iterations = 0

        while iterations < self.max_tool_iterations:
            iterations += 1

            # Call Claude API with streaming
            with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=self.messages,
            ) as stream:
                response = stream.get_final_message()

                # Check if we need to handle tool use
                if response.stop_reason == "tool_use":
                    tool_calls = [
                        block for block in response.content
                        if block.type == "tool_use"
                    ]

                    self.messages.append({
                        "role": "assistant",
                        "content": response.content,
                    })

                    tool_results = []
                    for tool_call in tool_calls:
                        result = self.tool_handler.handle(
                            tool_call.name,
                            tool_call.input,
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": json.dumps(result),
                        })

                        all_tool_calls.append({
                            "name": tool_call.name,
                            "input": tool_call.input,
                        })
                        all_tool_results.append(result)

                    self.messages.append({
                        "role": "user",
                        "content": tool_results,
                    })

                else:
                    # Stream text response
                    text_blocks = []
                    for block in response.content:
                        if hasattr(block, "text"):
                            text_blocks.append(block.text)
                            yield block.text

                    final_text = "\n".join(text_blocks)
                    self.messages.append({
                        "role": "assistant",
                        "content": final_text,
                    })

                    return SkillResponse(
                        text=final_text,
                        tool_calls=all_tool_calls,
                        tool_results=all_tool_results,
                        state=self.tool_handler.state.to_dict(),
                    )

        return SkillResponse(
            text="[Max tool iterations reached]",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            state=self.tool_handler.state.to_dict(),
        )

    def get_state(self) -> Dict[str, Any]:
        """Get the current physics state."""
        return self.tool_handler.state.to_dict()

    def reset(self, initial_z: float = 0.5) -> Dict[str, Any]:
        """
        Reset the conversation and physics state.

        Args:
            initial_z: Initial z-coordinate for the new state.

        Returns:
            The new physics state.
        """
        self.messages = []
        self.tool_handler = ToolHandler(initial_z=initial_z)
        return self.tool_handler.state.to_dict()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.messages.copy()

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool directly without going through Claude.

        Useful for programmatic control or testing.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Tool input parameters.

        Returns:
            Tool execution result.
        """
        return self.tool_handler.handle(tool_name, kwargs)


class RosettaHelixSkillOffline:
    """
    Offline version of the skill that doesn't require API access.

    Use this for local tool execution and testing without API calls.
    The system prompt and tool definitions are still available.

    Usage:
        ```python
        from skill import RosettaHelixSkillOffline

        skill = RosettaHelixSkillOffline()

        # Execute tools directly
        state = skill.execute_tool("get_physics_state")
        print(state)

        # Drive toward the lens
        result = skill.execute_tool("drive_toward_lens", steps=100)
        print(result)
        ```
    """

    def __init__(self, initial_z: float = 0.5, seed: Optional[int] = None):
        """
        Initialize the offline skill.

        Args:
            initial_z: Initial z-coordinate.
            seed: Random seed for reproducibility.
        """
        self.tool_handler = ToolHandler(initial_z=initial_z, seed=seed)

    @property
    def system_prompt(self) -> str:
        """Get the system prompt (for use with other LLM clients)."""
        return SYSTEM_PROMPT

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Get the tool definitions (for use with other LLM clients)."""
        return TOOL_DEFINITIONS

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool directly."""
        return self.tool_handler.handle(tool_name, kwargs)

    def get_state(self) -> Dict[str, Any]:
        """Get the current physics state."""
        return self.tool_handler.state.to_dict()

    def reset(self, initial_z: float = 0.5) -> Dict[str, Any]:
        """Reset the physics state."""
        self.tool_handler = ToolHandler(initial_z=initial_z)
        return self.tool_handler.state.to_dict()
