"""An agent designed to hold a conversation in addition to using tools."""
from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple

from langchain.agents.agent import Agent
from langchain.agents.conversational.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool

from langchain.agents.conversational.base import ConversationalAgent

# flake8: noqa
PREFIX = """Leo is a large language model trained by MZ.

Leo is designed to be able to assist with answering health, illness, and medical related questions, and is able to both answer simple questions as well as provide in-depth explanations and discussions.
As a language model, Leo is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Leo is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses.
Additionally, Leo is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions.

Leo is only able to answer questions about health, illness, and medical related questions, but is otherwise able to otherwise make polite conversation.
When answering health, illness, and medical related questions, Leo should almost always use the provided tools.

Whether you need help with a specific question or just want to have a conversation about a particular topic, Leo is here to assist.

TOOLS:
------

Leo has access to the following tools:"""
FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```"""

SUFFIX = """Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""


class HealthConversationalAgent(ConversationalAgent):
    """An agent designed to hold a conversation in addition to using tools."""

    ai_prefix: str = "AI"

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "health-conversational-react-description"

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            ai_prefix: String to use before AI output.
            human_prefix: String to use before human output.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        format_instructions = format_instructions.format(
            tool_names=tool_names, ai_prefix=ai_prefix, human_prefix=human_prefix
        )
        template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        return PromptTemplate(template=template, input_variables=input_variables)

    @property
    def finish_tool_name(self) -> str:
        """Name of the tool to use to finish the chain."""
        return self.ai_prefix

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        if f"{self.ai_prefix}:" in llm_output:
            return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
        regex = r"Action: (.*?)\nAction Input: (.*)"
        match = re.search(regex, llm_output)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1)
        action_input = match.group(2)
        return action.strip(), action_input.strip(" ").strip('"')

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLLM,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        ai_prefix: str = "AI",
        human_prefix: str = "Human",
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Agent:
        """Construct an agent from an LLM and tools."""
        cls._validate_tools(tools)
        prompt = cls.create_prompt(
            tools,
            ai_prefix=ai_prefix,
            human_prefix=human_prefix,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        return cls(
            llm_chain=llm_chain, allowed_tools=tool_names, ai_prefix=ai_prefix, **kwargs
        )