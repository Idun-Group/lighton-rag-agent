"""LightOn RAG agent powered by Idun Agent Engine and Gemini."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from idun_agent_engine.mcp import get_langchain_tools
from idun_agent_engine.prompts import get_prompt


def _prompt_text(prompt_id: str) -> str:
    prompt = get_prompt(prompt_id)
    if not prompt:
        raise RuntimeError(
            f"Prompt {prompt_id!r} not found. Make sure it is configured."
        )
    return prompt.content


planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
executor_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")


class InputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class GraphState(InputState):
    plan: str


class OutputState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def planner(state: GraphState):
    messages = [
        {"role": "system", "content": _prompt_text("plan-prompt")},
    ] + state["messages"]
    response = await planner_llm.ainvoke(messages)
    return {"plan": response.content}


async def executor(state: GraphState):
    tools = await get_langchain_tools()
    llm_with_tools = executor_llm.bind_tools(tools)
    plan_context = f"\n\nYour plan for this request:\n{state.get('plan', '')}"
    messages = [
        {"role": "system", "content": _prompt_text("system-prompt") + plan_context},
    ] + state["messages"]
    return {"messages": [await llm_with_tools.ainvoke(messages)]}


async def tool_node(state: GraphState):
    tools = await get_langchain_tools()
    node = ToolNode(tools)
    return await node.ainvoke(state)


def should_continue(state: GraphState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    g = StateGraph(GraphState, input=InputState, output=OutputState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("tools", tool_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "executor")
    return g


graph = build_graph()
