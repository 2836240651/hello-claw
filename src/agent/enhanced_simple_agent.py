"""增强版 SimpleAgent - 支持流式工具调用"""

import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator, TYPE_CHECKING, Union

from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.config import Config
from hello_agents.core.message import Message
from hello_agents.core.streaming import StreamEvent, StreamEventType

# 导入 HelloClaw 专用 LLM（支持流式工具调用）
from .enhanced_llm import EnhancedHelloAgentsLLM, StreamToolEventType

if TYPE_CHECKING:
    from hello_agents.tools.registry import ToolRegistry


class EnhancedSimpleAgent(SimpleAgent):
    """增强版 SimpleAgent，支持流式工具调用

    继承 hello_agents 的 SimpleAgent，增加：
    - 真正的流式工具调用（使用 EnhancedHelloAgentsLLM）
    - 工具调用状态的实时推送

    Note:
        推荐使用 EnhancedHelloAgentsLLM 以获得完整的流式工具调用支持。
        如果使用普通 HelloAgentsLLM，流式工具调用将回退到基类的非流式模式。
    """

    def __init__(
        self,
        name: str,
        llm: Union[HelloAgentsLLM, EnhancedHelloAgentsLLM],
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 10,
    ):
        """初始化 EnhancedSimpleAgent

        Args:
            name: Agent 名称
            llm: LLM 实例（推荐使用 EnhancedHelloAgentsLLM）
            system_prompt: 系统提示词
            config: 配置对象
            tool_registry: 工具注册表（可选）
            enable_tool_calling: 是否启用工具调用
            max_tool_iterations: 最大工具调用迭代次数
        """
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
            max_tool_iterations=max_tool_iterations,
        )

        # 检查是否支持流式工具调用
        self._supports_streaming_tools = isinstance(llm, EnhancedHelloAgentsLLM)

    def _build_messages(self, input_text: str) -> List[Dict[str, str]]:
        """构建发送给 LLM 的消息列表。

        MiniMax 的 OpenAI 兼容接口在多轮历史里对旧的 `tool` 消息兼容性较弱。
        `hello-agents` 基类会把历史中的 tool 消息直接带回去，但不会附带
        tool_call_id，导致服务端报 `tool result's tool id() not found`。

        这里对历史做一层清洗：
        - 保留 system / user / assistant 的纯文本内容
        - 跳过历史中的 tool 消息
        - 跳过没有文本内容的 assistant 占位消息
        当前轮次的 tool_calls / tool 结果仍由本类在运行时追加，不受影响。
        """
        messages: List[Dict[str, str]] = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        for msg in self._history:
            if msg.role not in ("user", "assistant"):
                continue

            if not msg.content:
                continue

            messages.append({
                "role": msg.role,
                "content": msg.content
            })

        messages.append({
            "role": "user",
            "content": input_text
        })

        return messages

    async def arun_stream_with_tools(
        self,
        input_text: str,
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """异步流式运行（支持工具调用）

        使用 EnhancedHelloAgentsLLM 的 astream_invoke_with_tools 方法实现优雅的流式工具调用。

        Args:
            input_text: 用户输入
            **kwargs: 其他参数

        Yields:
            StreamEvent: 流式事件
        """
        session_start_time = datetime.now()

        # 发送开始事件
        yield StreamEvent.create(
            StreamEventType.AGENT_START,
            self.name,
            input_text=input_text
        )

        print(f"\n🤖 {self.name} 开始处理问题（流式）: {input_text}")

        try:
            # 构建消息列表
            messages = self._build_messages(input_text)

            # 检查是否有工具
            if not self.enable_tool_calling or not self.tool_registry:
                # 纯对话模式，使用基类的方法
                async for event in self._stream_without_tools(messages, **kwargs):
                    yield event
                return

            # 检查 LLM 是否支持流式工具调用
            if not self._supports_streaming_tools:
                import warnings
                warnings.warn(
                    "当前 LLM 不支持流式工具调用，将使用非流式模式。"
                    "推荐使用 EnhancedHelloAgentsLLM 以获得更好的体验。",
                    UserWarning
                )
                # 回退到基类的非流式模式
                response = self.run(input_text, **kwargs)
                yield StreamEvent.create(
                    StreamEventType.AGENT_FINISH,
                    self.name,
                    result=response
                )
                return

            # === 流式工具调用模式 ===
            tool_schemas = self._build_tool_schemas()
            print(f"🔧 已启用工具调用，可用工具: {list(self.tool_registry._tools.keys())}")

            current_iteration = 0
            final_response = ""
            # 收集工具调用记录（用于存入会话）
            tool_call_records: List[Dict[str, Any]] = []

            while current_iteration < self.max_tool_iterations:
                current_iteration += 1

                # 发送步骤开始事件
                yield StreamEvent.create(
                    StreamEventType.STEP_START,
                    self.name,
                    step=current_iteration,
                    max_steps=self.max_tool_iterations
                )

                print(f"\n--- 第 {current_iteration} 轮 ---")
                print("💭 LLM 输出: ", end="", flush=True)

                # 使用 LLM 的流式工具调用方法
                try:
                    async for event in self.llm.astream_invoke_with_tools(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="auto",
                        **kwargs
                    ):
                        # 处理文本内容
                        if event.event_type == StreamToolEventType.CONTENT:
                            yield StreamEvent.create(
                                StreamEventType.LLM_CHUNK,
                                self.name,
                                chunk=event.content,
                                step=current_iteration
                            )
                            print(event.content, end="", flush=True)

                        # 工具调用开始（打印信息，不发送事件）
                        elif event.event_type == StreamToolEventType.TOOL_CALL_START:
                            pass  # 等工具调用完成后再发送事件

                    print()  # 换行

                except Exception as e:
                    error_msg = f"LLM 调用失败: {str(e)}"
                    print(f"\n❌ {error_msg}")
                    yield StreamEvent.create(
                        StreamEventType.ERROR,
                        self.name,
                        error=error_msg
                    )
                    break

                # 获取累积结果
                result = self.llm.get_last_stream_tool_result()
                if result is None:
                    break

                # 检查是否有工具调用
                complete_tool_calls = result.get_complete_tool_calls()

                # 无论是否有工具调用，都保存本轮的文本内容
                if result.content:
                    final_response = result.content

                if not complete_tool_calls:
                    # 没有工具调用，直接返回
                    if not final_response:
                        final_response = "抱歉，我无法回答这个问题。"
                    # 显示内容预览
                    preview = final_response[:100] + "..." if len(final_response) > 100 else final_response
                    print(f"💬 直接回复: {preview}")
                    break

                print(f"🔧 准备执行 {len(complete_tool_calls)} 个工具调用...")

                # 将助手消息添加到历史
                messages.append(result.to_assistant_message())

                # 执行所有工具调用
                for tc in complete_tool_calls:
                    tool_name = tc["name"]
                    tool_call_id = tc["id"]

                    try:
                        arguments = json.loads(tc["arguments"])
                    except json.JSONDecodeError as e:
                        print(f"❌ 工具参数解析失败: {e}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": f"错误：参数格式不正确 - {str(e)}"
                        })
                        continue

                    print(f"🎬 调用工具: {tool_name}({arguments})")

                    # 发送工具调用开始事件
                    yield StreamEvent.create(
                        StreamEventType.TOOL_CALL_START,
                        self.name,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        args=arguments
                    )

                    # 让出控制权，确保 SSE 发送 tool_start 事件
                    await asyncio.sleep(0)

                    # 执行工具
                    exec_result = self._execute_tool_call(tool_name, arguments)

                    # 截断显示
                    result_preview = exec_result[:200] + "..." if len(exec_result) > 200 else exec_result
                    if exec_result.startswith("❌"):
                        print(f"❌ 工具执行失败: {result_preview}")
                    else:
                        print(f"👀 观察: {result_preview}")

                    # 发送工具调用完成事件
                    yield StreamEvent.create(
                        StreamEventType.TOOL_CALL_FINISH,
                        self.name,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        result=exec_result
                    )

                    # 记录工具调用（用于存入会话）
                    tool_call_records.append({
                        "id": tool_call_id,
                        "name": tool_name,
                        "args": arguments,
                        "result": exec_result,
                        "status": "error" if exec_result.startswith("❌") else "done"
                    })

                    # 添加工具结果到消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": exec_result
                    })

                # 发送步骤完成事件
                yield StreamEvent.create(
                    StreamEventType.STEP_FINISH,
                    self.name,
                    step=current_iteration
                )

            # 如果超过最大迭代次数，获取最后一次回答
            if current_iteration >= self.max_tool_iterations and not final_response:
                print("⏰ 已达到最大迭代次数，获取最终回答...")

                try:
                    async for chunk in self.llm.astream_invoke(messages, **kwargs):
                        final_response += chunk
                        yield StreamEvent.create(
                            StreamEventType.LLM_CHUNK,
                            self.name,
                            chunk=chunk
                        )
                        print(chunk, end="", flush=True)
                    print()
                except Exception as e:
                    print(f"❌ 最终回答失败: {e}")
                    result = self.llm.get_last_stream_tool_result()
                    final_response = result.content if result else "抱歉，我无法回答这个问题。"

            # 保存到历史记录（按照 OpenAI 规范格式）
            self.add_message(Message(input_text, "user"))

            # 如果有工具调用，保存工具调用消息
            if tool_call_records:
                # 保存 assistant 消息（包含 tool_calls）
                tool_calls_for_message = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                        }
                    }
                    for tc in tool_call_records
                ]
                self.add_message(Message(
                    "",  # 工具调用时可能没有文本内容
                    "assistant",
                    metadata={"tool_calls": tool_calls_for_message}
                ))

                # 保存每个 tool 消息
                for tc in tool_call_records:
                    self.add_message(Message(
                        tc["result"],
                        "tool",
                        metadata={"tool_call_id": tc["id"]}
                    ))

            # 保存最终 assistant 回答
            if final_response:
                self.add_message(Message(final_response, "assistant"))

            duration = (datetime.now() - session_start_time).total_seconds()
            print(f"\n✅ 完成，耗时 {duration:.2f}s，共 {current_iteration} 轮")

            # 发送完成事件
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=final_response
            )

        except Exception as e:
            print(f"❌ Agent 执行失败: {e}")
            yield StreamEvent.create(
                StreamEventType.ERROR,
                self.name,
                error=str(e),
                error_type=type(e).__name__
            )
            # 不要 raise，确保流式响应正常结束
            # 发送完成事件以优雅结束
            yield StreamEvent.create(
                StreamEventType.AGENT_FINISH,
                self.name,
                result=""  # 空结果表示失败
            )

    async def _stream_without_tools(
        self,
        messages: List[Dict],
        **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        """纯对话模式（无工具调用）"""
        print("📝 纯对话模式（无工具调用）")

        full_response = ""
        async for chunk in self.llm.astream_invoke(messages, **kwargs):
            full_response += chunk
            yield StreamEvent.create(
                StreamEventType.LLM_CHUNK,
                self.name,
                chunk=chunk
            )
            print(chunk, end="", flush=True)

        print()

        # 保存历史
        self.add_message(Message(messages[-1]["content"], "user"))
        self.add_message(Message(full_response, "assistant"))

        print(f"💬 回复完成")

        yield StreamEvent.create(
            StreamEventType.AGENT_FINISH,
            self.name,
            result=full_response
        )
