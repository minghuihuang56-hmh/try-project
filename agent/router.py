"""Agent 核心路由层 — Function Calling 驱动

流程：
  用户问题 → LLM 选工具 → 执行工具 → 结果回传 LLM → LLM 写结论 → 展示给用户
  （若 LLM 继续要工具 → 循环，直到 LLM 返回纯文本）

历史格式与 Streamlit session_state 兼容：
  history = [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "...", "fig": Figure|None},
  ]
  route() 返回 {"answer": ..., "fig": ..., "msg": {"role": "assistant", ...}}
"""

import json
import os
import re
from typing import Optional

import pandas as pd
from openai import OpenAI

from .prompts import AGENT_SYSTEM_PROMPT
from .tools import (
    get_schema_info,
    make_chart,
    run_python,
    run_sql,
)


# ============================================================
# 工具 JSON Schema（Function Calling 使用）
# ============================================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "在数据集上执行 SQLite 查询。用于精确计数、聚合统计、分组排名、筛选过滤等场景。"
            "表名固定为 data，字段用双引号包裹。结果最多返回 100 行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQLite SQL 查询语句。示例: SELECT \"hotel\", COUNT(*) as cnt FROM data GROUP BY \"hotel\"",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "执行 Python 代码进行数据分析。用于相关性分析、统计建模、复杂计算、特征工程等。"
            "pandas (pd)、numpy (np)、plotly.express (px)、plotly.graph_objects (go) 已预置。"
            "数据在 df 变量中。文本结果赋值给 result，图表赋值给 fig。",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python 代码。示例:\n"
                        "import pandas as pd\n"
                        "cancel_rate = df.groupby('hotel')['is_canceled'].mean()\n"
                        "result = f'取消率:\\n{cancel_rate}'",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": "生成数据可视化图表。支持 bar（柱状图）、line（折线图）、scatter（散点图）、"
            "histogram（直方图）、pie（饼图）。传入列名即可，不接受原始数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "scatter", "histogram", "pie"],
                        "description": "图表类型：bar 柱状图 / line 折线图 / scatter 散点图 / histogram 直方图 / pie 饼图",
                    },
                    "x_col": {
                        "type": "string",
                        "description": "X 轴列名（pie 图时作为分类列）",
                    },
                    "y_col": {
                        "type": "string",
                        "description": "Y 轴列名（数值列）。histogram 和 pie 图可省略。",
                    },
                    "title": {
                        "type": "string",
                        "description": "图表标题，用中文描述图表内容",
                    },
                    "color_col": {
                        "type": "string",
                        "description": "分组/颜色列名（可选）。用于按类别着色。",
                    },
                },
                "required": ["chart_type", "x_col", "title"],
                "additionalProperties": False,
            },
        },
    },
]


# ============================================================
# 工具名称 → 实际函数 映射
# ============================================================
TOOL_FUNCTIONS = {
    "run_sql": lambda args, df: run_sql(sql=args["sql"], df=df),
    "run_python": lambda args, df: run_python(code=args["code"], df=df),
    "make_chart": lambda args, df: make_chart(
        df=df,
        chart_type=args.get("chart_type"),
        x_col=args.get("x_col"),
        y_col=args.get("y_col"),
        title=args.get("title", ""),
        color_col=args.get("color_col"),
    ),
}


# ============================================================
# 主路由函数
# ============================================================

def route(
    question: str,
    df: pd.DataFrame,
    history: list,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Agent 核心路由：自然语言 → LLM 意图识别 → 工具执行 → 结果解读

    完整流程（while 循环，支持多步）：
      1. 将问题 + schema + 样本数据 + 历史包装发给 LLM（含 tool definitions）
      2. LLM 返回 tool_use → 执行对应工具 → 结果拼为 tool_result 再发回 LLM
      3. LLM 返回 text → 结束循环，返回最终回答

    Args:
        question: 用户自然语言问题
        df: 数据 DataFrame
        history: 历史消息 [{"role": "user"|"assistant", "content": str, "fig"?}]
        api_key: API Key（默认从环境变量读取）
        base_url: API 地址（默认从环境变量读取）
        model: 模型名（默认从环境变量读取）

    Returns:
        {
            "answer": str,          # 最终自然语言回答
            "fig": Figure | None,   # 图表对象（如有）
            "msg": {                # 可直接 append 到 st.session_state.messages
                "role": "assistant",
                "content": str,
                "fig": Figure | None,
            },
            "tool_calls": int,      # 本轮触发的工具调用次数
            "steps": list,          # 中间步骤列表 [{tool, args, sql?, code?, error?}]
        }
    """
    # ── 1. 客户端初始化 ────────────────────────────────────
    resolved_key = api_key or os.getenv("API_KEY", "")
    resolved_url = base_url or os.getenv("API_BASE_URL", "")
    resolved_model = model or os.getenv("MODEL_NAME", "deepseek-chat")

    if not resolved_key:
        # 没有 API Key → 走规则模式
        return _route_rule(question, df, history)

    client = OpenAI(api_key=resolved_key, base_url=resolved_url) if resolved_url else OpenAI(api_key=resolved_key)

    # ── 2. 构造上下文 ──────────────────────────────────────
    schema_info = get_schema_info(df)
    context_block = _build_context_block(df, schema_info)

    # 构建 messages（system + history + 当前问题）
    messages = [_build_system_message(context_block)]
    for msg in history:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg.get("content", "")})
    messages.append({"role": "user", "content": question})

    # ── 3. Agent 循环：直到 LLM 输出纯文本 ──────────────────
    final_answer = ""
    final_fig = None
    tool_call_count = 0
    steps = []  # 记录中间步骤，供前端展示
    max_iterations = 10  # 安全上限

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=resolved_model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=2000,
        )

        choice = response.choices[0]
        msg = choice.message

        # ── 情况 A：LLM 返回纯文本 → 结束 ──
        if msg.content and not msg.tool_calls:
            final_answer = msg.content
            break

        # ── 情况 B：LLM 返回 tool_use → 执行工具 → 结果回传 ──
        if msg.tool_calls:
            # 关键：assistant 消息必须包含 tool_calls，否则后续 tool 消息报错
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                tool_call_count += 1

                # 解析参数
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                # 执行工具
                if fn_name in TOOL_FUNCTIONS:
                    tool_result = TOOL_FUNCTIONS[fn_name](fn_args, df)
                else:
                    tool_result = {"status": "error", "message": f"未知工具: {fn_name}"}

                # 提取图表（如有）
                if tool_result.get("fig") is not None:
                    final_fig = tool_result["fig"]

                # 记录步骤（供前端展示中间过程）
                step_entry = {"tool": fn_name, "args": fn_args}
                if tool_result.get("status") == "error":
                    step_entry["error"] = tool_result.get("message", "执行失败")
                steps.append(step_entry)

                # 将工具结果格式化为文本，回传给 LLM
                result_text = _format_result_for_llm(fn_name, fn_args, tool_result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

            # 继续循环 → LLM 用 tool_result 决定下一步
            continue

        # ── 情况 C：既无 content 又无 tool_calls（极边缘情况） ──
        final_answer = msg.content or "抱歉，我无法分析这个问题。请换一种问法。"
        break

    else:
        # max_iterations 耗尽
        final_answer = "分析步骤过多，请简化你的问题或分步提问。"

    # ── 4. 返回 ────────────────────────────────────────────
    if not final_answer:
        final_answer = "分析完成，请查看上方结果。"

    msg_entry = {"role": "assistant", "content": final_answer}
    if final_fig is not None:
        msg_entry["fig"] = final_fig

    return {
        "answer": final_answer,
        "fig": final_fig,
        "msg": msg_entry,
        "tool_calls": tool_call_count,
        "steps": steps,
    }


# ============================================================
# 规则模式后备（无 API Key 时使用）
# ============================================================
def _route_rule(question: str, df: pd.DataFrame, history: list) -> dict:
    """无 API Key 时的关键词规则模式"""
    from .router_fallback import route_fallback
    return route_fallback(question, df, history)


# ============================================================
# 消息构建
# ============================================================

def _build_system_message(context_block: str) -> dict:
    """构造 system 消息：角色指令 + 数据上下文"""
    content = AGENT_SYSTEM_PROMPT + "\n\n" + context_block
    return {"role": "system", "content": content}


def _build_context_block(df: pd.DataFrame, schema_info: dict) -> str:
    """构建数据上下文：Schema + 样本数据"""
    lines = [
        "=" * 50,
        "📋 当前数据集信息",
        "=" * 50,
        f"行数: {schema_info['rows']:,}",
        f"列数: {schema_info['columns']}",
        "",
        "--- 列信息 ---",
    ]
    for col in schema_info["schema"]:
        lines.append(
            f"  • {col['name']} ({col['dtype']}): "
            f"非空 {col['non_null']}/{schema_info['rows']}, "
            f"唯一值 {col['unique']}, "
            f"示例 {col['sample_values']}"
        )

    lines.extend([
        "",
        "--- 前 5 行数据预览 ---",
        df.head(5).to_string(),
        "",
        "=" * 50,
    ])

    return "\n".join(lines)


# ============================================================
# 工具结果格式化（回传给 LLM）
# ============================================================

def _format_result_for_llm(fn_name: str, fn_args: dict, result: dict) -> str:
    """将工具执行结果格式化为文本，供 LLM 理解并生成解读"""
    status = result.get("status", "error")
    parts = [f"工具: {fn_name}", f"参数: {json.dumps(fn_args, ensure_ascii=False)}"]

    if status == "error":
        parts.append(f"状态: 执行失败")
        parts.append(f"错误信息: {result.get('message', result.get('error', '未知错误'))}")
        return "\n".join(parts)

    parts.append("状态: 执行成功")

    # SQL 结果
    if fn_name == "run_sql" and "data" in result:
        data = result["data"]
        if hasattr(data, "shape"):
            parts.append(f"返回 {data.shape[0]} 行 × {data.shape[1]} 列")
        if hasattr(data, "to_string"):
            parts.append(f"数据预览（最多 20 行）:\n{data.head(20).to_string()}")
        if hasattr(data, "shape") and data.shape[0] > 100:
            parts.append(f"（仅显示前 20 行，共 {data.shape[0]} 行——结果可能截断，提醒用户）")

    # Python 结果
    elif fn_name == "run_python":
        if result.get("result") is not None:
            result_str = str(result["result"])
            parts.append(f"计算结果:\n{result_str[:2000]}")
        if result.get("output"):
            parts.append(f"程序输出:\n{result['output'][:1000]}")
        if result.get("fig"):
            parts.append("📊 图表已生成（包含在前端展示）")

    # 图表结果
    elif fn_name == "make_chart":
        if result.get("fig"):
            chart_info = []
            chart_info.append(f"图表类型: {fn_args.get('chart_type', 'N/A')}")
            chart_info.append(f"X 轴: {fn_args.get('x_col', 'N/A')}")
            if fn_args.get("y_col"):
                chart_info.append(f"Y 轴: {fn_args['y_col']}")
            if fn_args.get("color_col"):
                chart_info.append(f"颜色分组: {fn_args['color_col']}")
            parts.append(f"📊 图表已生成: {' | '.join(chart_info)}")

    return "\n".join(parts)
