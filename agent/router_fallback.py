"""规则模式后备 — 无 API Key 时使用关键词匹配路由"""

import pandas as pd

from .tools import (
    get_schema_info,
    get_summary_stats,
    make_chart,
    run_python,
    run_sql,
)


def route_fallback(question: str, df: pd.DataFrame, history: list) -> dict:
    """关键词规则模式路由"""
    schema_info = get_schema_info(df)

    # 1. 意图分类
    intent = _classify_intent_rule(question, schema_info)

    # 2. 执行
    if intent["tool"] == "multi_step":
        result = _handle_multi_step(question, schema_info, df)
    else:
        result = _handle_single_step(intent["tool"], question, schema_info, df)

    # 3. 回答
    answer = _generate_answer_rule(question, result)

    msg_entry = {"role": "assistant", "content": answer}
    if result.get("fig") is not None:
        msg_entry["fig"] = result["fig"]

    steps = [{"tool": "fallback", "args": {}}]
    if result.get("sql"):
        steps[0]["sql"] = result["sql"]
    if result.get("code"):
        steps[0]["code"] = result["code"]

    return {
        "answer": answer,
        "fig": result.get("fig"),
        "msg": msg_entry,
        "tool_calls": 1,
        "steps": steps,
    }


def _classify_intent_rule(query: str, schema_info: dict) -> dict:
    q = query.lower()
    viz_words = ["画", "图", "图表", "趋势", "分布", "散点图", "折线图", "柱状图",
                  "饼图", "热力图", "箱线图", "plot", "chart", "可视化", "展示"]
    sql_words = ["多少", "几个", "平均", "总和", "最大", "最小", "排名", "前",
                  "count", "sum", "avg", "average", "total", "top"]
    py_words = ["相关性", "相关系数", "分析", "预测", "模型", "回归",
                 "correlation", "analyze", "统计", "假设检验"]

    viz_score = sum(1 for w in viz_words if w in q)
    sql_score = sum(1 for w in sql_words if w in q)
    py_score = sum(1 for w in py_words if w in q)

    multi_words = ["先", "再", "然后", "接着", "分别", "对比"]
    has_multi = any(w in q for w in multi_words)

    if has_multi and (viz_score > 0 or py_score > 0):
        next_tool = "visualization" if viz_score > py_score else "python_analysis"
        return {"tool": "multi_step", "reason": "多步骤", "steps": [
            {"step": 1, "tool": "sql_query", "description": "查询数据"},
            {"step": 2, "tool": next_tool, "description": query},
        ]}

    if viz_score >= sql_score and viz_score >= py_score and viz_score > 0:
        return {"tool": "visualization", "reason": "可视化"}
    if py_score >= sql_score and py_score > 0:
        return {"tool": "python_analysis", "reason": "分析"}
    if sql_score > 0:
        return {"tool": "sql_query", "reason": "查询"}
    return {"tool": "sql_query", "reason": "默认"}


def _handle_single_step(tool: str, query: str, schema_info: dict, df: pd.DataFrame) -> dict:
    if tool == "sql_query":
        return _sql_rule(query, schema_info, df)
    elif tool == "python_analysis":
        return _python_rule(df)
    elif tool == "visualization":
        return _viz_rule(query, schema_info, df)
    return {"status": "error", "message": f"未知工具: {tool}"}


def _handle_multi_step(query: str, schema_info: dict, df: pd.DataFrame) -> dict:
    # 先 SQL 查 -> 再到 visualization/python
    sql_r = _sql_rule(query, schema_info, df)
    if sql_r.get("status") == "error":
        return sql_r
    # 简化为返回 SQL 结果 + 尝试画图
    return sql_r


def _sql_rule(query: str, schema_info: dict, df: pd.DataFrame) -> dict:
    q = query.lower()
    columns = [c["name"] for c in schema_info["schema"]]
    col_str = ", ".join(f'"{c}"' for c in columns)

    is_count = any(w in q for w in ["多少", "几个", "数量", "count", "总共"])
    is_group = any(w in q for w in ["每个", "各", "按", "per", "by", "group"])
    is_top = any(w in q for w in ["top", "前", "排名", "排行"])

    group_cols = [c["name"] for c in schema_info["schema"] if c["unique"] < 50]
    numeric_cols = [c["name"] for c in schema_info["schema"]
                    if any(t in str(c["dtype"]) for t in ["int", "float"])]

    if is_count and is_group and group_cols:
        sql = f'SELECT "{group_cols[0]}", COUNT(*) as count FROM data GROUP BY "{group_cols[0]}" ORDER BY count DESC LIMIT 20'
    elif is_top and group_cols and numeric_cols:
        sql = f'SELECT "{group_cols[0]}", AVG("{numeric_cols[0]}") as avg_value FROM data GROUP BY "{group_cols[0]}" ORDER BY avg_value DESC LIMIT 10'
    elif is_count:
        sql = "SELECT COUNT(*) as total_count FROM data"
    elif numeric_cols:
        sql = f'SELECT {col_str} FROM data ORDER BY "{numeric_cols[0]}" DESC LIMIT 100'
    else:
        sql = f"SELECT {col_str} FROM data LIMIT 100"

    result = run_sql(sql, df)
    return {"sql": sql, **result}


def _python_rule(df: pd.DataFrame) -> dict:
    code = 'result = f"数据集概况：{len(df)} 行 × {len(df.columns)} 列\\n\\n" + "数值列统计：\\n" + df.describe().to_string()'
    return run_python(code, df)


def _viz_rule(query: str, schema_info: dict, df: pd.DataFrame) -> dict:
    q = query.lower()
    columns = [c["name"] for c in schema_info["schema"]]
    numeric_cols = [c["name"] for c in schema_info["schema"]
                    if any(t in str(c["dtype"]) for t in ["int", "float"])]
    cat_cols = [c["name"] for c in schema_info["schema"]
                if "object" in str(c["dtype"]) and c["unique"] < 50]

    if any(w in q for w in ["趋势", "随时间", "每月", "每年"]):
        ct = "line"
        date_cols = [c for c in columns if any(w in c.lower() for w in ["date", "year", "month", "time"])]
        x = date_cols[0] if date_cols else (numeric_cols[0] if numeric_cols else columns[0])
        y = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else None)
    elif any(w in q for w in ["分布", "直方"]):
        ct = "histogram"
        x = numeric_cols[0] if numeric_cols else columns[1]
        y = None
    elif any(w in q for w in ["相关性", "散点", "关系"]):
        ct = "scatter"
        x = numeric_cols[0] if numeric_cols else columns[0]
        y = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else columns[0])
    elif any(w in q for w in ["饼", "比例", "占比"]):
        ct = "pie"
        x = cat_cols[0] if cat_cols else columns[0]
        y = numeric_cols[0] if numeric_cols else None
    else:
        ct = "bar"
        x = cat_cols[0] if cat_cols else columns[0]
        y = numeric_cols[0] if numeric_cols else None

    color = cat_cols[0] if len(cat_cols) > 0 and cat_cols[0] != x else None
    return make_chart(df, chart_type=ct, x_col=x, y_col=y, title=query, color_col=color)


def _df_to_markdown_table(d, max_rows=15):
    """将 DataFrame 前 N 行格式化为 markdown 表格"""
    head = d.head(max_rows)
    cols = list(head.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in head.iterrows():
        vals = [str(v)[:40] for v in row.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _generate_answer_rule(query: str, result: dict) -> str:
    status = result.get("status", result.get("success"))
    if status in ("error", False):
        return "❌ 执行出错：" + str(result.get("message") or result.get("error", "未知错误"))

    lines = ["✅ **分析完成！**"]
    if result.get("data") is not None:
        d = result["data"]
        if hasattr(d, "shape"):
            lines.append(f"\n查询结果：**{d.shape[0]:,}** 行 × **{d.shape[1]}** 列\n")
            lines.append(_df_to_markdown_table(d))
    if result.get("result"):
        lines.append(f"\n计算结果：\n{str(result['result'])[:800]}")
    if result.get("fig"):
        lines.append("\n📊 已生成图表，请查看上方。")
    if result.get("output") and result["output"].strip():
        lines.append(f"\n程序输出：\n{result['output'][:300]}")

    lines.append("\n---\n💡 *规则模式，配置 DeepSeek API Key 可获得更智能的分析。*")
    return "\n".join(lines)
