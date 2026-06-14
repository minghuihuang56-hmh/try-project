"""三个核心工具：run_sql / run_python / make_chart

返回值规范：
  run_sql:     {"status": "ok", "data": DataFrame, "rows": int}
               {"status": "error", "message": str}
  run_python:  {"status": "ok", "result": any, "output": str}
               {"status": "error", "message": str}
  make_chart:  {"status": "ok", "fig": plotly Figure}
               {"status": "error", "message": str}
"""

import io
import re
import sqlite3
import sys
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 可安全导入的白名单
# ============================================================
ALLOWED_IMPORTS = {
    "pandas", "numpy", "plotly", "plotly.express", "plotly.graph_objects",
    "scipy", "scipy.stats", "math", "datetime", "json", "re",
    "collections", "itertools", "typing", "copy", "functools", "operator",
    "decimal", "random", "statistics",
}


# ============================================================
# 工具 1：SQL 查询
# ============================================================
def run_sql(sql: str, df: pd.DataFrame) -> dict:
    """在内存 SQLite 中执行 SQL 查询，返回结果 DataFrame。

    Args:
        sql: SQL 查询语句
        df: 源数据 DataFrame

    Returns:
        {"status": "ok",   "data": pd.DataFrame, "rows": int}
        {"status": "error", "message": str}
    """
    if not sql or not sql.strip():
        return {"status": "error", "message": "SQL 语句不能为空"}

    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        # 将 DataFrame 写入内存数据库
        df.to_sql("data", conn, index=False, if_exists="replace")

        # 执行查询
        result_df = pd.read_sql_query(sql, conn)

        return {
            "status": "ok",
            "data": result_df,
            "rows": len(result_df),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"SQL 执行失败: {e}",
        }
    finally:
        if conn:
            conn.close()


# ============================================================
# 工具 2：Python 分析
# ============================================================
def run_python(code: str, df: pd.DataFrame) -> dict:
    """在受限命名空间中执行 Python 代码。

    安全策略：
    - 只允许白名单内的 import
    - 禁止 __import__ 外的调用方式（eval / exec / open 等被移除）
    - 预置 pd, np, px, go 到命名空间
    - 代码通过 result 变量返回文本结果，通过 fig 变量返回图表

    Args:
        code: Python 代码（可含 ```python 包装）
        df: 源数据 DataFrame

    Returns:
        {"status": "ok",   "result": any, "output": str, "fig": Figure|None}
        {"status": "error", "message": str}
    """
    # 1) 提取纯净代码
    cleaned = _extract_python_code(code)
    if not cleaned:
        return {"status": "error", "message": "代码为空"}

    # 2) 建立受限 builtins
    safe_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool,
        "dict": dict, "enumerate": enumerate, "float": float,
        "format": format, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "map": map, "max": max, "min": min,
        "range": range, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "type": type,
        "zip": zip, "filter": filter, "reversed": reversed,
        "True": True, "False": False, "None": None,
        "print": print, "__import__": _safe_import,
    }

    # 3) 准备命名空间（预置常用库）
    local_vars = {
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "df": df.copy(),
        "result": None,
        "fig": None,
    }

    # 4) 捕获 stdout
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        exec(cleaned, {"__builtins__": safe_builtins}, local_vars)
        output = captured.getvalue()
        result = local_vars.get("result")
        fig = local_vars.get("fig")

        return {
            "status": "ok",
            "result": result if result is not None else (output or "执行完成"),
            "output": output[:5000],
            "fig": fig if fig is not None else None,
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "status": "error",
            "message": f"Python 执行失败: {e}\n\n{tb[-2000:]}",
        }
    finally:
        sys.stdout = old_stdout


def _safe_import(name, *args, **kwargs):
    """安全的 import 钩子：只允许白名单模块。"""
    if name in ALLOWED_IMPORTS:
        return __import__(name, *args, **kwargs)
    # 处理 plotly.xxx 和 scipy.xxx 子模块
    parent = name.split(".")[0]
    if parent in {"plotly", "scipy"}:
        return __import__(name, *args, **kwargs)
    raise ImportError(
        f"[安全限制] 模块 '{name}' 不允许导入。"
        f"只允许: {', '.join(sorted(ALLOWED_IMPORTS))}"
    )


# ============================================================
# 工具 3：可视化（直接生成图表，无需 LLM 生成代码）
# ============================================================
def make_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str = None,
    title: str = "",
    color_col: str = None,
) -> dict:
    """使用 plotly.express 直接生成图表。

    Args:
        df: 数据 DataFrame
        chart_type: 图表类型 (bar / line / scatter / histogram / pie)
        x_col: X 轴列名
        y_col: Y 轴列名（histogram/pie 可省略）
        title: 图表标题
        color_col: 分组/颜色列（可选）

    Returns:
        {"status": "ok",   "fig": plotly.Figure}
        {"status": "error", "message": str}
    """
    # ── 参数校验 ──────────────────────────────────────────────
    if df.empty:
        return {"status": "error", "message": "DataFrame 为空，无法作图"}

    if x_col not in df.columns:
        return {
            "status": "error",
            "message": f"列 '{x_col}' 不在数据中。可用列: {list(df.columns)}",
        }

    chart_type = chart_type.lower().strip()
    supported = {"bar", "line", "scatter", "histogram", "pie"}
    if chart_type not in supported:
        return {
            "status": "error",
            "message": f"不支持的图表类型 '{chart_type}'。支持: {supported}",
        }

    if y_col and y_col not in df.columns:
        return {
            "status": "error",
            "message": f"列 '{y_col}' 不在数据中。可用列: {list(df.columns)}",
        }

    if color_col and color_col not in df.columns:
        color_col = None

    try:
        fig = None
        kwargs = dict(title=title or f"{chart_type.title()} Chart", template="plotly_white")

        # ── bar ──
        if chart_type == "bar":
            if y_col:
                fig = px.bar(df, x=x_col, y=y_col, color=color_col, **kwargs)
            else:
                # 无 y_col 时自动计数
                fig = px.bar(
                    df[x_col].value_counts().reset_index(),
                    x=x_col, y="count", title=kwargs["title"],
                    template="plotly_white",
                )

        # ── line ──
        elif chart_type == "line":
            if y_col:
                fig = px.line(df, x=x_col, y=y_col, color=color_col, **kwargs)
            else:
                fig = px.line(
                    df[x_col].value_counts().sort_index().reset_index(),
                    x=x_col, y="count", title=kwargs["title"],
                    template="plotly_white",
                )

        # ── scatter ──
        elif chart_type == "scatter":
            if not y_col:
                return {"status": "error", "message": "散点图需要 y_col 参数"}
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, **kwargs)

        # ── histogram ──
        elif chart_type == "histogram":
            kwargs.pop("template", None)  # px.histogram 的 template 另有写法
            fig = px.histogram(
                df, x=x_col, color=color_col, nbins=50,
                title=title or f"Distribution of {x_col}",
                template="plotly_white",
            )

        # ── pie ──
        elif chart_type == "pie":
            if y_col:
                # 聚合后画饼图
                agg = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.pie(agg, names=x_col, values=y_col, **kwargs)
            else:
                values = df[x_col].value_counts().reset_index()
                fig = px.pie(values, names=x_col, values="count", **kwargs)

        # ── 美化 ──
        if fig:
            fig.update_layout(
                title_font=dict(size=16),
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40),
            )
            # 更新轴标签（去除列名下划线）
            fig.update_xaxes(title_text=x_col.replace("_", " "))
            if y_col:
                fig.update_yaxes(title_text=y_col.replace("_", " "))

        return {"status": "ok", "fig": fig}

    except Exception as e:
        return {
            "status": "error",
            "message": f"图表生成失败: {e}",
        }


# ============================================================
# 辅助工具
# ============================================================

def get_schema_info(df: pd.DataFrame) -> dict:
    """获取 DataFrame Schema 信息"""
    schema = []
    for col in df.columns:
        schema.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "unique": int(df[col].nunique()),
            "sample_values": df[col].dropna().unique()[:3].tolist(),
        })

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "schema": schema,
    }


def get_summary_stats(df: pd.DataFrame) -> dict:
    """获取数据集基本统计信息"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "basic_stats": df.describe().to_dict() if numeric_cols else {},
    }


def _extract_python_code(text: str) -> str:
    """从可能包含 markdown 代码块的文本中提取 Python 代码"""
    # ```python ... ``` 或 ``` ... ```
    pattern = r"```(?:python)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()
