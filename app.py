"""Streamlit 主入口 — 自然语言驱动的数据分析 Agent"""
import os, io, traceback as tb_mod
import pandas as pd, numpy as np, streamlit as st, plotly.io as pio
from dotenv import load_dotenv

load_dotenv()

# ── API Key：st.secrets（云端）> os.environ（本地）──
def _resolve_key(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name)
        if val: return val
    except Exception: pass
    return os.getenv(name, default)

HAS_LLM = bool(_resolve_key("DEEPSEEK_API_KEY"))

from agent.router import route

# ── UI 配置 ──────────────────────────────────────────────────────
st.set_page_config(page_title="Hotel Booking Analysis Agent", page_icon="🏨",
                   layout="wide", initial_sidebar_state="expanded")

# ── 自定义 CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* 全局 */
    .stApp { font-family: "Inter", "Segoe UI", sans-serif; }
    h1 { color: #1a1a2e !important; font-weight: 700 !important; }
    h3 { color: #16213e !important; }

    /* 指标卡 */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
        border-radius: 12px; padding: 16px 20px; border: 1px solid #e0e0e0;
    }
    [data-testid="stMetricValue"] { font-weight: 700; color: #1a1a2e; }

    /* 聊天消息 */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important; padding: 16px 20px !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stChatMessage"]:has(.stChatMessageAvatarUser) {
        background: #e8f0fe; border-left: 4px solid #4285f4;
    }
    div[data-testid="stChatMessage"]:has(.stChatMessageAvatarAssistant) {
        background: #f8f9fa; border-left: 4px solid #34a853;
    }

    /* 按钮 */
    .stButton > button {
        border-radius: 10px !important; font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }

    /* 快捷问题按钮特殊样式 */
    .quick-btn button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }

    /* 洞察卡片 */
    .insight-card {
        background: #fefefe; border: 1px solid #e8e8e8; border-radius: 12px;
        padding: 14px 18px; margin: 6px 0; font-size: 0.92rem;
    }
    .insight-card b { color: #16213e; }

    /* 底部分隔 */
    hr { margin: 2rem 0 1rem 0; }

    /* 数据快照区 */
    [data-testid="stExpander"] { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────
for key in ["messages", "df", "ask"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None

AUTO_DATA_PATH = os.path.join("data", "hotel_bookings.csv")
if st.session_state.df is None and os.path.exists(AUTO_DATA_PATH):
    st.session_state.df = pd.read_csv(AUTO_DATA_PATH)

# ── 自动洞察引擎 ────────────────────────────────────────────────
def auto_insights(df: pd.DataFrame) -> list:
    """纯统计，不调 API。返回 [(emoji, 文本)] 列表。"""
    insights = []
    n = len(df)
    if n == 0: return [("📭", "数据集为空。")]

    rows = f"{n:,}"
    cols = f"{len(df.columns)}"
    insights.append(("📊", f"数据集共 **{rows}** 行 × **{cols}** 列"))

    if "is_canceled" in df.columns:
        r = df["is_canceled"].mean() * 100
        insights.append(("❌", f"总体取消率 **{r:.1f}%**" + ("（较高，值得分析原因）" if r > 30 else "")))

    if "lead_time" in df.columns and "is_canceled" in df.columns:
        canceled = df[df["is_canceled"] == 1]["lead_time"].mean()
        not_canceled = df[df["is_canceled"] == 0]["lead_time"].mean()
        insights.append(("📅", f"取消订单的平均提前预订天数（**{canceled:.0f}天**）远高于未取消订单（**{not_canceled:.0f}天**）"))

    if "adr" in df.columns:
        insights.append(("💰", f"日均房价 ADR 范围：**${df['adr'].min():.0f} ~ ${df['adr'].max():.0f}**，均值 **${df['adr'].mean():.0f}**"))

    if "country" in df.columns:
        top3 = df["country"].value_counts().head(3)
        tops = "、".join([f"{c}（{v:,}）" for c, v in top3.items()])
        insights.append(("🌍", f"Top 3 来源国家：{tops}"))

    if "market_segment" in df.columns and "is_canceled" in df.columns:
        seg = df.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False)
        worst = seg.index[0]
        insights.append(("📈", f"取消率最高的渠道是 **{worst}**（{seg[worst]*100:.1f}%），建议重点关注"))

    return insights

# ── 相关性热力图生成 ────────────────────────────────────────────
def make_heatmap(df: pd.DataFrame):
    import plotly.graph_objects as go
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return None
    # 选取关键数值列（不超过 15 列）
    cols = [c for c in ["lead_time","adr","is_canceled","stays_in_weekend_nights",
            "stays_in_week_nights","adults","children","babies",
            "previous_cancellations","booking_changes","days_in_waiting_list",
            "total_of_special_requests","required_car_parking_spaces"]
            if c in num.columns]
    sub = num[cols]
    corr = sub.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", zmin=-1, zmax=1, texttemplate="%{z:.2f}",
        textfont=dict(size=11),
    ))
    fig.update_layout(title="📊 数值特征相关性热力图", title_font_size=18,
                      template="plotly_white", height=550, width=700,
                      margin=dict(l=120, r=20, t=60, b=100))
    fig.update_xaxes(tickangle=45)
    return fig

# ==================================================================
# 侧边栏（精简：只有数据，无 LLM 配置）
# ==================================================================
with st.sidebar:
    st.header("🏨 Hotel Analysis Agent")

    # 模式标签
    if HAS_LLM:
        st.success("🤖 DeepSeek AI 已就绪")
    else:
        st.info("📐 离线规则模式（配置 Key 可启用 AI）")

    st.divider()
    st.subheader("📂 数据")

    btn_text = "🏨 加载酒店演示数据" if st.session_state.df is None else "🔄 重新加载演示数据"
    if st.button(btn_text, use_container_width=True):
        if os.path.exists(AUTO_DATA_PATH):
            st.session_state.df = pd.read_csv(AUTO_DATA_PATH)
        else:
            from shutil import copy as _cp
            fallback = r"C:\Users\黄明慧\Desktop\11_项目代码\酒店预定管理\hotel_bookings.csv"
            if os.path.exists(fallback):
                os.makedirs("data", exist_ok=True); _cp(fallback, AUTO_DATA_PATH)
                st.session_state.df = pd.read_csv(AUTO_DATA_PATH)
            else:
                st.error("找不到数据文件，请上传 CSV。")
        st.session_state.messages = []
        st.rerun()

    uploaded = st.file_uploader("或上传 CSV", type=["csv"])
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.session_state.messages = []
        st.rerun()

    if st.session_state.df is not None:
        df = st.session_state.df
        st.divider()
        st.subheader("📋 数据概览")
        c1, c2 = st.columns(2)
        c1.metric("行数", f"{len(df):,}")
        c2.metric("列数", len(df.columns))
        with st.expander("字段列表"):
            for c in df.columns:
                st.text(f"• {c}  [{df[c].dtype}]")
        with st.expander("前 5 行预览"):
            st.dataframe(df.head(5), use_container_width=True)

# ==================================================================
# 主区域
# ==================================================================

st.title("🏨 酒店预订数据分析 Agent")
st.markdown("用自然语言提问 → **AI 自动分析** → 结论 + 图表。无需写代码。")

# ── 未加载数据 ──────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("👈 点击左侧「加载酒店演示数据」开始。")
    st.markdown("""
    ### 快速开始
    1. 左侧点击 **🏨 加载酒店演示数据**（119,390 条真实酒店预订记录）
    2. 在底部输入自然语言问题
    3. AI 自动分析 → 返回结论 + 图表
    """)
    st.stop()

df = st.session_state.df

# ── 模式标签 ────────────────────────────────────────────────────
tag = "🤖 DeepSeek AI 模式" if HAS_LLM else "📐 规则模式"
st.caption(tag)

# ── 数据快照（指标卡） ───────────────────────────────────────────
if "is_canceled" in df.columns and "adr" in df.columns:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 总订单", f"{len(df):,}")
    k2.metric("❌ 取消率", f"{df['is_canceled'].mean()*100:.1f}%")
    k3.metric("💰 平均 ADR", f"${df['adr'].mean():.2f}")
    k4.metric("📅 平均提前天数", f"{df['lead_time'].mean():.0f}天")
    top_country = df["country"].value_counts().index[0] if "country" in df.columns else "N/A"
    k5.metric("🌍 Top 来源国", top_country)

# ── 自动洞察 ────────────────────────────────────────────────────
with st.expander("💡 自动数据洞察（基于统计，非 LLM）", expanded=(len(st.session_state.messages) == 0)):
    if "insights" not in st.session_state:
        st.session_state.insights = auto_insights(df)
    for emoji, text in st.session_state.insights:
        st.markdown(f'<div class="insight-card">{emoji} {text}</div>', unsafe_allow_html=True)

st.divider()

# ── 快捷提问 + 一键分析 ─────────────────────────────────────────
cq1, cq2 = st.columns([3, 1])
with cq1:
    qs = [
        ("📊", "各月取消率的趋势是什么？"),
        ("🏆", "取消率最高的前5个国家"),
        ("📈", "画出 lead_time 的分布直方图"),
        ("🏨", "City Hotel 和 Resort Hotel 的平均 ADR 对比"),
    ]
    cols = st.columns(len(qs))
    for i, (emoji, question) in enumerate(qs):
        with cols[i]:
            if st.button(f"{emoji} {question}", key=f"q_{i}", use_container_width=True):
                st.session_state.ask = question
                st.rerun()
with cq2:
    if st.button("🔥 相关性热力图", use_container_width=True):
        fig_heat = make_heatmap(df)
        if fig_heat:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "### 🔥 数值特征相关性热力图\n\n展示主要数值列之间的线性相关程度。红色=正相关，蓝色=负相关。",
                "fig": fig_heat,
                "steps": [{"tool": "make_chart", "args": {"chart_type": "heatmap"}}],
            })
            st.rerun()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.divider()

# ── 对话历史 ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))
        if msg.get("fig"):
            st.plotly_chart(msg["fig"], use_container_width=True)
            # 下载按钮
            buf = io.BytesIO()
            pio.write_image(msg["fig"], buf, format="png", scale=2, width=1200, height=700)
            st.download_button("📥 下载图表 (PNG)", data=buf.getvalue(),
                               file_name="chart.png", mime="image/png",
                               key=f"dl_{hash(str(msg.get('content',''))[-20:])}")
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("🔍 查看分析思路"):
                st.caption(f"调用了 {len(msg['steps'])} 次工具")
                for idx, s in enumerate(msg["steps"], 1):
                    t = s.get("tool", "?")
                    if t == "run_sql":
                        sql = s.get("args", {}).get("sql") or s.get("sql", "")
                        if sql: st.code(sql, language="sql")
                    elif t == "run_python":
                        code = s.get("args", {}).get("code") or s.get("code", "")
                        if code: st.code(code, language="python")
                    elif t in ("make_chart", "fallback", "heatmap"):
                        a = s.get("args", {})
                        ct = a.get("chart_type", t)
                        st.caption(f"{idx}. 图表 `{ct}`")
                    if s.get("error"): st.error(f"⚠️ {s['error']}")

# ── 输入框 ──────────────────────────────────────────────────────
prompt = st.session_state.get("ask") or st.chat_input("请输入你的数据分析问题…")

if prompt:
    if st.session_state.get("ask") == prompt:
        st.session_state.ask = None

    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 AI 分析中…"):
            try:
                clean = [{"role": m["role"], "content": str(m.get("content", ""))}
                         for m in st.session_state.messages]

                result = route(
                    question=prompt, df=df, history=clean,
                    api_key=_resolve_key("DEEPSEEK_API_KEY") if HAS_LLM else "",
                    base_url=_resolve_key("API_BASE_URL", "https://api.deepseek.com") if HAS_LLM else "",
                    model=_resolve_key("MODEL_NAME", "deepseek-chat") if HAS_LLM else "",
                )

                st.markdown(result["answer"])
                if result.get("fig"):
                    st.plotly_chart(result["fig"], use_container_width=True)
                    buf = io.BytesIO()
                    pio.write_image(result["fig"], buf, format="png", scale=2, width=1200, height=700)
                    st.download_button("📥 下载图表 (PNG)", data=buf.getvalue(),
                                       file_name="chart.png", mime="image/png",
                                       key=f"dl_{abs(hash(prompt))}")

                steps = result.get("steps", [])
                if steps:
                    with st.expander("🔍 查看分析思路"):
                        st.caption(f"调用了 {len(steps)} 次工具")
                        for idx, s in enumerate(steps, 1):
                            t = s.get("tool", "?")
                            if t == "run_sql":
                                sql = s.get("args", {}).get("sql") or s.get("sql", "")
                                if sql: st.code(sql, language="sql")
                            elif t == "run_python":
                                code = s.get("args", {}).get("code") or s.get("code", "")
                                if code: st.code(code, language="python")
                            elif t in ("make_chart", "fallback", "heatmap"):
                                a = s.get("args", {})
                                st.caption(f"{idx}. 图表 `{a.get('chart_type', t)}`")
                                if s.get("sql"): st.code(s["sql"], language="sql")
                                if s.get("code"): st.code(s["code"], language="python")
                            if s.get("error"): st.error(f"⚠️ {s['error']}")

                msg_entry = result["msg"]
                if steps: msg_entry["steps"] = steps
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append(msg_entry)

            except Exception as e:
                err = str(e).lower()
                if "401" in err or "unauthorized" in err:
                    st.error("🔑 API Key 无效。请在 `.env` 中配置 `DEEPSEEK_API_KEY`。")
                elif "timeout" in err or "timed" in err:
                    st.error("⏱️ 请求超时，请检查网络。")
                elif "rate" in err:
                    st.error("🔄 请求太频繁，稍等几秒。")
                else:
                    st.error("❌ 处理出错，请重试或换一种问法。")
                    with st.expander("技术详情"):
                        st.code(tb_mod.format_exc())

st.divider()
st.caption("Hotel Booking Analysis Agent · 自动 AI / 离线模式 · 随时可用")
