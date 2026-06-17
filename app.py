"""Streamlit — 自然语言数据分析 Agent"""
import os, io, traceback as tb_mod
import pandas as pd, numpy as np, streamlit as st, plotly.io as pio
from dotenv import load_dotenv

# ── 找项目根目录（兼容本地 + Streamlit Cloud）──
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_PROJECT_DIR, ".env")
load_dotenv(_ENV_PATH)

# ── API Key ──
def _get_api_config():
    """返回 (api_key, base_url, model) 三元组"""
    for name in ("DEEPSEEK_API_KEY",):
        try:
            v = st.secrets.get(name)
            if v: break
        except Exception: pass
    else:
        v = os.getenv("DEEPSEEK_API_KEY", "")
    b = os.getenv("API_BASE_URL", "https://api.deepseek.com")
    m = os.getenv("MODEL_NAME", "deepseek-chat")
    try:
        sv = st.secrets.get("API_BASE_URL")
        if sv: b = sv
    except Exception: pass
    try:
        sv = st.secrets.get("MODEL_NAME")
        if sv: m = sv
    except Exception: pass
    return v, b, m

API_KEY, API_URL, API_MODEL = _get_api_config()
HAS_LLM = bool(API_KEY)

from agent.router import route

# ── 页面配置 ──
st.set_page_config(page_title="Hotel Analysis Agent", page_icon="🏨",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ──
st.markdown("""
<style>
    .stApp { font-family: "Inter","Segoe UI",sans-serif; }
    h1 { color: #1a1a2e !important; font-weight:700 !important; }
    [data-testid="stMetric"] { background:linear-gradient(135deg,#f5f7fa,#e4e9f2); border-radius:12px; padding:16px 20px; border:1px solid #e0e0e0; }
    [data-testid="stMetricValue"] { font-weight:700; color:#1a1a2e; }
    [data-testid="stChatMessage"] { border-radius:16px !important; padding:16px 20px !important; margin-bottom:12px !important; }
    .stButton>button { border-radius:10px !important; font-weight:500 !important; transition:all .2s; }
    .stButton>button:hover { transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,0,0,.15); }
    .insight-card { background:#fefefe; border:1px solid #e8e8e8; border-radius:12px; padding:14px 18px; margin:6px 0; font-size:.92rem; }
    hr { margin:2rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "pending" not in st.session_state:
    st.session_state.pending = False

# ── 自动加载数据 ──
DATA_PATH = os.path.join(_PROJECT_DIR, "data", "hotel_bookings.csv")
if st.session_state.df is None and os.path.exists(DATA_PATH):
    st.session_state.df = pd.read_csv(DATA_PATH)

# ── 自动洞察 ──
def auto_insights(d):
    ins = []
    n = len(d)
    if n == 0: return [("📭", "数据集为空")]
    ins.append(("📊", f"共 **{n:,}** 行 × **{len(d.columns)}** 列"))
    if "is_canceled" in d.columns:
        r = d["is_canceled"].mean() * 100
        ins.append(("❌", f"取消率 **{r:.1f}%**" + ("（较高）" if r > 30 else "")))
    if "lead_time" in d.columns and "is_canceled" in d.columns:
        ca = d[d["is_canceled"] == 1]["lead_time"].mean()
        nc = d[d["is_canceled"] == 0]["lead_time"].mean()
        ins.append(("📅", f"取消订单平均提前 **{ca:.0f}** 天 vs 未取消 **{nc:.0f}** 天"))
    if "adr" in d.columns:
        ins.append(("💰", f"ADR 范围 **${d['adr'].min():.0f} ~ ${d['adr'].max():.0f}**，均值 **${d['adr'].mean():.0f}**"))
    if "country" in d.columns:
        t3 = d["country"].value_counts().head(3)
        ins.append(("🌍", "Top 3 来源：" + "、".join([f"{c}({v:,})" for c, v in t3.items()])))
    if "market_segment" in d.columns and "is_canceled" in d.columns:
        se = d.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False)
        ins.append(("📈", f"取消率最高渠道 **{se.index[0]}**（{se.iloc[0]*100:.1f}%）"))
    return ins

# ── 热力图 ──
def make_heatmap(d):
    import plotly.graph_objects as go
    num = d.select_dtypes(include=[np.number])
    if num.shape[1] < 2: return None
    cols = [c for c in [
        "lead_time","adr","is_canceled","stays_in_weekend_nights",
        "stays_in_week_nights","adults","children","babies",
        "previous_cancellations","booking_changes","days_in_waiting_list",
        "total_of_special_requests","required_car_parking_spaces"
    ] if c in num.columns]
    sub = num[cols]; corr = sub.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu", zmin=-1, zmax=1, texttemplate="%{z:.2f}", textfont=dict(size=11)))
    fig.update_layout(title="📊 数值特征相关性热力图", title_font_size=18,
                      template="plotly_white", height=550, width=700,
                      margin=dict(l=120, r=20, t=60, b=100))
    fig.update_xaxes(tickangle=45)
    return fig

# ═══════════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🏨 Hotel Analysis Agent")
    if HAS_LLM:
        st.success(f"🤖 {API_MODEL} 已就绪")
    else:
        st.info("📐 离线规则模式")
    st.divider()
    st.subheader("📂 数据")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🏨 加载演示数据", use_container_width=True):
            if os.path.exists(DATA_PATH):
                st.session_state.df = pd.read_csv(DATA_PATH)
                st.session_state.messages = []
                st.session_state.pending = False
                st.rerun()
            else:
                st.error("数据文件未找到")
    with col_b:
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    uploaded = st.file_uploader("上传 CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.session_state.messages = []
        st.session_state.pending = False
        st.rerun()

    if st.session_state.df is not None:
        df_sidebar = st.session_state.df
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("行数", f"{len(df_sidebar):,}")
        c2.metric("列数", len(df_sidebar.columns))
        with st.expander("字段 + 预览"):
            for c in df_sidebar.columns:
                st.text(f"• {c}  [{df_sidebar[c].dtype}]")
            st.dataframe(df_sidebar.head(5), use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# 主区域
# ═══════════════════════════════════════════════════════════════
st.title("🏨 酒店预订数据分析 Agent")
st.markdown("用自然语言提问 → AI 自动分析数据并生成图表。")

if st.session_state.df is None:
    st.info("👈 点击左侧「加载演示数据」开始。")
    st.stop()

df = st.session_state.df  # type: pd.DataFrame

# ── 状态栏 ──
tag = f"🤖 {API_MODEL} 模式" if HAS_LLM else "📐 规则模式"
st.caption(tag)

# ── 指标卡 ──
if "is_canceled" in df.columns and "adr" in df.columns:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📊 总订单", f"{len(df):,}")
    k2.metric("❌ 取消率", f"{df['is_canceled'].mean()*100:.1f}%")
    k3.metric("💰 平均 ADR", f"${df['adr'].mean():.2f}")
    k4.metric("📅 平均提前天数", f"{df['lead_time'].mean():.0f}天")
    tc = df["country"].value_counts().index[0] if "country" in df.columns else "N/A"
    k5.metric("🌍 Top 来源国", tc)

# ── 洞察 ──
with st.expander("💡 自动数据洞察（统计分析，非 LLM）", expanded=(len(st.session_state.messages) == 0)):
    for emoji, text in auto_insights(df):
        st.markdown(f'<div class="insight-card">{emoji} {text}</div>', unsafe_allow_html=True)

st.divider()

# ── 快捷问题 ──
_qs = [
    ("📊", "各月取消率的趋势是什么？"),
    ("🏆", "取消率最高的前5个国家"),
    ("📈", "画出 lead_time 的分布直方图"),
    ("🏨", "City Hotel 和 Resort Hotel 的平均 ADR 对比"),
]
qc1, qc2 = st.columns([3, 1])
with qc1:
    qcols = st.columns(len(_qs))
    for i, (e, q) in enumerate(_qs):
        with qcols[i]:
            if st.button(f"{e} {q}", key=f"qq_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.session_state.pending = True
                st.rerun()
with qc2:
    if st.button("🔥 相关性热力图", use_container_width=True):
        fh = make_heatmap(df)
        if fh:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "### 🔥 数值特征相关性热力图\n\n红色=正相关，蓝色=负相关。",
                "fig": fh,
                "steps": [{"tool": "heatmap", "args": {"chart_type": "heatmap"}}],
            })
            st.rerun()

st.divider()

# ═══════════════════════════════════════════════════════════════
# 核心：处理待处理消息
# ═══════════════════════════════════════════════════════════════
if st.session_state.pending:
    # 最后一条用户消息即为待处理问题
    user_msg = st.session_state.messages[-1]
    assert user_msg["role"] == "user"
    question = user_msg["content"]

    st.session_state.pending = False

    with st.chat_message("assistant"):
        with st.spinner("🤔 AI 分析中…"):
            try:
                # 构建 clean history（不含最后一条 user 消息，route 会单独接收 question）
                clean = [
                    {"role": m["role"], "content": str(m.get("content", ""))}
                    for m in st.session_state.messages[:-1]
                ]

                result = route(
                    question=question, df=df, history=clean,
                    api_key=API_KEY, base_url=API_URL, model=API_MODEL,
                )

                # ① 结论
                st.markdown(result["answer"])

                # ② 图表 + 下载
                if result.get("fig"):
                    st.plotly_chart(result["fig"], use_container_width=True)
                    try:
                        buf = io.BytesIO()
                        pio.write_image(result["fig"], buf, format="png", scale=2, width=1200, height=700)
                        st.download_button("📥 下载图表 (PNG)", data=buf.getvalue(),
                                           file_name="chart.png", mime="image/png",
                                           key=f"dl_{len(st.session_state.messages)}")
                    except Exception:
                        pass  # 下载按钮非必需，失败了不影响主功能

                # ③ 分析思路
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
                            if s.get("error"):
                                st.error(f"⚠️ {s['error']}")

                # 存 assistant 消息
                msg_entry = result["msg"]
                if steps: msg_entry["steps"] = steps
                st.session_state.messages.append(msg_entry)

            except Exception as e:
                err = str(e)
                if "401" in err.lower() or "unauthorized" in err.lower():
                    st.error("🔑 API Key 无效，请检查 `.env` 文件。")
                elif "timeout" in err.lower():
                    st.error("⏱️ 请求超时，请检查网络。")
                elif "rate" in err.lower():
                    st.error("🔄 请求太频繁，稍等几秒。")
                else:
                    st.error(f"❌ 处理出错，请重试。")
                    with st.expander("技术详情"):
                        st.code(tb_mod.format_exc())

    st.rerun()

# ═══════════════════════════════════════════════════════════════
# 对话历史
# ═══════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))
        if msg.get("fig"):
            st.plotly_chart(msg["fig"], use_container_width=True)
            try:
                buf = io.BytesIO()
                pio.write_image(msg["fig"], buf, format="png", scale=2, width=1200, height=700)
                st.download_button("📥 下载图表 (PNG)", data=buf.getvalue(),
                                   file_name="chart.png", mime="image/png",
                                   key=f"dl_hist_{abs(id(msg))}")
            except Exception:
                pass
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("🔍 查看分析思路"):
                for idx, s in enumerate(msg["steps"], 1):
                    t = s.get("tool", "?")
                    if t == "run_sql":
                        sql = s.get("args", {}).get("sql") or s.get("sql", "")
                        if sql: st.code(sql, language="sql")
                    elif t == "run_python":
                        code = s.get("args", {}).get("code") or s.get("code", "")
                        if code: st.code(code, language="python")
                    elif t in ("make_chart", "fallback", "heatmap"):
                        a = s.get("args", {}); st.caption(f"{idx}. 图表 `{a.get('chart_type', t)}`")
                    if s.get("error"): st.error(f"⚠️ {s['error']}")

# ═══════════════════════════════════════════════════════════════
# 输入框
# ═══════════════════════════════════════════════════════════════
if prompt := st.chat_input("请输入你的数据分析问题…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending = True
    st.rerun()

st.divider()
st.caption("Hotel Booking Analysis Agent · 自动模式 · 随时可用")
