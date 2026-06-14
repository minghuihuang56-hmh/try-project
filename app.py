"""Streamlit 主入口 — 自然语言驱动的数据分析 Agent"""
import os, traceback as tb_mod
import pandas as pd, streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── 统一读取 Key：st.secrets（云端） > os.environ（本地 .env） > 空 ──
def _resolve_key(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name)
        if val: return val
    except Exception:
        pass
    return os.getenv(name, default)

from agent.router import route

# ── 页面配置 ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel Booking Analysis Agent",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State ──────────────────────────────────────────────
for key in ["messages", "df", "ask"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "messages" else None

# 自动探测数据：如果 data/hotel_bookings.csv 存在，开机即加载
AUTO_DATA_PATH = os.path.join("data", "hotel_bookings.csv")
if st.session_state.df is None and os.path.exists(AUTO_DATA_PATH):
    st.session_state.df = pd.read_csv(AUTO_DATA_PATH)

# ==================================================================
# 侧边栏（精简版：只放配置 + 数据源）
# ==================================================================
with st.sidebar:
    st.header("⚙️ 配置")

    # —— LLM 后端 ——
    llm_backend = st.radio(
        "LLM 后端",
        ["📐 规则模式（免 API）", "🔌 DeepSeek API", "🖥️ Ollama 本地"],
        index=0,
    )

    api_key = ""
    api_base_url = ""
    model_name = ""

    if "Ollama" in llm_backend:
        st.info("需本地运行: `ollama pull qwen2.5-coder:7b`")
        api_base_url = st.text_input("Ollama 地址", value=os.getenv("API_BASE_URL", "http://localhost:11434/v1"))
        model_name  = st.text_input("模型名称",  value=os.getenv("MODEL_NAME", "qwen2.5-coder:7b"))
        api_key = "ollama"
    elif "DeepSeek" in llm_backend:
        api_key      = st.text_input("DeepSeek API Key", type="password", value=_resolve_key("DEEPSEEK_API_KEY", ""),
                                      help="https://platform.deepseek.com/sign_up")
        api_base_url = st.text_input("API 地址",  value=_resolve_key("API_BASE_URL", "https://api.deepseek.com"))
        model_name   = st.text_input("模型名称",   value=_resolve_key("MODEL_NAME", "deepseek-chat"))
    else:
        st.caption("📐 规则模式，无需 API Key，点击即用")

    # 写入 session_state
    st.session_state.api_key      = api_key
    st.session_state.api_base_url = api_base_url
    st.session_state.model_name   = model_name
    st.session_state.llm_backend  = llm_backend

    st.divider()

    # —— 数据源 ——
    st.subheader("📂 数据")
    # 演示数据一键加载
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

    # 上传自己的 CSV
    uploaded = st.file_uploader("或上传 CSV", type=["csv"])
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.session_state.messages = []
        st.rerun()

    # —— 数据概览 ——
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

    # —— 警告 ——
    if "DeepSeek" in llm_backend and not api_key:
        st.warning("⚠️ 未配置 API Key，请注册 DeepSeek 或切到规则模式。")

# ==================================================================
# 主区域
# ==================================================================

st.title("🏨 酒店预订数据分析 Agent")
st.markdown("用自然语言提问 → AI 自动分析数据并生成图表。支持免 API 模式。")

# ── 未加载数据 ──────────────────────────────────────────────────
if st.session_state.df is None:
    st.info("👈 请在左侧加载数据（点击「加载酒店演示数据」）。")
    st.markdown("""
    ### 快速开始
    1. 左侧选择 LLM 后端 → **推荐先用「规则模式」体验**
    2. 点击 **🏨 加载酒店演示数据**
    3. 输入问题开始分析
    """)
    st.stop()

# ── 数据已加载 → 展示核心指标 + 快捷问题 ───────────────────────
df = st.session_state.df

# 模式标签
api_key_val = str(st.session_state.get("api_key", ""))
if api_key_val and api_key_val != "ollama":
    tag = f"🤖 {st.session_state.get('model_name','LLM')} 模式"
else:
    tag = "📐 规则模式"
st.caption(tag)

# 数据快照（指标卡）
if "is_canceled" in df.columns and "adr" in df.columns and "lead_time" in df.columns:
    k1, k2, k3, k4, k5 = st.columns(5)
    cancel_rate = df["is_canceled"].mean() * 100
    k1.metric("📊 总订单", f"{len(df):,}")
    k2.metric("❌ 取消率", f"{cancel_rate:.1f}%")
    k3.metric("💰 平均 ADR", f"${df['adr'].mean():.2f}")
    k4.metric("📅 平均提前天数", f"{df['lead_time'].mean():.0f}天")
    top_country = df["country"].value_counts().index[0] if "country" in df.columns else "N/A"
    k5.metric("🌍 Top 来源国", top_country)
    st.divider()

# 快捷提问（在主区域，显眼）
qs = [
    ("📊", "各月取消率的趋势是什么？"),
    ("🏆", "取消率最高的前5个国家"),
    ("📈", "画出 lead_time 的分布直方图"),
    ("🏨", "City Hotel 和 Resort Hotel 的平均 ADR 对比"),
]
cols = st.columns(len(qs))
for i, (emoji, question) in enumerate(qs):
    with cols[i]:
        if st.button(f"{emoji} {question}", key=f"quick_{i}", use_container_width=True):
            st.session_state.ask = question
            st.rerun()

# 对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg.get("content", ""))
        if msg.get("fig"):
            st.plotly_chart(msg["fig"], use_container_width=True)
        # 分析思路
        if msg["role"] == "assistant" and msg.get("steps"):
            with st.expander("🔍 查看分析思路"):
                steps = msg["steps"]
                st.caption(f"调用了 {len(steps)} 次工具")
                for idx, s in enumerate(steps, 1):
                    tool = s.get("tool", "?")
                    if tool == "run_sql":
                        sql = s.get("args",{}).get("sql") or s.get("sql","")
                        if sql:
                            st.markdown(f"**{idx}．SQL 查询**")
                            st.code(sql, language="sql")
                    elif tool == "run_python":
                        code = s.get("args",{}).get("code") or s.get("code","")
                        if code:
                            st.markdown(f"**{idx}．Python 分析**")
                            st.code(code, language="python")
                    elif tool in ("make_chart", "fallback"):
                        a = s.get("args",{})
                        ct = a.get("chart_type","?"); xc=a.get("x_col","?"); yc=a.get("y_col","-")
                        st.markdown(f"**{idx}．图表** `{ct}` x=`{xc}` y=`{yc}`")
                        if s.get("sql"): st.code(s["sql"], language="sql")
                        if s.get("code"): st.code(s["code"], language="python")
                    if s.get("error"):
                        st.error(f"⚠️ {s['error']}")

# 输入框
is_deepseek = "DeepSeek" in st.session_state.get("llm_backend", "")
has_key = bool(st.session_state.get("api_key", ""))
disabled = is_deepseek and not has_key

placeholder = (
    "请输入你的数据分析问题..." if not disabled
    else "请先在左侧配置 DeepSeek API Key，或切换到规则模式"
)

prompt = st.session_state.get("ask") or st.chat_input(placeholder, disabled=disabled)

if prompt:
    if st.session_state.get("ask") == prompt:
        st.session_state.ask = None

    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 AI 分析中…"):
            try:
                # 剥离 history 中的自定义字段（只传 role + content）
                clean = [{"role": m["role"], "content": m.get("content", "")}
                          for m in st.session_state.messages]

                result = route(
                    question=prompt,
                    df=df,
                    history=clean,
                    api_key=st.session_state.get("api_key"),
                    base_url=st.session_state.get("api_base_url"),
                    model=st.session_state.get("model_name"),
                )

                # ① 结论
                st.markdown(result["answer"])

                # ② 图表
                if result.get("fig"):
                    st.plotly_chart(result["fig"], use_container_width=True)

                # ③ 分析思路
                steps = result.get("steps", [])
                if steps:
                    with st.expander("🔍 查看分析思路"):
                        st.caption(f"调用了 {len(steps)} 次工具")
                        for idx, s in enumerate(steps, 1):
                            t = s.get("tool", "?")
                            if t == "run_sql":
                                sql = s.get("args",{}).get("sql") or s.get("sql","")
                                if sql: st.code(sql, language="sql")
                            elif t == "run_python":
                                code = s.get("args",{}).get("code") or s.get("code","")
                                if code: st.code(code, language="python")
                            elif t in ("make_chart","fallback"):
                                a = s.get("args",{})
                                st.markdown(f"`{a.get('chart_type','?')}` x=`{a.get('x_col','?')}` y=`{a.get('y_col','-')}`")
                                if s.get("sql"): st.code(s["sql"], language="sql")
                                if s.get("code"): st.code(s["code"], language="python")
                            if s.get("error"): st.error(f"⚠️ {s['error']}")

                # 存历史
                msg_entry = result["msg"]
                if steps: msg_entry["steps"] = steps
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append(msg_entry)

            except Exception as e:
                err = str(e).lower()
                if "401" in err or "unauthorized" in err:
                    st.error("🔑 API Key 无效。请检查或切换到「规则模式」。")
                elif "timeout" in err or "timed" in err:
                    st.error("⏱️ 请求超时，请检查网络。")
                elif "rate" in err:
                    st.error("🔄 请求太频繁，稍等几秒。")
                else:
                    st.error("❌ 处理出错，请重试或换一种问法。")
                    with st.expander("技术详情"):
                        st.code(tb_mod.format_exc())

# 底部
st.divider()
st.caption("Hotel Booking Analysis Agent · 规则模式免 API 可用 · Streamlit Cloud Ready")
