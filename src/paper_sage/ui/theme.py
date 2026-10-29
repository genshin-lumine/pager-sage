"""工作台外观与静态引导；动态文本先转义，不参与业务流程。"""

from html import escape
from pathlib import Path

import streamlit as st


STYLES = """
<style>
:root {
    --sage-ink: #233c36;
    --sage-muted: #62746d;
    --sage-green: #25634e;
    --sage-line: #dce5df;
    --sage-paper: #ffffff;
}
.stApp { background: #f6f8f5; color: var(--sage-ink); }
[data-testid="stHeader"] { background: rgba(246,248,245,.96); }
[data-testid="stMainBlockContainer"] {
    max-width: 1140px; padding: 4.7rem 3rem 3rem;
}
[data-testid="stSidebar"] {
    background: #edf2ed; border-right: 1px solid var(--sage-line);
}
[data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .8rem; }
.sage-brand { display: flex; align-items: center; gap: 11px; margin-bottom: 1.1rem; }
.sage-brand-mark {
    display: grid; place-items: center; width: 38px; height: 38px;
    background: var(--sage-green); color: white; border-radius: 12px;
    font-family: Georgia, serif; font-size: 24px; font-style: italic;
}
.sage-brand strong { font-size: 19px; letter-spacing: -.6px; }
.sage-brand small { display: block; color: var(--sage-muted); font-size: 11px; letter-spacing: 2px; }
.sage-hero { padding: .2rem 0 1.3rem; border-bottom: 1px solid var(--sage-line); margin-bottom: .4rem; }
.sage-eyebrow { color: var(--sage-green); font-size: 11px; letter-spacing: 2.4px; font-weight: 700; }
.sage-hero h1 { font-size: clamp(27px, 3vw, 38px); letter-spacing: -1.3px; line-height: 1.25; margin: 12px 0; padding: 0; }
.sage-hero p { color: var(--sage-muted); font-size: 14px; margin: 0; line-height: 1.8; }
.sage-scope { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; padding: .6rem 0 .2rem; }
.sage-project { font-weight: 650; margin-right: auto; overflow-wrap: anywhere; }
.sage-pill { font-size: 12px; padding: 5px 11px; border: 1px solid var(--sage-line); border-radius: 99px; background: var(--sage-paper); white-space: nowrap; }
.sage-dot { display: inline-block; height: 7px; width: 7px; border-radius: 50%; background: #4b8d67; margin-right: 7px; }
.sage-files { display: flex; flex-wrap: wrap; gap: 6px; padding: .2rem 0 .7rem; }
.sage-file { font-size: 11px; color: var(--sage-muted); background: #eaf0e9; border-radius: 5px; padding: 4px 8px; overflow-wrap: anywhere; max-width: 100%; }
.sage-welcome { text-align: center; padding: 2rem 0 1.2rem; }
.sage-welcome h2 { font-size: 23px; font-weight: 600; letter-spacing: -.5px; margin: 0 0 8px; }
.sage-welcome p { color: var(--sage-muted); font-size: 13px; margin: 0; }
.sage-prompts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: .5rem 0 1.6rem; }
.sage-prompt { background: var(--sage-paper); border: 1px solid var(--sage-line); border-radius: 12px; padding: 17px 18px; }
.sage-prompt small { color: var(--sage-green); font-size: 10px; letter-spacing: 1.7px; }
.sage-prompt h3 { font-size: 14px; font-weight: 650; margin: 9px 0 6px; padding: 0; }
.sage-prompt p { color: var(--sage-muted); font-size: 12px; line-height: 1.7; margin: 0; }
.st-key-paper_composer {
    border: 1px solid #ccdcd0; border-radius: 16px; padding: 20px 22px;
    background: var(--sage-paper); box-shadow: 0 8px 28px rgba(35,60,54,.04);
}
.sage-composer-heading { font-size: 14px; font-weight: 650; margin-bottom: 3px; }
.sage-composer-heading span { font-size: 11px; color: var(--sage-muted); font-weight: 400; margin-left: 10px; }
.st-key-paper_actions { margin-top: 2px; }
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button { border-radius: 9px; min-height: 40px; }
[data-testid="stButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
    background: var(--sage-green); border-color: var(--sage-green); color: white;
}
[data-testid="stButton"] button[kind="primary"]:hover { background: #1b513e; border-color: #1b513e; }
[data-testid="stTextInput"] input { font-size: 14px; }
[data-testid="stExpander"] { border-color: var(--sage-line); border-radius: 10px; background: rgba(255,255,255,.6); }
[data-testid="stExpander"] summary { font-size: 13px; }
[data-testid="stChatMessage"] { background: var(--sage-paper); border: 1px solid var(--sage-line); border-radius: 15px; padding: 22px; margin-bottom: 10px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #edf4ee; border-color: #d9e5db; }
[data-testid="stChatMessage"] p { line-height: 1.85; }
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] p { line-height: 1.65; overflow-wrap: anywhere; }
[data-testid="stCode"] { border: 1px solid #e3e9e2; border-radius: 9px; }
[data-testid="stCaptionContainer"] { color: var(--sage-muted); }
.sage-footnote { font-size: 11px; color: var(--sage-muted); text-align: center; padding: 13px 0; }
@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"] { padding: 4.4rem 1rem 2rem; }
    .sage-prompts { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .sage-prompt { padding: 12px 9px; }
    .sage-prompt small { font-size: 8px; letter-spacing: .6px; }
    .sage-prompt h3 { font-size: 12px; line-height: 1.6; margin: 6px 0; }
    .sage-prompt p { display: none; }
    .sage-welcome { padding-top: 1rem; }
    .sage-welcome h2 { font-size: 20px; }
    .st-key-paper_composer { padding: 15px; }
    [data-testid="stChatMessage"] { padding: 15px; }
    .sage-composer-heading span { display: block; margin: 4px 0 0; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
"""


def apply_theme():
    st.html(STYLES)


def render_sidebar_brand():
    with st.sidebar:
        st.html('<div class="sage-brand"><span class="sage-brand-mark" aria-hidden="true">p</span>'
                '<div><strong>Paper Sage</strong><small>论文研究工作台</small></div></div>')


def render_workspace_header():
    st.html('<header class="sage-hero"><div class="sage-eyebrow">PAPER SAGE / RESEARCH WORKSPACE</div>'
            '<h1>让阅读更深入，让依据更清晰。</h1>'
            '<p>围绕你的论文提问、总结与比较，每一步都能回到原文。</p></header>')


def render_scope_overview(project, documents, chunk_count):
    project_name = escape(project["name"])
    st.html(f'<div class="sage-scope"><div class="sage-project"><span class="sage-dot"></span>{project_name}</div>'
            f'<span class="sage-pill">已选 {len(documents)} / {len(project["documents"])} 篇论文</span>'
            f'<span class="sage-pill">{chunk_count} 个可检索片段</span></div>'
            '<div class="sage-files">' + ''.join(
                f'<span class="sage-file">{escape(Path(doc["source"]).name)}</span>' for doc in documents
            ) + '</div>')


def render_empty_conversation():
    st.html('<div class="sage-welcome"><h2>今天，想从论文里找到什么？</h2>'
            '<p>在下方输入问题。你可以从一个细节开始，也可以梳理全篇。</p></div>'
            '<div class="sage-prompts">'
            '<article class="sage-prompt"><small>01 / ASK</small><h3>追问一个细节</h3><p>这篇论文用了哪些训练数据？</p></article>'
            '<article class="sage-prompt"><small>02 / SUMMARIZE</small><h3>梳理论文全貌</h3><p>概括这篇论文的方法和主要结论。</p></article>'
            '<article class="sage-prompt"><small>03 / COMPARE</small><h3>对照两篇研究</h3><p>比较这两篇论文的训练数据与结果。</p></article>'
            '</div>')
