# -*- coding: utf-8 -*-
"""
基于语义检索的电商知识产权风险智能筛查系统
—— 毕设原型系统

当前实现阶段：关键词检索（基于 TF-IDF + 简单中文分词）
后续升级方向：接入 Sentence-BERT / text2vec 等语义模型实现真正的语义检索
"""

import json
import math
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# 0. 页面配置（必须在第一条 st 命令之前）
# ============================================================
st.set_page_config(
    page_title="电商知识产权风险筛查系统",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 1. 数据加载
# ============================================================

@st.cache_data
def load_patents(path: str) -> list[dict]:
    """从 JSON 文件加载专利数据"""
    with open(path, "r", encoding="utf-8") as f:
        patents = json.load(f)
    return patents


def get_data_path() -> str:
    """获取 data/patents.json 的绝对路径"""
    return str(Path(__file__).parent / "data" / "patents.json")

# ============================================================
# 2. 内置中文词典 + 简单分词器（不依赖 jieba）
# ============================================================

# 常用中文词汇表（按分类组织，用于正向最大匹配分词）
_BUILTIN_DICT = [
    # 手机配件类
    "手机壳", "保护壳", "手机配件", "手机支架", "指环扣", "磁吸",
    "支架", "折叠", "蓝牙耳机", "充电盒", "无线充电", "收纳盒", "快充",
    "蓝牙音箱", "户外", "防水", "运动", "音箱",
    "平板支架", "平板电脑", "铝合金", "旋转", "桌面支架",
    "自拍杆", "三脚架", "伸缩", "蓝牙遥控", "拍照",
    "内窥镜", "摄像头", "USB", "管道检测", "高清",
    "磁悬浮", "氛围灯", "LED", "悬浮",
    "散热支架", "笔记本", "散热风扇",
    "充电器", "USB-C", "集线器", "多口充电",
    "鼠标垫", "无线充电", "Qi", "桌面",
    # 个护健康类
    "紫外线", "消毒", "灭菌", "便携", "消毒棒",
    "体脂秤", "体重秤", "BIA", "智能", "健康",
    # 家居用品类
    "保温杯", "温度显示", "温控", "水杯",
    "插座保护盖", "儿童安全", "防触电", "安全", "插座",
    "吸管", "硅胶", "可水洗", "环保", "重复使用",
    "洗手液机", "感应", "自动出液", "红外", "智能家居",
    "水壶", "旅行",
    # 办公用品类
    "收纳架", "办公", "文具架", "组合式",
    # 车载配件类
    "车载支架", "出风口", "磁吸支架",
    "头盔", "电动车", "安全",
    # 宠物用品类
    "宠物喂食器", "自动", "APP控制", "定时",
    # 通用词汇
    "蓝牙", "充电", "电池", "锂电池",
    "手机", "平板", "电脑", "耳机", "音响",
    "玻璃", "塑料", "金属", "硅胶", "橡胶",
    "防水", "防摔", "防尘", "防滑",
    "智能", "APP", "远程", "控制", "传感器",
    "便携", "折叠", "伸缩", "可拆卸", "可调节",
    "LED", "显示屏", "触摸", "按键",
    "家用", "商用", "户外", "室内", "旅行",
    "黑色", "白色", "红色", "蓝色",
    # 常见动词
    "支持", "具有", "包括", "用于", "采用", "配备", "设计",
    "结构", "功能", "系统", "装置", "设备", "组件",
    "表面", "内部", "底部", "顶部", "侧面",
    "方便", "轻松", "快速", "高效",
]

def _build_trie(words: list[str]) -> dict:
    """构建字典树（Trie），用于正向最大匹配分词"""
    trie = {}
    for word in words:
        node = trie
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]
        node["#"] = True  # 标记词尾
    return trie


# 全局字典树
_TRIE = _build_trie(_BUILTIN_DICT)


def tokenize(text: str, max_word_len: int = 6) -> list[str]:
    """
    基于字典树的正向最大匹配中文分词

    参数:
        text: 输入文本
        max_word_len: 最大匹配词长（字符数）
    返回:
        分词结果列表
    """
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        # 跳过非中文字符和空白
        if not ('\u4e00' <= c <= '\u9fff' or 'a' <= c <= 'z'
                or 'A' <= c <= 'Z' or c.isdigit()):
            i += 1
            continue

        # 正向最大匹配：从当前位置尝试匹配最长词
        matched = None
        for end in range(i + 1, min(i + max_word_len + 1, n + 1)):
            sub = text[i:end]
            node = _TRIE
            found = True
            for ch in sub:
                if ch in node:
                    node = node[ch]
                else:
                    found = False
                    break
            if found and "#" in node:
                matched = sub

        if matched:
            tokens.append(matched)
            i += len(matched)
        else:
            i += 1
    return tokens


def preprocess(text: str) -> list[str]:
    """
    文本预处理：小写化 + 中文分词，返回有意义的词汇列表
    """
    text = text.lower()
    tokens = tokenize(text)
    return [t for t in tokens if len(t) >= 2]

# ============================================================
# 3. 基于 numpy 的 TF-IDF 实现（不依赖 scikit-learn）
# ============================================================

class SimpleTfidfVectorizer:
    """简化的 TF-IDF 向量化器，纯 numpy 实现"""

    def __init__(self):
        self.vocab_: dict[str, int] = {}   # 词 -> 列索引
        self.idf_: np.ndarray = None        # IDF 向量
        self.n_docs_: int = 0

    def fit_transform(self, corpus: list[list[str]]) -> np.ndarray:
        """拟合语料并返回 TF-IDF 矩阵"""
        vocab_set = set()
        doc_freq = Counter()
        for doc in corpus:
            unique_terms = set(doc)
            vocab_set.update(unique_terms)
            for term in unique_terms:
                doc_freq[term] += 1

        self.vocab_ = {t: i for i, t in enumerate(sorted(vocab_set))}
        self.n_docs_ = len(corpus)
        n_features = len(self.vocab_)

        # 计算平滑 IDF：log((1 + n) / (1 + df)) + 1
        self.idf_ = np.zeros(n_features)
        for term, idx in self.vocab_.items():
            df = doc_freq.get(term, 0)
            self.idf_[idx] = math.log((1 + self.n_docs_) / (1 + df)) + 1

        # 计算 TF-IDF 矩阵（sublinear TF: 1 + log(tf)）
        matrix = np.zeros((self.n_docs_, n_features), dtype=np.float64)
        for doc_idx, doc in enumerate(corpus):
            tf = Counter(doc)
            for term, freq in tf.items():
                if term in self.vocab_:
                    col = self.vocab_[term]
                    tf_val = 1 + math.log(freq) if freq > 0 else 0
                    matrix[doc_idx, col] = tf_val * self.idf_[col]

        # L2 归一化
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def transform(self, docs: list[list[str]]) -> np.ndarray:
        """将新文档转换为 TF-IDF 向量"""
        n_features = len(self.vocab_)
        matrix = np.zeros((len(docs), n_features), dtype=np.float64)

        for doc_idx, doc in enumerate(docs):
            tf = Counter(doc)
            for term, freq in tf.items():
                if term in self.vocab_:
                    col = self.vocab_[term]
                    tf_val = 1 + math.log(freq) if freq > 0 else 0
                    matrix[doc_idx, col] = tf_val * self.idf_[col]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

# ============================================================
# 4. 检索引擎构建
# ============================================================

@st.cache_resource
def build_search_engine(patents: list[dict]):
    """
    构建关键词检索引擎（TF-IDF + 余弦相似度）

    第一阶段实现，后续可替换为语义编码模型。
    """
    # 将标题 + 关键词 + 描述拼接为检索文本
    corpus_raw = [
        p["title"] + " " + " ".join(p["keywords"]) + " " + p["description"]
        for p in patents
    ]
    # 分词
    corpus_tokens = [preprocess(doc) for doc in corpus_raw]
    # 向量化
    vectorizer = SimpleTfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_tokens)
    return vectorizer, tfidf_matrix, corpus_raw


# ============================================================
# 5. 检索与风险评估
# ============================================================

def compute_risk(score: float) -> tuple[str, str]:
    """
    根据相似度得分划分风险等级

    - 高（>= 0.45）：建议立即审查
    - 中（>= 0.25）：需要人工判断
    - 低（>= 0.10）：风险较低
    - 无（< 0.10）：未发现明显相似专利
    """
    if score >= 0.45:
        return "高风险", "\U0001f534"
    elif score >= 0.25:
        return "中风险", "\U0001f7e0"
    elif score >= 0.10:
        return "低风险", "\U0001f7e1"
    else:
        return "无风险", "\U0001f7e2"


def search(query: str, patents: list[dict],
            vectorizer, tfidf_matrix, top_k: int = 10):
    """执行检索：对用户输入做向量化，计算与所有专利的余弦相似度"""
    query_tokens = preprocess(query)
    if not query_tokens:
        return []

    query_vec = vectorizer.transform([query_tokens])
    # L2 归一化后，点积即余弦相似度
    similarities = np.dot(query_vec, tfidf_matrix.T)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        risk_label, risk_icon = compute_risk(score)
        results.append({
            "patent": patents[idx],
            "similarity": score,
            "risk_label": risk_label,
            "risk_icon": risk_icon,
        })
    return results

# ============================================================
# 6. 主界面
# ============================================================

def main():
    # ---------- 侧边栏 ----------
    with st.sidebar:
        st.markdown("### \U0001f50d 系统状态")
        data_path = get_data_path()
        patents = load_patents(data_path)
        st.success(f"专利库已加载：**{len(patents)}** 条")

        vectorizer, tfidf_matrix, _ = build_search_engine(patents)
        st.info("\U0001f4c4 检索模式：关键词 / TF-IDF（第一阶段）")
        st.caption(f"词表大小：{len(vectorizer.vocab_)} 个特征词")

        st.markdown("---")
        st.markdown("### \U0001f4d6 检索说明")
        st.markdown(
            "- 在右侧输入框中描述你要检索的商品\n"
            "- 系统会与专利库做相似度匹配\n"
            "- 根据相似度划分 **高 / 中 / 低 / 无** 四个风险等级\n"
            "- 目前使用 **TF-IDF + 余弦相似度** 实现关键词检索\n"
            "- 后续可接入语义模型（Sentence-BERT 等）"
        )
        st.markdown("---")
        st.markdown("### \U0001f527 风险等级说明")
        st.markdown(
            "- \U0001f534 **高风险**（≥ 0.45）：建议立即审查\n"
            "- \U0001f7e0 **中风险**（≥ 0.25）：需要人工判断\n"
            "- \U0001f7e1 **低风险**（≥ 0.10）：风险较低\n"
            "- \U0001f7e2 **无风险**（< 0.10）：未发现明显相似专利"
        )

    # ---------- 主面板 ----------
    st.markdown("# \U0001f50d 电商知识产权风险筛查系统")
    st.markdown("基于语义检索（第一阶段：关键词/TF-IDF）的专利侵权风险智能筛查")

    # 搜索输入区
    col1, col2 = st.columns([6, 1])
    with col1:
        query = st.text_input(
            "请输入商品描述：",
            placeholder="例如：一款带支架的硅胶手机保护壳，背面有可折叠支架...",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.number_input(
            "展示数量", min_value=3, max_value=22, value=10,
            label_visibility="collapsed",
        )
        st.caption("展示数量")

    search_clicked = st.button(
        "\U0001f50d 开始检索", type="primary", use_container_width=True
    )

    if search_clicked and query.strip():
        with st.spinner("正在检索专利库..."):
            results = search(query, patents, vectorizer, tfidf_matrix, top_k)

        if not results:
            st.warning("未找到与输入描述相关的专利，请尝试修改关键词后重试。")
            return

        # ---------- 统计概览 ----------
        st.markdown("---")
        st.markdown("### \U0001f4ca 检索结果概览")

        risk_counts = {"高风险": 0, "中风险": 0, "低风险": 0, "无风险": 0}
        for r in results:
            risk_counts[r["risk_label"]] += 1

        cols = st.columns(4)
        colors = {
            "高风险": "#ff4b4b", "中风险": "#ffa421",
            "低风险": "#21c354", "无风险": "#1c83e1",
        }
        for i, (level, count) in enumerate(risk_counts.items()):
            with cols[i]:
                st.markdown(
                    f"<div style='background:{colors[level]}; padding:10px; "
                    f"border-radius:8px; text-align:center; color:white;'>"
                    f"<span style='font-size:24px; font-weight:bold;'>{count}</span><br>"
                    f"<span>{level}</span></div>",
                    unsafe_allow_html=True,
                )

        # ---------- 详细结果列表 ----------
        st.markdown("---")
        st.markdown("### \U0001f4cb 详细匹配结果")

        for i, r in enumerate(results):
            p = r["patent"]
            score_pct = r["similarity"] * 100
            with st.container():
                cols = st.columns([0.05, 0.7, 0.15, 0.1])
                with cols[0]:
                    st.markdown(f"**{i + 1}**")
                with cols[1]:
                    desc = p["description"]
                    if len(desc) > 80:
                        desc = desc[:80] + "..."
                    st.markdown(
                        f"**{p['title']}**  \n"
                        f"<span style='color:#888; font-size:0.9em;'>{desc}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"专利号：{p['patent_id']}  |  分类：{p['category']}")
                with cols[2]:
                    sim_val = min(r["similarity"], 1.0)
                    st.markdown(
                        f"<div style='text-align:center;'>"
                        f"<span style='font-size:1.2em; font-weight:bold;'>"
                        f"{score_pct:.1f}%</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.progress(sim_val)
                with cols[3]:
                    st.markdown(
                        f"<div style='text-align:center; padding-top:10px;'>"
                        f"<span style='font-size:1.5em;'>{r['risk_icon']}</span><br>"
                        f"<span style='font-weight:bold;'>{r['risk_label']}</span></div>",
                        unsafe_allow_html=True,
                    )
                if i < len(results) - 1:
                    st.markdown(
                        "<hr style='margin:8px 0; opacity:0.3;'>",
                        unsafe_allow_html=True,
                    )

        # ---------- 导出功能 ----------
        st.markdown("---")
        with st.expander("\U0001f4e5 导出检索报告"):
            df_export = pd.DataFrame([
                {
                    "序号": i + 1,
                    "专利号": r["patent"]["patent_id"],
                    "专利标题": r["patent"]["title"],
                    "分类": r["patent"]["category"],
                    "相似度": f"{r['similarity'] * 100:.1f}%",
                    "风险等级": r["risk_label"],
                }
                for i, r in enumerate(results)
            ])
            st.dataframe(df_export, use_container_width=True, hide_index=True)
            csv = df_export.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="\U0001f4e5 下载 CSV 报告",
                data=csv,
                file_name="patent_risk_report.csv",
                mime="text/csv",
            )

    elif search_clicked and not query.strip():
        st.warning("请输入商品描述后再进行检索。")


# ============================================================
# 7. 入口
# ============================================================

if __name__ == "__main__":
    main()
