import streamlit as st
import pandas as pd

st.set_page_config(page_title="商品专利风险筛查系统", page_icon="🛡️", layout="wide")

st.title("🛡️ 商品专利风险筛查系统")
st.write("输入商品描述，系统将基于关键词匹配判断专利侵权风险")

# ---------- 专利数据 ----------
PATENTS = [
    {"id": "CN001", "title": "石墨烯发热护膝", "abstract": "一种石墨烯发热护膝，包括护膝本体和石墨烯发热片，通过USB供电加热", "category": "智能穿戴"},
    {"id": "CN002", "title": "纳米保温杯", "abstract": "一种纳米材料保温杯，内壁涂覆纳米陶瓷涂层，具有长效保温功能", "category": "家居用品"},
    {"id": "CN003", "title": "红外理疗仪", "abstract": "一种红外理疗仪，包括碳纤维红外加热管和温度控制系统", "category": "个护健康"},
    {"id": "CN004", "title": "快充充电宝", "abstract": "一种智能便携充电宝，支持快充协议，具有过充过放保护功能", "category": "数码配件"},
    {"id": "CN005", "title": "磁疗保健鞋垫", "abstract": "一种磁疗保健鞋垫，对应人体足底穴位分布永磁体，具有按摩保健功能", "category": "个护健康"},
    {"id": "CN006", "title": "石墨烯电池导热膜", "abstract": "一种石墨烯导热膜，用于电池组散热和温度均衡", "category": "新能源"},
    {"id": "CN007", "title": "蓝牙降噪耳机", "abstract": "一种蓝牙降噪耳机，支持主动降噪和触控操作", "category": "数码配件"},
    {"id": "CN008", "title": "碳纤维电热毯", "abstract": "一种碳纤维电热毯，具有均匀发热和安全保护功能", "category": "家居用品"},
    {"id": "CN009", "title": "车载磁吸手机支架", "abstract": "一种车载磁吸手机支架，强磁吸附，单手操作", "category": "车载配件"},
    {"id": "CN010", "title": "智能宠物喂食器", "abstract": "一种智能宠物喂食器，支持定时喂食和远程控制", "category": "宠物用品"},
    {"id": "CN011", "title": "防摔硅胶手机壳", "abstract": "一种防摔硅胶手机壳，四角加厚气囊设计，有效缓冲冲击力", "category": "手机配件"},
    {"id": "CN012", "title": "紫外线消毒棒", "abstract": "一种手持式紫外线消毒棒，用于日常物品杀菌消毒", "category": "个护健康"},
    {"id": "CN013", "title": "无线蓝牙音箱", "abstract": "一种便携式无线蓝牙音箱，支持IPX7级防水", "category": "数码配件"},
    {"id": "CN014", "title": "智能体脂秤", "abstract": "一种智能体脂秤，可测量BMI、体脂率等多项身体数据", "category": "个护健康"},
    {"id": "CN015", "title": "桌面多功能收纳架", "abstract": "一种桌面多功能收纳架，可放置手机、文具、眼镜等物品", "category": "办公用品"},
]

# ---------- 检索函数 ----------
def search_patents(query, top_k=5):
    if not query or len(query.strip()) == 0:
        return []

    query_clean = query.lower().strip()
    results = []

    for p in PATENTS:
        text = (p["title"] + " " + p["abstract"]).lower()

        if query_clean in text:
            score = 0.90
        else:
            chars = list(query_clean)
            matched = 0
            for c in chars:
                if c in text:
                    matched += 1
            if matched == 0:
                continue
            score = min(matched / len(chars) * 0.7 + 0.2, 0.85)

        results.append({
            "专利号": p["id"],
            "专利标题": p["title"],
            "摘要": p["abstract"],
            "品类": p["category"],
            "相似度": round(score, 3),
            "风险等级": "高风险" if score >= 0.7 else ("中风险" if score >= 0.4 else "低风险")
        })

    results.sort(key=lambda x: x["相似度"], reverse=True)
    return results[:top_k]

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("📊 系统信息")
    st.write(f"专利库数量：**{len(PATENTS)}** 条")
    st.write(f"覆盖品类：**{len(set(p['category'] for p in PATENTS))}** 个")

    # 检索历史
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.write("📋 **检索历史**")
        for h in st.session_state.history[-5:]:
            st.write(f"- {h}")

# ---------- 初始化历史记录 ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- 主界面 ----------
query = st.text_input("请输入商品描述", placeholder="例如：蓝牙耳机、石墨烯发热护膝")

if st.button("🔍 开始筛查"):
    if not query or len(query.strip()) == 0:
        st.warning("⚠️ 请输入商品描述")
    else:
        # 记录检索历史
        if query not in st.session_state.history:
            st.session_state.history.append(query)

        with st.spinner("正在检索..."):
            results = search_patents(query)

        if results:
            st.subheader(f"📊 检索结果（共找到 {len(results)} 条相关专利）")

            # 导出报告按钮
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 导出检索报告 (CSV)",
                data=csv,
                file_name=f"专利检索报告_{query}.csv",
                mime="text/csv",
            )

            st.markdown("---")

            for i, r in enumerate(results):
                with st.expander(f"#{i+1} {r['专利标题']}  —  相似度：{r['相似度']*100:.1f}%  {r['风险等级']}"):
                    st.write(f"**专利号：** {r['专利号']}")
                    st.write(f"**摘要：** {r['摘要']}")
                    st.write(f"**品类：** {r['品类']}")
        else:
            st.info("未找到匹配的专利，建议更换关键词尝试。")
