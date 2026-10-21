import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

st.set_page_config(page_title="商品专利风险初筛", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --brand: #0f766e;
        --brand-dark: #0b5f59;
        --ink: #17232d;
        --muted: #66737d;
        --line: #dfe7e9;
        --surface: #ffffff;
        --canvas: #f4f7f7;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: var(--ink);
        letter-spacing: 0;
    }

    h1 {
        font-size: 2.15rem;
        font-weight: 650;
        margin-bottom: 0.2rem;
    }

    label, .stMarkdown p {
        color: var(--ink);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] .stMetric {
        background: #f8fafa;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 1.45rem;
    }

    [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
    }

    [data-testid="stExpander"] summary {
        font-weight: 600;
    }

    .stButton > button[kind="primary"],
    [data-testid="stBaseButton-primary"] {
        background: var(--brand) !important;
        border: 1px solid var(--brand) !important;
        border-radius: 8px;
        color: #ffffff !important;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(15, 118, 110, 0.18);
    }

    .stButton > button[kind="primary"] *,
    [data-testid="stBaseButton-primary"] *,
    [data-testid="stBaseButton-primary"] p,
    [data-testid="stBaseButton-primary"] span {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    .stButton > button[kind="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: var(--brand-dark) !important;
        border-color: var(--brand-dark) !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover p,
    .stButton > button[kind="primary"]:hover span,
    [data-testid="stBaseButton-primary"]:hover p,
    [data-testid="stBaseButton-primary"]:hover span {
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:focus-visible,
    [data-testid="stBaseButton-primary"]:focus-visible {
        outline: none;
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.2);
    }

    [data-testid="stDownloadButton"] button,
    .stDownloadButton button {
        background: #e7f4f1 !important;
        border: 1px solid #85bdb3 !important;
        border-radius: 8px;
        color: var(--brand-dark) !important;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(15, 118, 110, 0.08);
    }

    [data-testid="stDownloadButton"] button *,
    .stDownloadButton button * {
        color: var(--brand-dark) !important;
        opacity: 1 !important;
    }

    [data-testid="stDownloadButton"] button:hover,
    .stDownloadButton button:hover {
        background: #d7ece7 !important;
        border-color: var(--brand) !important;
    }

    [data-testid="stLinkButton"] a,
    .stLinkButton a {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface);
        color: var(--brand-dark) !important;
        font-weight: 600;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--line) !important;
        border-radius: 8px;
    }

    .stTextInput input {
        background: var(--surface);
        border-color: var(--line);
        color: var(--ink);
        border-radius: 8px;
    }

    .stTextInput input:focus {
        border-color: var(--brand);
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
    }

    [data-testid="stProgress"] > div > div > div {
        background-color: var(--brand) !important;
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
    }

    a {
        color: var(--brand-dark);
    }

    @media (max-width: 760px) {
        .block-container {
            padding-top: 1.4rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 1.75rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("商品专利风险初筛")
st.caption("快速识别商品描述中的相关专利线索，辅助业务团队开展风险复核。")

# ===== 通义千问 API 配置 =====
DASHSCOPE_API_KEY = st.secrets["DASHSCOPE_API_KEY"]
DASHSCOPE_BASE_URL = st.secrets["DASHSCOPE_BASE_URL"]

# 检索策略：最多返回 5 条，低相关结果不再为了凑满条数而展示
MAX_RESULTS = 5
MIN_RELEVANCE_SCORE = 0.45
MAX_QUERY_CHARS = 200
QUERY_COOLDOWN_SECONDS = 3

# 品类约束只使用明确的商品/场景词，避免把“石墨烯发热护膝”误判为新能源电池
CATEGORY_KEYWORDS = {
    "家居用品": ["保温杯", "水杯", "杯子", "饮水", "药盒", "杯盖"],
    "数码配件": ["蓝牙耳机", "耳机", "耳塞", "充电仓"],
    "车载配件": ["车载支架", "车载", "吸盘", "仪表台"],
    "手机配件": ["手机壳", "保护壳", "镜头盖", "手机背板"],
    "个护健康": ["理疗仪", "理疗", "艾灸", "护膝", "按摩仪", "美容仪"],
    "新能源": ["石墨烯电池", "电池", "充电", "新能源"],
    "智能穿戴": ["智能穿戴", "可穿戴", "智能手表", "手表", "手环"],
}


def patent_detail_url(patent_id):
    """返回国家知识产权局专利检索平台入口。"""
    return "https://pss-system.cponline.cnipa.gov.cn/"


def infer_query_categories(query):
    """根据明确的商品词推断候选品类；未命中时返回空集合。"""
    text = (query or "").strip().lower()
    if not text:
        return set()

    matched = set()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            matched.add(category)
    return matched


def get_embeddings_batch(texts, batch_size=20):
    """批量调用通义千问 Embedding API，失败时重试一次。"""
    if not texts:
        return None

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "qwen3.7-text-embedding",
            "input": {"texts": batch},
            "dimensions": 1024
        }

        batch_succeeded = False
        for attempt in range(2):
            try:
                response = requests.post(
                    f"{DASHSCOPE_BASE_URL}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=(5, 30)
                )

                if response.status_code == 200:
                    result = response.json()
                    items = result.get("data", [])
                    if len(items) != len(batch):
                        print("Embedding response count mismatch")
                        st.error("分析服务返回异常，请稍后重试。")
                        return None

                    for item in sorted(items, key=lambda x: x.get("index", 0)):
                        all_embeddings.append(
                            np.array(item["embedding"], dtype=np.float32)
                        )
                    batch_succeeded = True
                    break

                if attempt == 0 and response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1)
                    continue

                print(f"Embedding request failed: HTTP {response.status_code}")
                st.error("智能分析服务暂时不可用，请稍后重试。")
                return None

            except (requests.RequestException, ValueError, KeyError, TypeError):
                if attempt == 0:
                    time.sleep(1)
                    continue
                print("Embedding request timeout or invalid response")
                st.error("智能分析服务暂时不可用，请稍后重试。")
                return None

        if not batch_succeeded:
            return None

    return all_embeddings


# ---------- 专利数据（50条，覆盖7个品类） ----------
PATENTS = [
    # ===== 保温杯（8条） =====
    {"id": "CN122460793A", "title": "一种保温杯", "abstract": "本发明公开了一种保温杯，包括可拆卸连接的杯体和杯盖，杯盖顶部凹设有第一容置空间，盖体可开合地盖设在第一容置空间的开口处，药盒容置在第一容置空间内。解决了老年用户在外出场景下药盒易遗忘的技术问题。", "category": "家居用品"},
    {"id": "CN122478363A", "title": "一种保温杯", "abstract": "本发明公开了一种保温杯，包括杯体和杯盖，杯盖上设有出水口和排气口，排气口内设有泄压组件，出水口下方可拆卸连接有水箱。具有能够规避烫伤风险并避免卫生隐患的优点。", "category": "家居用品"},
    {"id": "CN224343351U", "title": "一种蓝牙耳机", "abstract": "本实用新型公开一种蓝牙耳机，包括底盖、转轴底座及耳挂，底盖与转轴底座之间以绕底盖的垂直轴线方向转动连接，耳挂包括挂耳部及连接部，通过两个方向的转动实现稳固固定，适应耳廓角度。", "category": "数码配件"},
    {"id": "CN224459957U", "title": "一种用于运动的防脱落蓝牙耳机", "abstract": "本实用新型公开一种用于运动的防脱落蓝牙耳机，包括运动发带和发音头，运动发带上固定设置有蓝牙耳机主体和连接布筒，运动发带为运动时所穿戴在头部的发带，能够帮助用户吸汗并防止耳机脱落。", "category": "数码配件"},
    {"id": "CN222736255U", "title": "一种入耳式可降噪蓝牙耳机", "abstract": "本实用新型涉及蓝牙耳机技术领域，公开了一种入耳式可降噪蓝牙耳机，包括耳机主体、耳塞连接管和气泵，耳塞连接管外表面固定安装有气囊，方便使用者调节隔音结构大小。", "category": "数码配件"},
    {"id": "CN220673912U", "title": "一种方便清洗的蓝牙耳机", "abstract": "本实用新型公开了一种方便清洗的蓝牙耳机，包括蓝牙耳机主体，主体上端活动安装有防护盖，驱动箱上端活动安装有盖板，通过清理机构便于清理耳机及耳机槽内附着的污垢，提高清洁效率。", "category": "数码配件"},
    {"id": "CN224401647U", "title": "可识别姿态的蓝牙耳机", "abstract": "本实用新型公开了可识别姿态的蓝牙耳机，包括外壳，外壳的正面开设有通槽，内部设置有可识别姿态的传感器，传音管活动贯穿于通槽内部，传音管一端粘贴有胶套，另一端固定有引导块。", "category": "数码配件"},
    {"id": "CN224610888U", "title": "一种防滑的蓝牙耳机", "abstract": "本实用新型公开了一种防滑的蓝牙耳机，包括耳机外机壳和套环握把，耳机管的外部包裹有防滑包套，套环握把设置于耳机外机壳的左右两端，提高蓝牙耳机使用时的抓握稳定性，减少摔落可能性。", "category": "数码配件"},
    {"id": "CN224385664U", "title": "一种磁吸式可拆卸电池的蓝牙耳机", "abstract": "本实用新型涉及蓝牙耳机技术领域，公开了一种磁吸式可拆卸电池的蓝牙耳机，包括耳机壳体、电池柱及充电仓，通过磁吸式可拆卸设计简化电池更换流程，用户无需专业工具即可自主更换电池，延长耳机使用寿命。", "category": "数码配件"},
    {"id": "CN224329570U", "title": "一种智能运动蓝牙耳机", "abstract": "本实用新型涉及蓝牙耳机技术领域，公开了一种智能运动蓝牙耳机，包括壳体和盖体，壳体上设有带USB接头的电源线，电源线的USB接头穿过盖体，具备充电线一体化收纳以及盒盖稳固闭合的优点。", "category": "数码配件"},
    {"id": "CN224626753U", "title": "一种开放式蓝牙耳机", "abstract": "本实用新型提供了一种开放式蓝牙耳机，包括耳机外壳，耳机外壳内设有蓝牙连接模块，开放式耳塞壳内设有发声单元，声音传导组件包括导向管，通过插拔导向管可在开放式与入耳式切换，适应嘈杂环境。", "category": "数码配件"},
    # ===== 车载支架（6条） =====
    {"id": "CN120792690A", "title": "一种用于移动智能终端的车载支架", "abstract": "本发明公开了一种用于移动智能终端的车载支架，涉及车载支架技术领域，通过可伸缩的设计能够安装在不同尺寸的汽车中控凸体显示屏上，避免现有支架专车专用的局限性，提高了支架的适应性。", "category": "车载配件"},
    {"id": "CN224447661U", "title": "一种多功能车载支架", "abstract": "本实用新型提供一种多功能车载支架，具备由一对夹持臂构造出的第一夹持空间和第二夹持空间，分别适于夹持固定灭火器，以及柄部容置槽，适于置入带柄工具的柄部，增强灭火能力，降低火灾损失。", "category": "车载配件"},
    {"id": "CN224447669U", "title": "车载支架", "abstract": "本实用新型公开了一种车载支架，涉及支架技术领域，包括底座、折叠架和吸盘组件，第一按压组件将吸盘组件从仪表台上快速提拉起来，使吸盘组件与紧贴的平面之间产生真空腔，达到快速吸附的目的。", "category": "车载配件"},
    {"id": "CN224617598U", "title": "一种车载支架", "abstract": "本实用新型公开一种车载支架，包括支架主体、夹持组件、安装件、夹持驱动模组和旋转驱动模组，旋转驱动模组驱动支架主体相对于安装件转动，使电子设备可在竖屏状态和横屏状态之间切换，提升便利性。", "category": "车载配件"},
    {"id": "CN224465788U", "title": "车载支架", "abstract": "一种车载支架，包括壳体、控制电路板、夹持组件、散热模块和控制键，通过控制键启动散热模块对固定在车载支架的电子设备进行散热制冷，使电子设备不易发烫和卡顿。", "category": "车载配件"},
    {"id": "CN224385551U", "title": "一种具有物理滑动式镜头保护组件的手机壳", "abstract": "本实用新型提供一种具有物理滑动式镜头保护组件的手机壳，包括手机壳，手机壳背部开设有镜孔，镜孔两端依次设有上防护盖和下防护盖，上防护盖顶端转动连接有旋转件，下防护盖和上防护盖可拆卸连接。", "category": "手机配件"},
    # ===== 手机壳（8条） =====
    {"id": "CN224401564U", "title": "手机壳", "abstract": "本申请提供一种手机壳，包括本体、电子纸、第一磁吸件、第二磁吸件、电路板、NFC天线和闪光件，闪光件用于在电子纸刷新时闪烁，即时直观地提示刷新状态，提高用户体验，兼顾外观简洁性。", "category": "手机配件"},
    {"id": "CN224459861U", "title": "一种多元素镜头盖手机壳", "abstract": "本实用新型涉及一种手机壳，具体为一种多元素镜头盖手机壳，包括镜头盖以及固定于镜头盖上的背板，由PC材质的背板和镜头盖通过套啤工艺注塑成型组合，可以有效提高镜头盖位置的强度。", "category": "手机配件"},
    {"id": "CN224626686U", "title": "一体式隐形支架手机壳", "abstract": "本实用新型涉及手机壳技术领域，提供了一体式隐形支架手机壳，通过设置等距且平行的错位槽线结构，实现了主支板和附支板的功能性组合，使得手机支架的支撑角度可以在30-120度范围内自由调整。", "category": "手机配件"},
    {"id": "CN224356158U", "title": "一种可牵引宠物的手机壳", "abstract": "本实用新型涉及一种可牵引宠物的手机壳，包括设置于手机壳背面的绳壳，绳壳内部设置有可转动的转盘，转盘由固定组件锁定或解锁，转盘上设有可延伸出绳壳的牵引绳，增加遛宠物的乐趣与安全性。", "category": "手机配件"},
    {"id": "CN224385556U", "title": "一种手提件支架手机壳", "abstract": "本实用新型涉及手机壳技术领域，公开了一种手提件支架手机壳，通过按压滑动打开的结构实现手提件的滑出，连接软带是软体材料可以让手机横竖都能使用，不使用时通过两组磁铁对吸自动回位。", "category": "手机配件"},
    {"id": "CN224555662U", "title": "一种具有支撑功能的手机壳", "abstract": "本实用新型涉及手机壳技术领域，具体为一种具有支撑功能的手机壳，通过手机壳本体集成安装槽、固定杆、支撑板等部件，构建可调节、强锁止的支撑体系，实现角度调节和多场景适配。", "category": "手机配件"},
    {"id": "CN224329497U", "title": "多轴式旋转支架手机壳", "abstract": "本实用新型提供一种多轴式旋转支架手机壳，包含壳体和支架本体，支架本体具有多轴旋转机构，包含第一枢轴及第二枢轴，通过双重翻转保护支架本体避免断裂或损坏，增强手机壳的支撑耐用性。", "category": "手机配件"},
    {"id": "CN224367882U", "title": "一种多模式高效散热且可适配外部制冷设备的手机壳", "abstract": "本实用新型属于手机壳技术领域，提供了一种多模式高效散热且可适配外部制冷设备的手机壳，通过多种散热方式的有机结合实现高效散热，包括进风口、出风口、散热器卡扣和导热铜片。", "category": "手机配件"},
    # ===== 理疗仪（8条） =====
    {"id": "CN122399248A", "title": "一种具有柔性电极的中频脉冲理疗仪", "abstract": "本发明公开了一种具有柔性电极的中频脉冲理疗仪，包括主机、柔性电极单元、穿戴机构和肢体适应机构，柔性电极单元电连接主机并贴附于人体皮肤，肢体适应机构在脉冲理疗过程中维持柔性电极单元与皮肤的贴合状态。", "category": "个护健康"},
    {"id": "CN122399258A", "title": "一种具有抑菌功能的多物理因子复合理疗仪", "abstract": "本发明公开了一种具有抑菌功能的多物理因子复合理疗仪，包括主机、治疗头、物理因子理疗模块和抑菌处理机构，物理因子理疗模块对治疗执行面接触的人体部位施加多种物理因子理疗，抑菌处理机构对治疗执行腔进行清洁和消毒。", "category": "个护健康"},
    {"id": "CN122272343A", "title": "便于适配调节的背部经络理疗仪", "abstract": "本申请涉及一种便于适配调节的背部经络理疗仪，包括系统控制主机、理疗摄像头、座椅、理疗臂和头枕调节组件，头枕调节组件可滑动设置在系统控制主机上，能够调节理疗仪上头枕的俯仰倾斜角度和滑动距离。", "category": "个护健康"},
    {"id": "CN122537167A", "title": "一种手持理疗仪", "abstract": "本发明公开了一种手持理疗仪，涉及理疗设备技术领域，包括基体、面状的理疗组件和浮动支撑组件，多个浮动支撑组件与理疗组件连接点的阵列分布，使理疗仪的接触面更加贴合人体皮肤，提高理疗效果。", "category": "个护健康"},
    {"id": "CN224345305U", "title": "一种半导体冷激光理疗仪", "abstract": "本实用新型公开了一种半导体冷激光理疗仪，包括上盖和下盖，下盖的内侧固定连接有五金件，五金件的顶端固定连接有辅助散热机构，通过散热片增加五金件的散热效果，避免散热效果下降影响理疗仪正常工作。", "category": "个护健康"},
    {"id": "CN122183008A", "title": "一种红绿光良性刺激理疗仪", "abstract": "本发明提供一种红绿光良性刺激理疗仪，包括导杆、机壳和机架，通过滚动齿轮、机壳、灯架和语音采集器的设置，实现红绿光对用户面部特定理疗区域的集中区域式照射和灯光切换的全语音控制。", "category": "个护健康"},
    {"id": "CN122163446A", "title": "一种防烫伤艾灸理疗仪", "abstract": "本发明涉及一种防烫伤艾灸理疗仪，包括主体、灸头、安装台、固定模块、转移模块、检测模块和联动模块，通过检测模块对艾柱的燃烧情况进行感应，在艾柱燃烧殆尽后主动提醒进行艾柱更换工作。", "category": "个护健康"},
    {"id": "CN122272340A", "title": "一种上下双盘式全面气血循环美容理疗仪", "abstract": "本发明公开了一种上下双盘式全面气血循环美容理疗仪，可俯卧或仰卧使用，同时对人体面部与胸腹部进行分区同步按摩理疗，实现多重效果叠加，提升理疗效率。", "category": "个护健康"},
    # ===== 石墨烯电池（8条） =====
    {"id": "CN209461572U", "title": "一种石墨烯电池", "abstract": "本实用新型提供了一种石墨烯电池，电池体由下至上为多层复合结构，通过将石墨烯设置成多层孔状堆叠，减小接触电阻，增强导电性，电池容量提高10%-30%，充电速度是现有电池的3-4倍。", "category": "新能源"},
    {"id": "CN109390524A", "title": "一种石墨烯电池", "abstract": "本发明涉及石墨烯电池装置领域，包括固定壳体、连接机构、第一导电块、电池芯体、第二导电块、第一散热机构及第二散热机构，通过第一散热机构和第二散热机构提高电池散热性能，电池之间连接方便且稳定。", "category": "新能源"},
    {"id": "CN206558624U", "title": "一种石墨烯电池", "abstract": "本实用新型公开了一种石墨烯电池，从上而下依次为正极集流体、石墨烯导电涂层、正极活性材料涂层、隔膜、负极活性材料涂层、负极集流体，提高了石墨烯的导电性能和稳定性，延长了电池使用寿命。", "category": "新能源"},
    {"id": "CN107658497A", "title": "一种石墨烯电池板", "abstract": "本发明公开了一种石墨烯电池板，包括钴酸锂离子聚合物电芯和石墨烯导热层，石墨烯导热层的一端设置有绝缘层，通过添加缓冲层降低电池意外摔落碰撞时可能造成的损坏。", "category": "新能源"},
    {"id": "CN215732023U", "title": "一种石墨烯电池", "abstract": "本实用新型涉及电池技术领域，公开了一种石墨烯电池，包括石墨烯电池本体，电池本体的外部安装有保护盒，通过弹簧起到减震作用，减少了对电池造成损坏的可能性。", "category": "新能源"},
    {"id": "CN207409619U", "title": "一种石墨烯电池", "abstract": "本实用新型公开了一种石墨烯电池，包括安装壳，安装壳的内部通过限位座固定安装有石墨烯电池本体，安装壳的顶部粘贴有光伏发电板可自行充电，功能多样，安全稳定。", "category": "新能源"},
    {"id": "CN204481052U", "title": "一种石墨烯电池", "abstract": "本实用新型提出了一种石墨烯电池，由上而下依次包括金属上电极、氮化硅薄膜、N型石墨烯薄膜、P型基底晶体硅片和金属下电极，结构和工艺简单，生产成本低，安全可靠。", "category": "新能源"},
    {"id": "CN108232166A", "title": "一种石墨烯电池", "abstract": "本发明公开了一种石墨烯电池，由正极极片、半导体及负极极片层压后构成，正极极片呈长条状截面为椭圆形，正极极片具有至少三片且呈非等间距设置在半导体前部，负极极片贴在半导体后部。", "category": "新能源"},
    # ===== 智能穿戴（9条） =====
    {"id": "CN122261382A", "title": "一种运动姿态监测的智能穿戴系统", "abstract": "本发明涉及一种运动姿态监测的智能穿戴系统，通过穿戴设备内置的惯性测量单元采集运动员击球前后多轴加速度与角速度原始信号，通过长短期记忆网络实现动作初步分类，可精准识别十五种以上运动专项技术动作。", "category": "智能穿戴"},
    {"id": "CN122372826A", "title": "摄像模组和智能穿戴设备", "abstract": "本申请公开了一种摄像模组和智能穿戴设备，包括摄像头模块、支架组件和控制模块，控制模块能够通过控制至少部分磁性组件的极性使摄像头模块相对于支架组件具有移动自由度和旋转自由度，提高获取图像信息的灵活度。", "category": "智能穿戴"},
    {"id": "CN122067512A", "title": "多方通话方法及系统、智能穿戴设备和基站", "abstract": "本申请提出一种多方通话方法及系统，属于多方通话技术领域，智能穿戴设备用于语音采集与播放，基站进行语种识别、实时翻译与多路语音合成，实现接近自然对话的无感同声传译体验。", "category": "智能穿戴"},
    {"id": "CN122207935A", "title": "一种多模态智能穿戴装置", "abstract": "本发明涉及一种多模态智能穿戴装置，包括主体，主体上设置有穿戴空间和出音孔，能够在第一尺寸状态和第二尺寸状态之间切换，实现运动健康监测与音频交互功能的无缝融合。", "category": "智能穿戴"},
    {"id": "CN122179651A", "title": "摄像模组和智能穿戴设备", "abstract": "本申请公开了一种摄像模组和智能穿戴设备，包括摄像头模块、压电组件和控制模块，控制模块通过控制施加于各压电片上的电压值使各压电片能够带动弹性件的第一端弯曲并偏离第一方向，具有更大的视角范围。", "category": "智能穿戴"},
    {"id": "CN122030883A", "title": "一种用于估计压力反射敏感性的智能穿戴系统", "abstract": "本申请涉及可穿戴健康监测技术领域，具体涉及一种用于估计压力反射敏感性的智能穿戴系统，通过可穿戴设备同步采集用户至少两种不同类型的生理信号，实现了在可穿戴设备中对压力反射敏感性的无创、连续估计。", "category": "智能穿戴"},
    {"id": "CN122192486A", "title": "一种具有测量功能的智能穿戴装置", "abstract": "本发明涉及穿戴用具技术领域，具体为一种具有测量功能的智能穿戴装置，包括鞋底件、鞋帮、称重传感器，称重传感器集成设置于鞋底件内部，能够实时测量穿戴者的体重特征，无需到特定的体重秤处测量。", "category": "智能穿戴"},
    {"id": "CN122410928A", "title": "一种基于AI智能的协作创新智能穿戴终端", "abstract": "本发明公开了一种基于AI智能的协作创新智能穿戴终端，采用充电线供电加辅助锂电池供电的设计方式解决长续航问题，采用曲面柔性屏加贴合手臂的结构设计使显示面积比传统手表有数倍的增加优势。", "category": "智能穿戴"},
    {"id": "CN121813007A", "title": "电连接组件和智能穿戴设备", "abstract": "本申请公开了一种电连接组件和智能穿戴设备，公座包括插接槽和第一限位部，插接槽呈环形且内壁设有第一端子组，母座包括插接部和第二限位部，插接部包括电源连接部和接地连接部，电源连接部和接地连接部为同心且间隔设置的环形件。", "category": "智能穿戴"},
]

# ---------- 语义检索函数（批量调用优化版） ----------
@st.cache_data(ttl=86400, show_spinner=False)
def get_patent_embeddings():
    """缓存专利摘要向量，避免每次查询重复计算。"""
    return get_embeddings_batch([p["abstract"] for p in PATENTS])



def search_patents(query, top_k=MAX_RESULTS):
    """语义检索，并过滤低相关结果和明显不匹配的品类。"""
    if not query or not query.strip():
        return []

    with st.spinner("正在分析商品信息..."):
        patent_vectors = get_patent_embeddings()
        query_vectors = get_embeddings_batch([query])

    if patent_vectors is None or query_vectors is None or not query_vectors:
        get_patent_embeddings.clear()
        return []

    if len(patent_vectors) != len(PATENTS):
        print(f"Patent vector count mismatch: {len(patent_vectors)}/{len(PATENTS)}")
        st.error("分析数据加载异常，请刷新页面后重试。")
        return []

    query_vec = query_vectors[0]
    query_categories = infer_query_categories(query)
    norm_q = np.linalg.norm(query_vec)
    if norm_q == 0:
        return []

    results = []
    for i, p in enumerate(PATENTS):
        patent_vec = patent_vectors[i]
        norm_p = np.linalg.norm(patent_vec)
        similarity = 0.0
        if norm_p != 0:
            similarity = float(np.dot(query_vec, patent_vec) / (norm_q * norm_p))

        score = round(similarity, 3)
        if score < MIN_RELEVANCE_SCORE:
            continue
        if query_categories and p["category"] not in query_categories:
            continue

        results.append({
            "专利号": p["id"],
            "专利标题": p["title"],
            "摘要": p["abstract"],
            "品类": p["category"],
            "相似度": score,
            "相关性等级": "高相关" if score >= 0.7 else "中相关"
        })

    results.sort(key=lambda x: x["相似度"], reverse=True)
    return results[:min(top_k, MAX_RESULTS)]

# ---------- 多维度评估函数 ----------
def evaluate_dimensions(query, patent):
    """生成多维度评分"""
    text = (patent.get("专利标题", "") + " " + patent.get("摘要", "")).lower()
    
    tech_keywords = ["装置", "系统", "方法", "设备", "组件", "结构", "模块", "单元", "机构"]
    tech_match = sum(1 for kw in tech_keywords if kw in text) / len(tech_keywords)
    
    func_keywords = ["控制", "检测", "监测", "连接", "传输", "处理", "调节", "保护", "充电", "散热"]
    func_match = sum(1 for kw in func_keywords if kw in text) / len(func_keywords)
    
    query_categories = infer_query_categories(query)
    cat_match = 0.5
    if query_categories:
        cat_match = 0.9 if patent.get("品类", "") in query_categories else 0.2
    
    return {
        "整体技术匹配度": round(min(tech_match * 0.7 + 0.2, 0.95), 3),
        "功能特征一致性": round(min(func_match * 0.7 + 0.2, 0.95), 3),
        "领域适用性": round(min(cat_match + 0.1, 0.95), 3)
    }

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("数据概览")
    st.metric("专利样本", f"{len(PATENTS)} 条")
    st.metric("覆盖品类", f"{len(set(p['category'] for p in PATENTS))} 个")
    st.caption("分析方式：语义相关性匹配")
    st.caption(f"结果范围：优先展示同品类高相关专利，最多 {MAX_RESULTS} 条")

    if "history" in st.session_state and st.session_state.history:
        st.divider()
        st.subheader("最近查询")
        for h in st.session_state.history[-5:]:
            st.write(f"- {h}")

# ---------- 初始化 ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- 主界面 ----------
with st.container(border=True):
    st.subheader("商品信息")
    query = st.text_input(
        "商品名称或描述",
        placeholder="例如：带茶滤的保温杯",
        max_chars=MAX_QUERY_CHARS,
        help=f"建议填写商品品类、材质、结构或核心用途，最多 {MAX_QUERY_CHARS} 字。"
    ).strip()

    option_col, action_col = st.columns([3, 2])
    with option_col:
        show_aux_metrics = st.checkbox(
            "显示匹配详情",
            value=False,
            help="查看技术要素、功能要素和领域匹配等辅助信息。"
        )
    with action_col:
        analyze_clicked = st.button(
            "开始分析",
            type="primary",
            use_container_width=True
        )

if analyze_clicked:
    if not query or len(query.strip()) == 0:
        st.warning("请输入商品描述")
    else:
        now = time.time()
        elapsed = now - st.session_state.get("last_query_time", 0)
        if elapsed < QUERY_COOLDOWN_SECONDS:
            wait_seconds = max(1, int(QUERY_COOLDOWN_SECONDS - elapsed + 0.999))
            st.warning(f"查询过于频繁，请等待 {wait_seconds} 秒后重试。")
            st.stop()
        st.session_state.last_query_time = now

        if query not in st.session_state.history:
            st.session_state.history.append(query)

        status_container = st.container()
        with status_container:
            st.subheader("分析进度")
            status_placeholder = st.empty()
            
            status_placeholder.info("正在识别商品特征...")
            time.sleep(0.2)

            status_placeholder.info("正在匹配相关专利...")
            time.sleep(0.2)

            results = search_patents(query)

            status_placeholder.info("正在整理分析结果...")
            time.sleep(0.2)
            status_placeholder.success("分析完成")

        if results:
            st.subheader("筛查结果")
            query_categories = infer_query_categories(query)
            top_similarity = max(r["相似度"] for r in results)
            matched_categories = sorted({r["品类"] for r in results})

            summary1, summary2, summary3 = st.columns(3)
            summary1.metric("相关专利", f"{len(results)} 条")
            summary2.metric("最高匹配度", f"{top_similarity * 100:.1f}%")
            summary3.metric(
                "涉及品类",
                matched_categories[0] if len(matched_categories) == 1 else f"{len(matched_categories)} 个品类"
            )

            if query_categories:
                st.caption("匹配范围：" + "、".join(sorted(query_categories)))
            if len(results) < MAX_RESULTS:
                st.info(
                    f"本次分析获得 {len(results)} 条匹配度较高的专利。"
                    "补充商品材质、结构或用途可获得更全面的分析结果。"
                )

            st.caption("分析结果基于商品描述与专利摘要生成，仅供风险初筛和人工复核参考，不构成法律意见。")

            df = pd.DataFrame(results)
            df.insert(0, "查询文本", query)
            csv = df.to_csv(index=False, encoding="utf-8-sig")

            export_note, export_action = st.columns([3, 1.4])
            with export_note:
                st.caption("可将本次分析结果导出为 CSV，便于后续复核和归档。")
            with export_action:
                st.download_button(
                    label="导出筛查结果",
                    data=csv,
                    file_name=f"商品专利筛查结果_{query}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.divider()

            for i, r in enumerate(results):
                dimensions = evaluate_dimensions(query, r)

                with st.expander(
                    f"{i + 1}. {r['专利标题']}｜{r['相关性等级']}"
                ):
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        st.caption(f"专利号 {r['专利号']} · {r['品类']}")
                        st.markdown(f"**摘要**\n\n{r['摘要']}")
                        st.link_button(
                            "查看专利原文",
                            patent_detail_url(r["专利号"]),
                            help="将打开国家知识产权局专利检索平台。复制上方专利号后即可查询。"
                        )

                    with col2:
                        st.markdown("**匹配表现**")
                        st.metric("文本匹配度", f"{r['相似度'] * 100:.1f}%")
                        st.metric("相关性判断", r["相关性等级"])
                        if show_aux_metrics:
                            st.markdown("**更多匹配信息**")
                            st.progress(
                                dimensions["整体技术匹配度"],
                                text=f"技术要素覆盖：{dimensions['整体技术匹配度'] * 100:.0f}%"
                            )
                            st.progress(
                                dimensions["功能特征一致性"],
                                text=f"功能要素覆盖：{dimensions['功能特征一致性'] * 100:.0f}%"
                            )
                            st.progress(
                                dimensions["领域适用性"],
                                text=f"领域匹配度：{dimensions['领域适用性'] * 100:.0f}%"
                            )
        else:
            st.info("暂未发现匹配度较高的同品类专利。建议补充商品材质、结构或核心用途后再次分析。")
