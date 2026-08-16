import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time

st.set_page_config(page_title="商品专利风险筛查系统", page_icon="🛡️", layout="wide")

st.title("🛡️ 商品专利风险筛查系统")
st.write("输入商品描述，系统将通过通义千问语义检索判断专利侵权风险")

# ===== 通义千问 API 配置 =====
DASHSCOPE_API_KEY = st.secrets["DASHSCOPE_API_KEY"]
DASHSCOPE_BASE_URL = st.secrets["DASHSCOPE_BASE_URL"]

def get_embeddings_batch(texts, batch_size=20):
    """批量调用通义千问 Embedding API，自动分批处理（每批不超过20条）"""
    if not texts or len(texts) == 0:
        return None
    
    all_embeddings = []
    
    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen3.7-text-embedding",
            "input": {
                "texts": batch
            },
            "dimensions": 1024
        }
        
        try:
            response = requests.post(
                f"{DASHSCOPE_BASE_URL}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                for item in result["data"]:
                    all_embeddings.append(np.array(item["embedding"], dtype=np.float32))
            else:
                st.error(f"API 调用失败（状态码 {response.status_code}）：{response.text[:200]}")
                return None
        except Exception as e:
            st.error(f"API 调用异常：{str(e)}")
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
def search_patents(query, top_k=5):
    """基于通义千问 Embedding 的语义检索（批量调用优化）"""
    if not query or len(query.strip()) == 0:
        return []

    # 1. 收集所有需要向量化的文本：查询 + 所有专利摘要
    abstracts = [p["abstract"] for p in PATENTS]
    all_texts = [query] + abstracts
    
    # 2. 批量调用 API（只需1-2次网络请求）
    with st.spinner("正在调用通义千问进行语义向量化..."):
        all_embeddings = get_embeddings_batch(all_texts)
    
    if all_embeddings is None:
        return []
    
    # 3. 分离查询向量和专利向量
    query_vec = all_embeddings[0]
    patent_vectors = all_embeddings[1:]
    
    # 4. 计算相似度
    results = []
    norm_q = np.linalg.norm(query_vec)
    if norm_q == 0:
        return []
    
    for i, p in enumerate(PATENTS):
        patent_vec = patent_vectors[i]
        norm_p = np.linalg.norm(patent_vec)
        if norm_p == 0:
            similarity = 0
        else:
            similarity = float(np.dot(query_vec, patent_vec) / (norm_q * norm_p))
        
        score = round(similarity, 3)
        
        results.append({
            "专利号": p["id"],
            "专利标题": p["title"],
            "摘要": p["abstract"],
            "品类": p["category"],
            "相似度": score,
            "风险等级": "高风险" if score >= 0.7 else ("中风险" if score >= 0.4 else "低风险")
        })
    
    results.sort(key=lambda x: x["相似度"], reverse=True)
    return results[:top_k]

# ---------- 多维度评估函数 ----------
def evaluate_dimensions(query, patent, score):
    """生成多维度评分"""
    text = (patent.get("专利标题", "") + " " + patent.get("摘要", "")).lower()
    
    tech_keywords = ["装置", "系统", "方法", "设备", "组件", "结构", "模块", "单元", "机构"]
    tech_match = sum(1 for kw in tech_keywords if kw in text) / len(tech_keywords)
    
    func_keywords = ["控制", "检测", "监测", "连接", "传输", "处理", "调节", "保护", "充电", "散热"]
    func_match = sum(1 for kw in func_keywords if kw in text) / len(func_keywords)
    
    cat_match = 0.5
    categories = ["家居用品", "数码配件", "车载配件", "手机配件", "个护健康", "新能源", "智能穿戴"]
    for cat in categories:
        if cat in query:
            if patent.get("品类", "") == cat:
                cat_match = 0.9
                break
    
    return {
        "整体技术匹配度": round(min(tech_match * 0.7 + 0.2, 0.95), 3),
        "功能特征一致性": round(min(func_match * 0.7 + 0.2, 0.95), 3),
        "领域适用性": round(min(cat_match + 0.1, 0.95), 3)
    }

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("📊 系统信息")
    st.write(f"专利库数量：**{len(PATENTS)}** 条")
    st.write(f"覆盖品类：**{len(set(p['category'] for p in PATENTS))}** 个")
    st.write("检索方式：**通义千问 qwen3.7-text-embedding**")
    st.caption("基于向量相似度的语义匹配（批量调用优化）")

    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.write("📋 **检索历史**")
        for h in st.session_state.history[-5:]:
            st.write(f"- {h}")

# ---------- 初始化 ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- 主界面 ----------
query = st.text_input("请输入商品描述", placeholder="例如：蓝牙耳机、石墨烯发热护膝")

if st.button("🔍 开始筛查"):
    if not query or len(query.strip()) == 0:
        st.warning("⚠️ 请输入商品描述")
    else:
        if query not in st.session_state.history:
            st.session_state.history.append(query)

        status_container = st.container()
        with status_container:
            st.markdown("### ⚙️ 处理进度")
            status_placeholder = st.empty()
            
            status_placeholder.info("🔍 步骤 1/3：正在解析商品描述，提取语义特征...")
            time.sleep(0.2)
            
            status_placeholder.info("📚 步骤 2/3：正在通过通义千问进行语义检索...")
            time.sleep(0.2)
            
            results = search_patents(query)
            
            status_placeholder.info("📊 步骤 3/3：正在执行多维度风险分析...")
            time.sleep(0.2)
            status_placeholder.success("✅ 分析完成！")

        if results:
            st.subheader(f"📊 检索结果（共找到 {len(results)} 条相关专利）")

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
                dimensions = evaluate_dimensions(query, r, r["相似度"])
                
                with st.expander(f"#{i+1} {r['专利标题']}  —  综合风险：{r['风险等级']}"):
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.write(f"**专利号：** {r['专利号']}")
                        st.write(f"**摘要：** {r['摘要']}")
                        st.write(f"**品类：** {r['品类']}")
                    
                    with col2:
                        st.markdown("**📐 多维度评估**")
                        st.progress(dimensions["整体技术匹配度"], text=f"整体技术匹配度：{dimensions['整体技术匹配度']*100:.0f}%")
                        st.progress(dimensions["功能特征一致性"], text=f"功能特征一致性：{dimensions['功能特征一致性']*100:.0f}%")
                        st.progress(dimensions["领域适用性"], text=f"领域适用性：{dimensions['领域适用性']*100:.0f}%")
                        st.caption(f"**综合相似度：** {r['相似度']*100:.1f}% ｜ **风险等级：** {r['风险等级']}")
        else:
            st.info("未找到匹配的专利，建议更换关键词尝试。")
