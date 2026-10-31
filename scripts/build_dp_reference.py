"""Build the Chinese DP reference PDF. Requires reportlab and a CJK TTF/TTC.

Run from any directory; output and supporting paths resolve from this repository.
Use --font /path/to/font.ttf on Linux when no supported CJK font is discovered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Flowable, Paragraph, Preformatted, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "DiscoverPhysics_研究笔记.pdf"
WIDTH, HEIGHT = A4
MARGIN = 46
CW = WIDTH - 2 * MARGIN
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#007E83")
INK = colors.HexColor("#243648")
MUTED = colors.HexColor("#607284")
GRID = colors.HexColor("#D8E2E9")
PALE = colors.HexColor("#F3F7FA")
AMBER = colors.HexColor("#FFF5E4")
DOC_REVISION = "DP-R9"
STYLES: dict[str, ParagraphStyle] = {}


def register_fonts(font: str | None, bold: str | None) -> None:
    options = [font] if font else [
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
    ]
    selected = None
    errors = []
    for candidate in options:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            pdfmetrics.registerFont(TTFont("CJK", candidate))
            selected = candidate
            break
        except Exception as exc:
            errors.append(str(exc))
    if selected is None:
        raise RuntimeError("No compatible CJK TTF/TTC. Pass --font. " + "; ".join(errors))
    bold_candidate = bold or ("C:/Windows/Fonts/msyhbd.ttc" if selected.startswith("C:/Windows") else selected)
    pdfmetrics.registerFont(TTFont("CJK-Bold", bold_candidate))
    pdfmetrics.registerFontFamily("CJK", normal="CJK", bold="CJK-Bold", italic="CJK", boldItalic="CJK-Bold")
    common = dict(fontName="CJK", textColor=INK, wordWrap="CJK", splitLongWords=True)
    STYLES.update({
        "body": ParagraphStyle("body", fontSize=11.0, leading=18.5, spaceAfter=7, rightIndent=11, **common),
        "small": ParagraphStyle("small", fontSize=9.4, leading=15.2, spaceAfter=6, rightIndent=9.4, **common),
        "h2": ParagraphStyle("h2", fontName="CJK-Bold", fontSize=12.3, leading=19.5, textColor=NAVY, spaceBefore=9, spaceAfter=6, wordWrap="CJK"),
        "cell": ParagraphStyle("cell", fontSize=10.0, leading=15.8, rightIndent=4, **common),
        "head": ParagraphStyle("head", fontName="CJK-Bold", fontSize=10.0, leading=16, textColor=colors.white, wordWrap="CJK"),
        "note": ParagraphStyle("note", fontSize=10.0, leading=16.5, rightIndent=9, **common),
        "formula": ParagraphStyle("formula", fontName="CJK", fontSize=12.5, leading=21, textColor=NAVY, alignment=TA_CENTER, spaceBefore=5, spaceAfter=9, wordWrap="CJK"),
        "title": ParagraphStyle("title", fontName="CJK-Bold", fontSize=20.0, leading=29, textColor=NAVY, wordWrap="CJK", spaceAfter=9),
        "caption": ParagraphStyle("caption", fontSize=8.5, leading=13, rightIndent=8.5, textColor=MUTED, fontName="CJK", wordWrap="CJK", spaceAfter=5),
    })


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def h(text: str) -> Paragraph:
    return p(text, "h2")


def note(title: str, text: str, warning: bool = False) -> Table:
    result = Table([[p(f"<b>{title}</b><br/>{text}", "note")]], colWidths=[CW])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER if warning else colors.HexColor("#EAF5F5")),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#CE8A28") if warning else TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    result.spaceAfter = 9
    return result


def table(headers: list[str], rows: list[list[str]], ratios: list[float], small: bool = False) -> Table:
    data = [[p(x, "head") for x in headers]]
    data += [[p(str(x), "cell" if not small else "small") for x in row] for row in rows]
    widths = [CW * value / sum(ratios) for value in ratios]
    result = Table(data, colWidths=widths, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, NAVY),
        ("LINEBELOW", (0, 1), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    result.spaceAfter = 9
    return result


def formula(text: str) -> Paragraph:
    return p(text, "formula")


def code(text: str) -> Table:
    style = ParagraphStyle("code", fontName="Courier", fontSize=8.6, leading=12)
    block = Preformatted(text, style)
    result = Table([[block]], colWidths=[CW])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 11), ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    result.spaceAfter = 9
    return result


class LoopDiagram(Flowable):
    def __init__(self, labels: list[str]):
        super().__init__()
        self.width = CW
        self.height = 83
        self.labels = labels

    def draw(self):
        c = self.canv
        gap = 17
        bw = (CW - gap * (len(self.labels) - 1)) / len(self.labels)
        for index, label in enumerate(self.labels):
            x = index * (bw + gap)
            c.setFillColor(PALE)
            c.setStrokeColor(GRID)
            c.roundRect(x, 24, bw, 49, 5, fill=1, stroke=1)
            body = p(label, "note")
            _, bh = body.wrap(bw - 13, 46)
            body.drawOn(c, x + 7, 24 + (49 - bh) / 2)
            if index < len(self.labels) - 1:
                c.setStrokeColor(TEAL)
                c.line(x + bw + 2, 48, x + bw + gap - 3, 48)
                c.line(x + bw + gap - 7, 51, x + bw + gap - 3, 48)
                c.line(x + bw + gap - 7, 45, x + bw + gap - 3, 48)
        c.setFont("CJK", 8)
        c.setFillColor(MUTED)
        c.drawCentredString(CW / 2, 7, "真实实验提供数据；候选拟合复用数据；二者不改写真实定律")


def build_pages(date: str) -> list[dict]:
    pages = []

    def add(title: str, items: list, sources: str, subtitle: str = ""):
        pages.append(dict(title=title, items=items, sources=sources, subtitle=subtitle))

    add("DiscoverPhysics 研究笔记", [
        p("公开11宇宙 · 实验与观测 · 关键物理概念 · 草稿对照", "h2"),
        p(f"更新日期：{date}　｜　文档版本：{DOC_REVISION}　｜　用途：命题研究与日常查阅", "small"),
        note("阅读口径", "DP指论文；源码结论限定本项目导入的公开版本。本文将<b>论文设置、源码事实、设计估计、草稿现状</b>分别表述，不把估计写成实测。world与宇宙在本文中同义。"),
        p("本册限附录C.1至C.11的11个公开宇宙。22世界统计仅用于说明原始评价方法；额外注册的wave与不在这11项内的species不扩入本册。"),
        p("草稿为六项保真迁移与三项自设计；具体对应和机制私有。已有局部参考原型检查，完整噪声、LLM与端到端难度仍待试做。", "small"),
        h("查阅导航（可点击）"),
        table(["页", "主题", "页", "主题"], [
            ["02", '<link href="#p2">DP完整发现流程</link>', "10", '<link href="#p10">拟合、积分与主动实验</link>'],
            ["03", '<link href="#p3">粒子、探针与隐藏信息</link>', "11", '<link href="#p11">误差、解释与通过判据</link>'],
            ["04", '<link href="#p4">源强s、响应c、惯性m</link>', "12", '<link href="#p12">难度估计与论文排序</link>'],
            ["05", '<link href="#p5">六个单移动粒子宇宙</link>', "13", '<link href="#p13">实验规模与配额估计</link>'],
            ["06", '<link href="#p6">五个多粒子宇宙</link>', "14", '<link href="#p14">DP与现行草稿流程对照</link>'],
            ["07", '<link href="#p7">控制、返回和计分粒子数</link>', "15", '<link href="#p15">属性、声明与规则快照</link>'],
            ["08", '<link href="#p8">时间、轮次、噪声与种子</link>', "16", '<link href="#p16">自动交互及保密边界</link>'],
            ["09", '<link href="#p9">默认测试初态与空间外推</link>', "17", '<link href="#p17">来源索引与核查提醒</link>'],
        ], [0.07, 0.43, 0.07, 0.43]),
        p("源码基线：33b7fa9df96de9c35744efd181ca7e5a8dd60ad5（Add citation）。本文是截至更新日的整理，不代表相关缺陷已经修复，也不代表竞赛控制台已实现。", "caption"),
    ], "[P1] 论文；[C1-C7] 固定源码；[D1] 现行草稿及项目记忆。")

    add("DP完整发现流程", [
        LoopDiagram(["提出假设<br/>设计实验", "真实模拟器<br/>执行实验", "返回观测<br/>位置 / 速度", "分析与拟合<br/>修正规律"]),
        table(["阶段", "实际发生的操作", "信息边界"], [
            ["进入任务", "参考agent获得mission、拓扑说明、请求格式和候选函数接口。", "知道如何操作，不等于知道真实定律。"],
            ["提出实验", "输出run_experiment中的JSON数组；一轮可以包含多组实验。", "控制量因宇宙而异。"],
            ["数值模拟", "执行器按真实力核、背景配置和初态积分，在指定时刻记录状态。", "agent收到筛选后的观测，不是源代码或真实力值。"],
            ["分析与拟合", "分析运动、比较假设；可用已有数据请求候选定律拟合。", "拟合不是新实验，也不是用隐藏测试拟合。"],
            ["继续探索", "根据反馈调整位置、属性、观测时刻、时间跨度等。", "轮数限制作用于外层agent循环。"],
            ["结束发现", "提交可执行轨迹预测函数与自然语言解释。", "随后独立进行留出轨迹测试与解释评价。"],
        ], [0.16, 0.46, 0.38]),
        h("开局已经告诉模型什么？"),
        p("拓扑协议事先告知可控粒子数量、输入输出格式、哪些是固定背景。无质量五探针协议还说明探针只受场、不产生场。模型主要发现的是作用规律、参数角色与隐藏结构，而不是自行猜测接口要填1个还是5个粒子。"),
        note("第一束光与多智能体", "当前默认入口没有草稿式成套免费数值轨迹；论文示意图不能直接当作现有注入实现。参考主流程以一个发现agent为中心，可选Critic默认关闭；解释Judge属于评价端。DP未要求参赛式多智能体协作。"),
        p("DP主要是training-free基准。参数拟合改变候选物理程序的数值，不改变LLM权重；训练、提示、工具与实验设计是不同层面的优化。"),
        p("DP参考实现提交定律Python源码与自然语言explanation，解释由LLM按逐世界rubric评分；没有本赛统一三字段或旧12字段claims。实验JSON不是科学声明协议，LLM解释评分也不是本赛曾讨论、现已取消的自洽项。", "small"),
    ], "[P1] 发现流程；[C1-C2] 协议与agent；[C4、C6] 解释评价与逐世界rubric。")

    add("先分清粒子、探针与隐藏信息", [
        table(["概念", "含义", "容易混淆的地方"], [
            ["背景粒子", "宇宙预设的天体，通常不能由agent任意改初态或属性。", "背景可以运动；固定配置不等于位置永远不动。"],
            ["可控粒子", "agent能指定某些初态或属性的粒子。", "不一定是零源探针；circle粒子会相互作用。"],
            ["被动探针", "连续受力、自由运动，但不扰动背景或其他探针。", "pos/vel是投放初态；被动不等于固定测点。"],
            ["隐藏源", "真实模拟器内部有、正常返回中看不见的源。", "仍可通过可见粒子运动推断其存在。"],
            ["实验观测", "该局已经获得的位置、速度和协议允许的辅助数据。", "并非自动包含力、加速度或隐藏状态。"],
            ["留出测试", "未作为本局探索/拟合数据提供的测试初态与轨迹。", "不是11个未公开世界的同义词。"],
            ["真值", "可以指真实动力学机制，也可以指参考模拟器轨迹。", "轨迹真值无噪声，仍受数值精度约束。"],
        ], [0.19, 0.40, 0.41]),
        h("暗物质宇宙的三个数量"),
        formula("内部35 = 可见背景20 + 隐藏源10 + 探针5"),
        p("正常run返回可见背景与探针共25个粒子的状态；最终误差只统计5个探针。内部run_full可以含完整35粒子和暗源初态，供内部画图/评测使用，不能当作agent额外观测工具。"),
        note("三层权限必须分别数", "可控制多少粒子、接口返回多少粒子、最终计分多少粒子，是三个问题。three_species的提示承诺与当前实际计分对象存在不一致，见第7页。", True),
        h("场的时间阶数与粒子运动阶数"),
        p("temporal_order=0表示每时刻的场按瞬时关系求解，不代表粒子的一阶运动。公开11宇宙当前默认均使用瞬时作用与二阶粒子运动；G(t)显式变化也不自动等于波传播。"),
        p("当前时间声明看固定几何/状态下的显式t依赖：非恒定且按T重复才属periodic，恒定归none；轨道周期和动态背景不直接等于定律显式时变。旧field_type字段已退役。", "small"),
    ], "[P1] 附录C；[C1] probes_instructions.md；[C3] NBodyDarkMatterExecutor；[C4] evaluator.py。")

    add("源强s、响应c与惯性m", [
        p("s决定粒子对别人产生多强的场；c决定粒子对外部场响应多强；m决定同样的力造成多大加速度。三者可以独立，也可以按世界规则联动。"),
        formula("F(i←j) = force_law(r, c_i, s_j, m_i, m_j)；a_i = Σ_j F(i←j) / m_i"),
        p("对这里常见的线性核，可简写为a_i约等于(c_i/m_i)×Σ_j s_j K_ij，再加额外背景加速度。该示意式不替代每个世界的完整方程。"),
        table(["宇宙", "agent能改的属性", "固定或联动关系"], [
            ["gravity / yukawa / fractional / oscillator / extra_dimensions", "p1改变中心源强；p2改变移动粒子惯性。", "移动粒子c=1；中心c=0。移动粒子不全是零源。"],
            ["coulomb_easy", "p1、p2改变电荷幅值。", "中心s=|p1|；移动粒子s=c=-|p2|，m=1；符号固定相反。"],
            ["circle", "环半径、共同切向速度。", "所有粒子s=c=m=1。"],
            ["three_species / dark_matter", "五探针位置、速度。", "probe的s=0、c=1、m=1；背景源强固定未知。"],
            ["ether / hubble", "五探针初态与probe_masses。", "probe的s=0、c=m，质量与响应联动。"],
        ], [0.34, 0.31, 0.35]),
        note("零源与零响应", "s=0、c非零是被动探针；c=0只是不响应成对场。nbody_sampler.py附近有一句注释写反，本文依据实际矩阵索引，不沿用该句。", True),
        p("轨迹常直接约束c_i s_j/m_i组合，未必能唯一分离裸源强/响应/惯性。草稿仍按mission公开目录控制属性，相关推断写候选规律；旧parameter_scaling等不再提交。"),
        p("旧12字段的径向指数、周期值、属性标度已退役。自洽项也已取消，下列误判讨论只保留研究性质，不是当前评分规则。完整核、参数和源几何仍由PARAMS、BODY_PROPS、HIDDEN及候选代码表达，主要通过预测检验。", "small"),
        p("HIDDEN空/非空只是表示，内联源项或等价场也可表达暗源，不能据此证明必要性。合法代码可能用t重建背景几何，冻结bodies再换t就可能误判。须先约定传入状态的使用语义，再讨论诊断。", "small"),
    ], "[C3] executor.py各拓扑；[C5] nbody_sampler.py:139-149。")

    add("六个单移动粒子宇宙", [
        p("共同结构：一个固定中心源与一个移动粒子。agent控制位置、初速度、p1/p2与观测安排；研究复杂规律并不意味着需要不断观测静止中心。"),
        table(["宇宙", "当前真实作用的核心", "需要辨认什么"], [
            ["gravity", "二维Poisson / 对数势；吸引力大小约随1/r衰减。", "不要直接套三维逆平方；还要区分源强与惯性。"],
            ["coulomb_easy", "逆平方中心吸引；电荷符号强制相反。", "p2改变电荷幅值而非惯性，加速度随|p1 p2|变化。"],
            ["fractional", "二维Riesz核：F随r^{-(3-2α)}变化；当前注册α=0.5，因此为r^-2。", "可先拟合幂指数；算子解释与有效力核不是同一层任务。"],
            ["yukawa", "二维屏蔽核，径向力含K1(r/λ)；默认λ=2。", "需要覆盖屏蔽尺度两侧；不能简单替换为三维exp(-r/λ)/r²。"],
            ["oscillator", "空间为二维1/r。<br/>G(t)=G0 cos(ωt+φ)。<br/>默认G0=5、T=4、φ=0。", "时间变号、周期、相位；它不是普通弹簧振子。"],
            ["extra_dimensions", "Kaluza-Klein像源求和；Rc=0.5。<br/>远处约1/r，近处趋向1/r²。", "必须进入短距离，发现尺度过渡，避免只在普通引力区采样。"],
        ], [0.22, 0.41, 0.37]),
        note("fractional版本不一致", "当前配置α=0.5，但展示公式仍出现0.75；部分解释/rubric还误称比1/r衰减更慢。当前r^-2实际更快。研究与命题应以力核和实际配置为准，不能复制互相冲突的答案。", True),
        p("本项目现保真迁移六项参考规律，可观测运动均为二维。extra_dimensions的额外维度进入有效力核，不是增加可直接投放、返回或评分的坐标轴。", "small"),
    ], "[P1] 附录C.1-C.3、C.9-C.11；[C3] executor.py；[C5] force_laws.py；[C6] worlds.py。")

    add("五个多粒子宇宙", [
        table(["宇宙", "结构与真实机制", "实验中的关键任务"], [
            ["circle", "1中心+10环粒子，全部产生并响应场；α=0.75的分数阶核，力约r^-1.5。", "改变共同环半径和切向速度，识别多体作用；对称性提供冗余信息。"],
            ["three_species", "30背景分为3类，源强1、3、-2；另有5个零源探针。背景会随时间运动。", "区分正负、相对源强和隐藏类别；不要把固定背景初态当成静态天体。"],
            ["dark_matter", "20可见背景、10隐藏暗源、5探针；可见源强1、暗源5，隐藏源初始构成集中分布。", "从可见运动推断额外源；源分布、强度与力律可能退化。"],
            ["ether", "锚点1、轨道天体20、探针5。中心场仅由锚点产生，另有统一定向外加速度。", "识别共同背景加速度并与中心场分开；许多可见轨迹提供交叉验证。"],
            ["hubble", "同样26粒子；中心吸引叠加H×相对中心位置的外推加速度。", "比较不同半径，分离向心吸引和随距离增长的外推。"],
        ], [0.19, 0.47, 0.34]),
        h("被动性与质量变化"),
        p("three_species、dark_matter、ether、hubble的五探针均不产生场。ether/hubble还令响应c随惯性m一起增大，c/m=1；外加速度也不依赖探针质量，所以同初态下改变probe_masses不会改变无噪声运动。"),
        p("ether的统一加速度也作用于锚点，不能把它永久钉在原点；返回坐标相对固定初始域中心，不能改成跟随锚点。hubble参考固定中心，初在中心的锚点保持静止。两者执行器的中心源强均为50，部分旧展示文字的40不能照搬。", "small"),
        note("不是同一个探针拓扑", "circle是相互作用体系，不能按草稿的理想被动探针规则直接搬运。背景运动复杂度、可见性与可控属性决定实验难度，粒子总数只是其中一个因素。"),
    ], "[P1] 附录C.4-C.8；[C3] Circle/ThreeSpecies/DarkMatter/Ether/Hubble执行器。")

    add("控制、返回、计分：三份数量表", [
        p("以下为当前默认N-body实现与默认测试配置。case数不是每次同时投放的探针数，也不是整局实验次数；调用方可以改测试配置。"),
        table(["宇宙", "内部总数", "每实验可控", "正常返回", "实际计分", "测试case×时刻"], [
            ["gravity", "2", "1移动", "2", "1移动", "3×10"],
            ["yukawa", "2", "1移动", "2", "1移动", "3×10"],
            ["fractional", "2", "1移动", "2", "1移动", "3×10"],
            ["circle", "11", "10环粒子", "11", "全部11", "1×5"],
            ["three_species", "35", "5探针", "35", "全部35", "2×5"],
            ["dark_matter", "35", "5探针", "25", "仅5探针", "2×10"],
            ["ether", "26", "5探针", "26", "仅5探针", "2×10"],
            ["hubble", "26", "5探针", "26", "仅5探针", "2×10"],
            ["oscillator", "2", "1移动", "2", "1移动", "3×10"],
            ["extra_dimensions", "2", "1移动", "2", "1移动", "3×10"],
            ["coulomb_easy", "2", "1移动", "2", "1移动", "3×10"],
        ], [0.22, 0.10, 0.17, 0.13, 0.17, 0.21]),
        note("three_species的提示/实现矛盾", "共享五探针提示说只评最后5个探针，实际evaluator对35个粒子累计位置误差。本文列实际代码；不能把提示矛盾解释成需要模型自己发现的物理规律。", True),
        h("模型如何知道这些结构？"),
        p("get_world选择拓扑说明，入口把说明注入系统提示；第一次模型调用前，粒子控制数量和格式已经告知。隐藏物种、源强与暗源结构仍可能未知。无质量五探针协议明说其零源；带质量协议没有同样明确宣称零源，虽然代码里确实如此。"),
        p("草稿同样区分任务目标与操作协议：mission说明要发现、提交什么；拓扑协议公开可见背景数量/ID、可控对象、每批最多8探针、属性与输入输出、探针被动性。知道如何操作，不等于知道真实力律或隐藏源数量。", "small"),
    ], "[C1] 各instructions；[C2] prompt注入；[C3] 返回数据；[C4] 默认case与计分索引；[D1] 草稿接口。")

    add("时间、轮次、噪声与种子", [
        table(["计数层次", "实际含义", "不能等同于"], [
            ["完整session", "从任务进入到最终规律提交的一次运行。", "一个实验、一个噪声读数。"],
            ["LLM轮次", "外层agent循环的一次模型响应与工具处理。", "一个实验；一轮可含多组。"],
            ["实验组", "一组初态/属性/时间安排，执行器一次模拟。", "一个粒子或一个时刻。"],
            ["测量时刻", "同次实验返回轨迹的采样位置。", "积分步或再次调用LLM。"],
            ["积分步", "数值求解内部推进。", "真实经过一秒或一个读数。"],
        ], [0.19, 0.45, 0.36]),
        p("论文常用最多16轮，代码默认10轮。最后一轮要求提交。fit-only也占一个外层轮次；同轮实验加拟合不额外加轮。DP没有整局60分钟上限；拟合与定律执行另有局部超时。"),
        h("绝对时间与重放"),
        p("多数DP实验重建模拟器，从固定背景初态与相对时间0开始。oscillator支持start_time，绝对时刻为start_time+measurement_times；它与草稿t_launch+readout可对应。"),
        p("草稿现已明确整局统一物理t=0；各实验按t_launch查询或重放同一背景B(t)，不逐次重置宇宙物理相位。readout是投放后偏移，绝对采样时刻为t_launch+readout。在mission允许范围内，可以先查较晚时刻，再重放较早时刻。"),
        p("故事60分钟是mission首次成功后计现实elapsed，不是48小时开发期或t_launch；绝对t仅指宇宙统一t0坐标。节约只按能量、不计时间，1h运行及宽松物理T/D护栏仍保留。重测独立抽噪声，archive/第一束光缓存不重抽。", "small"),
        h("主实验的位置噪声"),
        formula("观测位置 = r* + ε；ε ~ N(0, σ² I)；σ = 0.05 √V"),
        p("5%指标准差相对轨迹尺度，不是方差的5%。论文主实验只加位置噪声，速度保持干净；两路加噪声属于另行消融/代码选项。评分参考轨迹不加观测噪声。当前脚本使用预置世界方差常数，具体归一化轴不能凭空补定。"),
        p("论文五次尝试是完整session重复。benchmark seed控制观测噪声，随机模式还控制实验抽样；当前LLM客户端未统一传采样seed，不能声称它固定全部LLM随机性。seed也不重抽世界定律或默认测试初态。", "small"),
    ], "[P1] 第6、8页；[C2] agent.py；[C3] start_time；[C7] noise_frac、WORLD_VARS与seed；[D1] 绝对时间规则。")

    add("默认空间测试与外推边界", [
        p("以下是公开代码的默认位置设置，非论文全基准统一外推协议。Typical只是提示中的典型范围，不能视为本题式服务器硬投放边界。"),
        table(["宇宙", "典型探索范围", "默认测试初始位置/半径"], [
            ["gravity", "坐标[-10,10]²", "移动粒子(3,0)、(5,0)、(-4,2)，r约3-5。"],
            ["yukawa", "坐标[-10,10]²", "同上三个初态。"],
            ["fractional", "坐标[-10,10]²", "同上三个初态。"],
            ["circle", "环半径2-10", "共同环半径5。"],
            ["three_species", "坐标[-15,15]²", "探针初始半径0-约9.90。"],
            ["dark_matter", "坐标[-15,15]²", "探针初始半径约10.30-15。"],
            ["ether", "坐标[-22,22]²", "探针初始半径10-18。"],
            ["hubble", "坐标[-22,22]²", "探针初始半径6-18。"],
            ["oscillator", "坐标[-10,10]²", "同三个两粒子初态。"],
            ["extra_dimensions", "坐标[-10,10]²", "同三个两粒子初态。"],
            ["coulomb_easy", "坐标[-10,10]²", "同三个两粒子初态。"],
        ], [0.23, 0.28, 0.49]),
        note("没有统一的区外两倍保证", "这些默认测试初态均在各自典型范围内。测试未作为本局观测提供，不等于位置必在区外。轨迹后来飞出某区域，也不等于初态被设计在区外。"),
        p("草稿另外要求投放区以外两倍距离、更长时间和未用过的公开属性组合。属性是否可变及取值由各宇宙mission目录规定。其泛化要求与DP不同；Hubble约12.6的力平衡尺度是物理尺度，不是投放区边界。"),
        p("投放区只限初始pos，探针后续可离开。mission合法连续域内可自由选位置，第一束光两标准点不限制后续投放。三项自设计的2-8环域只是试做，六项迁移范围仍待校准，不能说九宇宙探测域均已充分验证。跨尺度可达性与采样需试做；不新增探针类别。", "small"),
    ], "[C1] Typical范围；[C4] 默认test_cases；项目静态审计public-worlds-audit.json；[D1] 草稿验证要求。")

    add("拟合、积分与主动实验", [
        h("数值拟合不是训练LLM"),
        p("run_mse_fit把候选程序与已观测数据对照，返回拟合前后误差和参数。它不提供真实力律、不修改模拟器，也不使用留出轨迹。该工具可配置关闭，不能说所有运行都具有相同拟合支持。"),
        p("当前最终可选拟合声明至多5个标量，最多抽4条已获轨迹、每条5个时刻；这不是整局只能观测4条。中途拟合不受同款抽样上限约束，而且不是所有拓扑的最终evaluator都实现自动拟合。"),
        table(["职责", "DP", "草稿"], [
            ["真实实验", "真实执行器积分，返回观测。", "私有控制台积分，返回许可观测。"],
            ["候选预测", "提交函数输出预测位置，双粒子还输出速度；可用解析解或自己积分。", "提交加速度；使用本地官方rift.rollout及同版本数值契约推演候选。"],
            ["拟合与分析", "参考agent已有工具接线。", "各队自行选优化器、拟合参数、符号回归、画图；官方候选积分不查隐藏真值。"],
        ], [0.19, 0.41, 0.40]),
        p("草稿law允许NumPy与SciPy数学计算，不再仅开放K1；仍须纯确定性、无文件/网络IO及副作用。库许可不开放真实宇宙查询，也不更换官方rollout或终评积分器；数值协议仍为1.0，完整SDK与远端依赖核验尚未完成。", "small"),
        h("主动与随机，差别在实验怎么选"),
        p("主动实验根据已有数据与待区分假设选择条件；随机实验按预定分布选择条件，智能体仍分析数据。主动增益是两模式成绩差，不是最终实力本身。公平归因需要相同资源与工具支持。"),
        p("例如主动80、随机65的增益是15；另一系统主动50、随机20的增益虽为30，最终主动成绩仍较低。增益用于分析选实验的贡献，草稿当前也只将其列为选做单项。", "small"),
        note("当前随机对照不能直接当严格等预算证据", "随机分支每轮一个自动实验、固定时刻；主动可一轮多实验。当前随机分支还跳过中途拟合，末尾训练提取只认experiment记录。分差不只反映选点策略。", True),
        p("执行边界也需另行设计：候选源码当前通过同进程exec编译；10秒定律超时仅在特定Unix主线程条件生效，且不覆盖所有编译/拟合阶段。这不是参赛程序与真实模拟器之间的安全隔离。", "small"),
    ], "[C2] agent.py；[C4] 编译/拟合函数；[C7] benchmark入口。")

    add("误差、解释与通过判据", [
        h("先分清原始误差、归一化误差与分数"),
        formula("d_k² = ||r_pred,k - r_true,k||²；MSE = Σ_k d_k² / N"),
        p("k遍历成功测试中被计分的粒子-时刻记录，包括各case。二维坐标平方先相加；不是距离，也不是RMSE。max_pos_error是单条记录的最大平方距离。默认case同规模时可等价看作多层均值；自定义case不同规模时，按记录数加权，不能一概说case等权。"),
        formula("归一化误差 e = MSE / V_world"),
        p("论文以测试轨迹方差归一化，便于跨世界比较。当前汇总脚本使用硬编码的每世界方差常数；仓库未提供其统一生成器，不能擅自宣称某一具体方差轴已验证。无噪声真值仍是指定积分器精度下的参考。"),
        table(["指标", "含义", "关键限制"], [
            ["预测误差", "候选程序对本局留出轨迹的误差。", "拟合loss不混入最终测试MSE。"],
            ["解释分", "独立LLM按逐世界rubric评0-10，再归一到0-1。", "不是程序与解释自洽性的直接检验。"],
            ["论文通过", "e&lt;0.1且解释分≥0.9。", "当前汇总脚本解释门槛为0.75；底层passed又只看原始误差。"],
            ["pass@k", "从多次尝试中取k次，至少一次联合通过的期望比例。", "不是误差的连续分数；pass@5不是单次通过率。"],
        ], [0.18, 0.43, 0.39]),
        p("论文跨世界误差采用几何均值，解释分另报，没有本赛预言80/三声明15/能量节约5百分制。全基准含22世界和多次尝试，不能直接当本赛9宇宙完成率。"),
        formula("S_pred = 80 × clip(1 - 0.5 log10(e/0.01), 0, 1)"),
        p("e≤0.01记80、e=0.1记40、e≥1记0，零误差直接记满分。三声明各5分共15。仅在S_pred+S_law严格&gt;20时，S_save=5×(1-E_used/100)，否则为0；不计时间，恰好20不获节约分。独立自洽项已取消。", "small"),
        p("统计不确定性也不是模型自报信心。论文以重复尝试和bootstrap报告稳定性；草稿当前以9个宇宙重采样，其区间长度只在均分相同时参与排名，不能等同于同一宇宙多seed重跑的方差。", "small"),
        note("隐藏测试的两层含义", "11个保留世界是世界集合是否公开；同一公开世界也有本局未观测的留出轨迹。当前公开仓库能读到默认测试配置，因此它不是人类竞赛级保密测试。"),
    ], "[P1] 第6-8页；[C4] evaluator.py误差记录；[C7] 归一化、门槛与汇总。")

    add("难度：机制估计与模型实测分开", [
        p("下表是综合实验设计、机制识别与预测实现负担的估计。不是官方实测名次，不保证所有白名单模型遵循此序。"),
        table(["序", "宇宙", "估计依据"], [
            ["1", "gravity", "静态单源纯幂律，可逐项控制。"],
            ["2", "coulomb_easy", "低维逆平方，但要辨明属性乘积而非惯性比。"],
            ["3", "fractional", "运动仍是纯幂律；算子解释与源码版本不一致另算。"],
            ["4", "yukawa", "需要跨屏蔽尺度，排除局部幂律近似。"],
            ["5", "ether", "分离定向背景加速度，多个可见运动相互验证。"],
            ["6", "hubble", "跨半径分离中心吸引和增长的外推。"],
            ["7", "oscillator", "发现时间依赖、变号与周期，防止混叠。"],
            ["8", "circle", "多体相互作用与预测；对称性降低难度。"],
            ["9", "extra_dimensions", "普通范围伪装成gravity，要主动探近场。"],
            ["10", "three_species", "多个源与隐藏类别、正负及相对强度。"],
            ["11", "dark_matter", "不可见状态、源强与力律之间存在退化。"],
        ], [0.07, 0.26, 0.67]),
        h("论文图5的另一份排序"),
        p("去掉private列后的显示顺序为：gravity → yukawa → hubble → ether → oscillator → coulomb_easy → circle → extra_dimensions → fractional → dark_matter → three_species。按Opus4.7解释分中位数排序，有并列，不是按MSE或平均分统一排序。", "small"),
        note("相邻名次不应冻结为难度定理", "三种背景/时变复合作用的相邻顺序可互换。最后两项也受评分对象影响：three_species评35粒子，dark_matter只评5探针。oscillator成功示例4个LLM轮提交，说明关键实验设计可能比增加轮数更重要。"),
    ], "[P1] 第10页图5、附录A；[C3-C6] 世界实现；难度表为研究估计。")

    add("实验规模与草稿额度的数量级", [
        p("附录E示例中的拟合日志可以读出已有实验记录总数。每条对应一组输入/输出，不是LLM轮数；最终拟合只抽4条，也不代表只做过4个实验。"),
        table(["宇宙", "示例记录数", "如何使用这个参照"], [
            ["gravity / yukawa", "14 / 25", "单移动粒子实验；示例不全成功，不能当最低需求。"],
            ["fractional / extra_dimensions", "38 / 30", "说明探索可能有数十组配置；草稿属性目录与批量能力会改变需求。"],
            ["circle", "8", "每组10环粒子相互作用，不能折算成8或80个被动探针。"],
            ["ether", "12", "每组5探针，相当于60次探针投放。"],
            ["hubble", "9", "每组5探针，相当于45次探针投放。"],
        ], [0.32, 0.15, 0.53]),
        h("近似同步应比较相对消耗"),
        formula("O/20 ≈ P/100；即 P ≈ 5O"),
        p("O为observe次数，P为付费新探针数。12次背景与60探针同为60%；9次与45探针同为45%。不要求耗满，也不强制每次launch都投5个。100探针按8个分批需13次，按1个分批可有100次。"),
        p("launch每批共用readout，每探针至多50个采样时刻：8探针×50=400个探针-时刻状态，非整批合计50。每状态含二维pos/vel，不把四个标量另算四次。observe的50则是50帧全体可见背景位置。", "small"),
        p("若相同跨度D内含两端点且等距采样，间隔Δ=D/(N-1)。20点比50点少60%，间隔扩大49/19≈2.579倍；不是积分成本也降60%。现仍最多50，未授权改20；50非实测最优，readout也不强制等距。", "small"),
        table(["待试做机制", "背景次数目标", "探针数目标"], [
            ["简单固定中心、纯幂律", "0-4", "12-40"],
            ["屏蔽/跨尺度/时变，背景简单", "1-6", "24-64"],
            ["有用动态背景、复合作用", "8-16", "40-80"],
            ["可解改造后的复杂/隐藏背景", "12-20", "60-100"],
        ], [0.56, 0.22, 0.22]),
        p("上表是规划目标，不是DP测出的需求。六个固定中心世界不应为消耗额度而反复看背景。复用已有观测应允许；重新读取独立噪声是新测量。若20次常在探针尚余很多时形成真实瓶颈，应调整背景预算/分批方式或信息依赖，而不是做无效投放填平比例。", "small"),
    ], "[P1] 附录E；[C4] _extract_training_trajectories；[D1] 同步目标，区间为待校准估计。")

    add("DP与现行草稿：流程对照", [
        p("快照截至更新日。定律参数固定、位置噪声和无噪声参考已对齐；以下重点列仍影响实验策略的差异。"),
        table(["环节", "DP公开实现", "现行草稿"], [
            ["任务入口", "任务与拓扑说明；默认无同等第一束光包。", "mission启动窗口，公开控量、范围、噪声和数值/提交版本，不给真值。"],
            ["实验对象", "单移动粒子、五探针或相互作用环。", "理想被动探针连续受力、自由移动；pos/vel仅为初态，不扰动背景。"],
            ["物理控制", "部分能调源强/惯性/电荷或真实质量。", "按mission目录控制source.p1、probe.p2或probe.mass，以及初态、投放时刻。"],
            ["数据返回", "相关实验联合返可见背景与探针位置、速度。", "observe只有背景位置；launch返探针位置、速度。"],
            ["辅助真值", "部分给干净背景初态；ether/hubble给真实质量。", "尚未承诺等量干净信息。"],
            ["时间/空间", "多数实验重建；oscillator可start_time；典型范围。", "统一t=0并重放B(t)，不重置相位；t_launch+readout为绝对采样时刻；投放区只限初始pos。"],
            ["采样/预算", "常见≤10时刻，LLM轮限制，一轮可多实验。", "每批≤8探针，共用≤50个时刻，每探针均可返回50状态；A为100点和20次背景。"],
            ["分析/结束", "已有参考编排；预测函数自行生成轨迹。", "各队编排工具；加速度+统一积分自测，可反复存草稿。"],
        ], [0.15, 0.43, 0.42]),
        note("第一束光已确定暂定规模", "免费提供2个标准探针各8个读数，加同期8个时刻的可见背景位置；探针取该世界公开默认属性。包中给配置与时间元数据，仅生成一次，archive复用缓存不重抽噪声，不扣付费探针或observe额度。时窗和位置仍须按世界校准、赛前冻结。"),
        p("现行探索仍沿A：100点、调用/读数限额及单宇宙60分钟，不加LLM轮数。B留历史、gauge停用；launch键与上限不变，第一束光预置、observe只选时间。能量节约计5分，完整SDK仍待实现。", "small"),
    ], "[D1] 最新草稿、项目记忆与现行探索契约；[C1-C3] DP协议与执行。")

    add("草稿属性、声明与规则快照", [
        p("mission控量目录、source_properties与probe属性操作不变；删声明不改变实验键、数值接口或PARAMS/BODY_PROPS/HIDDEN表达。旧12字段已退役，仅保留历史。", "small"),
        h("claims 3.0.0：只保留三个字段"),
        table(["字段", "格式", "含义与分值"], [
            ["dynamics_order", "1 / 2", "粒子运动阶数，5分。加速度接口已给二阶，是易得分字段，按用户决定保留。"],
            ["hidden_matter", "true / false", "是否存在不可见作用源，5分；未知可见物种不等于暗源。"],
            ["time_dependence", "none / periodic / other", "固定几何/状态下的显式时间依赖，5分；不从轨道周期直接推断。"],
        ], [0.28, 0.25, 0.47]),
        p("三项必填，不接受旧字段。periodic要求作用对显式t非恒定且按T&gt;0重复；恒定归none。声明不含周期T，轨道周期不是显式时变；旧自洽对照已取消。", "small"),
        h("九宇宙与现行主榜规则简记"),
        table(["档位", "名字（顺序已确定）", "档位总权重"], [
            ["一级", "Adam、Lilith、Tabris", "30%"],
            ["二级", "Sachiel、Ramiel、Matriel", "40%"],
            ["三级", "Leliel、Arael、Lilin", "30%"],
        ], [0.15, 0.65, 0.20]),
        p("六项保真迁移、三项自设计；具体对应与三声明真值私有。<br/>总100：预言80+三声明15+能量节约5；独立自洽取消。<br/>节约仅在预言与定律之和严格&gt;20时启用，按剩余能量比例计，不按时间计。<br/>排名按加权均分，同分比95%区间长度，再比少自由度；PARAMS≤5、奥卡姆保留，两奖停用、主动增益选做。", "small"),
        p("三声明与私有标签比较，不等于完整物理规律已正确；完整轨迹仍由预言项检验。旧自洽误判/耗时讨论只作研究，已不属于现行计分；完整控制台与端到端评测仍待验收。", "small"),
    ], "[D1] 最新草稿、项目记忆及评分设计；claims 3.0.0；完整竞赛实现仍待验收。")

    add("自动实验、保密边界与待办", [
        LoopDiagram(["各队LLM<br/>智能体与SDK", "私有控制台<br/>校验 / 计费", "真实模拟器<br/>积分与筛选", "许可观测<br/>返回该队"]),
        p("组委会侧应部署自动服务，不由工作人员逐轮收发。完整智能体进入会话后可反复实验；每次launch不必重新上传镜像、重载模型或人工审批。拟合与绘图由各队完成，本地候选积分只运行选手自己的定律，不获得隐藏真值。"),
        table(["边界", "需要满足的条件"], [
            ["开发与正式世界", "公开练习源码可本地无限生成数据；正式私有配置、力律和验证数据留在组委会侧。"],
            ["部署", "组委会管理同机私有进程和受控IPC，或独立服务节点。参赛者有宿主机管理员权限时，不应同机保存私有实现。"],
            ["数据", "mission给任务与公开拓扑/操作协议、控量范围、噪声/数值/提交版本；观测只返许可数据和余额，不返定律、隐藏源数量或内部配置。"],
            ["候选HIDDEN", "来自选手推断，不能把真实隐藏源自动传入候选bodies。"],
            ["资源与可靠性", "服务端保存预算/时间，独立会话并发；积分、吞吐、排队影响均需实测。"],
        ], [0.22, 0.78]),
        h("九宇宙定义的试做与工程待办"),
        p("① 校准第一束光位置/时窗。② 验证t=0与B(t)重放、缓存/噪声语义。③ 明确干净初态边界。④ 校准六迁移/三自设计空间与宽松时间护栏，完成端到端试做。⑤ 实现80/15/5及严格&gt;20节约资格，核边界。⑥ 私有通信、权限、并发与独立墙钟。"),
        note("提案与事实分开", "上述自动服务是实现建议，现仓库尚无完整rift竞赛服务。仅把真实代码另开进程、设成只读、改名或不放进提示词，不自动构成保密。DP源码的执行隔离也不能直接当正式竞赛基础设施。", True),
        p("旧自洽研究（该项已取消、不计分）：有限t采样一般不能证明周期、非周期或无时变，更不能据此证明暗源必要。动态背景和几何重建会影响诊断，三字段无周期值也使旧对照失效。此处仅留研究记录，未实施相关判据或阈值。", "small"),
    ], "[C3-C4] DP执行与编译；[D1] 第7.10、7.15节及现行公平条款。")

    commit = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
    base = f"https://github.com/SampsonML/DiscoverPhysics/blob/{commit}/"

    def link(label: str, relative: str) -> str:
        source_base = base.replace("/blob/", "/tree/") if relative == "PhysicsSchool/prompts" else base
        return f'<link href="{source_base + relative}" color="#007E83">{label}</link>'

    add("来源索引与核查提醒", [
        table(["标识", "来源与定位（链接可点击）"], [
            ["P1 论文", '<link href="https://arxiv.org/abs/2605.26087" color="#007E83">DiscoverPhysics: Benchmarking LLMs for Out-of-the-Box Scientific Thinking</link>。第6-8页评价/噪声；第10页图5；附录A实验过程；附录C公开世界；附录E示例日志。'],
            ["C1 协议", link("PhysicsSchool/prompts/", "PhysicsSchool/prompts") + "：2particle、circle、probes、probes_with_masses。"],
            ["C2 主循环", link("ScienceAgent/scienceagent/agent.py", "ScienceAgent/scienceagent/agent.py") + "；入口run_discovery.py。"],
            ["C3 执行器", link("ScienceAgent/scienceagent/executor.py", "ScienceAgent/scienceagent/executor.py") + "：实验初态、真实属性、数据筛选、start_time。"],
            ["C4 评价/拟合", link("ScienceAgent/scienceagent/evaluator.py", "ScienceAgent/scienceagent/evaluator.py") + "：默认case、计分对象、误差、编译、拟合与数据提取。"],
            ["C5 物理底座", link("nbody_sampler.py", "PhysicsSchool/physchool/worlds/nbody_sampler.py") + "：139-149行源/响应索引；" + link("force_laws.py", "PhysicsSchool/physchool/worlds/force_laws.py") + "：核与幂律。"],
            ["C6 世界配置", link("ScienceAgent/scienceagent/worlds.py", "ScienceAgent/scienceagent/worlds.py") + "：真实参数、说明、rubric、拓扑映射。"],
            ["C7 汇总", link("scripts/run_benchmark.py", "scripts/run_benchmark.py") + "、run_stats.py及configs/bench.yml：轮次、seed、噪声、方差常数、门槛。"],
            ["D1 草稿", "项目内docs/彼岸物理规律发现挑战赛_赛题说明.docx及AGENTS.md，截至本册更新日。本文不是替代规则。"],
        ], [0.19, 0.81]),
        h("迁移前不能照抄的四处结论"),
        p("1. fractional的α、衰减文字与rubric存在不一致。<br/>2. three_species提示只评5，代码实际评35。<br/>3. 论文解释门槛0.9、脚本0.75、底层passed各异。<br/>4. 默认10轮、论文16轮、每轮多实验、拟合占轮与时间保护，不能合并成一个硬实验额度。", "small"),
        p("源码定位基于上列固定commit；新增版本应重审。该PDF使用嵌入中文字体、可搜索文字、目录书签和稳定输出路径。修改研究事实或草稿后，可更新本生成脚本并重建；最终规则以用户确认的现行草稿为准。", "caption"),
    ], "来源核查截止 " + date + "；本册不披露命题对应与真值，三项自设计已有定义、待试做。")
    return pages


def draw_page(c: canvas.Canvas, number: int, total: int, page: dict, date: str) -> dict:
    c.bookmarkPage(f"p{number}")
    c.addOutlineEntry(f"{number:02d}  {page['title']}", f"p{number}", level=0)
    c.setFillColor(TEAL)
    c.rect(MARGIN, HEIGHT - 38, 24, 3, fill=1, stroke=0)
    c.setFont("CJK", 8.2)
    c.setFillColor(MUTED)
    c.drawString(MARGIN + 34, HEIGHT - 38, "DISCOVERPHYSICS / 研究参考")
    c.drawRightString(WIDTH - MARGIN, HEIGHT - 38, f"{DOC_REVISION} · {date}")
    title = p(page["title"], "title")
    _, th = title.wrap(CW, 80)
    y = HEIGHT - 67 - th
    title.drawOn(c, MARGIN, y)
    y -= 12
    if page.get("subtitle"):
        flow = p(page["subtitle"], "small")
        _, fh = flow.wrap(CW, HEIGHT)
        flow.drawOn(c, MARGIN, y - fh)
        y -= fh + 8
    start_y = y
    for index, flow in enumerate(page["items"]):
        before = flow.getSpaceBefore() if hasattr(flow, "getSpaceBefore") else getattr(flow, "spaceBefore", 0)
        after = flow.getSpaceAfter() if hasattr(flow, "getSpaceAfter") else getattr(flow, "spaceAfter", 0)
        y -= before
        _, fh = flow.wrap(CW, HEIGHT)
        if y - fh < 74:
            raise RuntimeError(f"Page {number} '{page['title']}' overflow at item {index}: bottom={y-fh:.1f}; need >=74")
        flow.drawOn(c, MARGIN, y - fh)
        y -= fh + after
    source = p("依据：" + escape(page["sources"]), "caption")
    _, sh = source.wrap(CW, 70)
    if y - sh < 59:
        raise RuntimeError(f"Page {number} sources overflow: y={y:.1f}, h={sh:.1f}")
    source.drawOn(c, MARGIN, y - sh - 2)
    c.setStrokeColor(GRID)
    c.setLineWidth(0.5)
    c.line(MARGIN, 44, WIDTH - MARGIN, 44)
    c.setFillColor(MUTED)
    c.setFont("CJK", 8)
    c.drawString(MARGIN, 29, "公开11宇宙 / 论文、代码与草稿分开解读")
    c.drawRightString(WIDTH - MARGIN, 29, f"{number:02d} / {total:02d}")
    c.showPage()
    return {"page": number, "title": page["title"], "body_top": round(start_y, 1), "body_bottom": round(y, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font")
    parser.add_argument("--font-bold")
    parser.add_argument("--date", default="2026-10-30")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    register_fonts(args.font, args.font_bold)
    pages = build_pages(args.date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".building.pdf")
    c = canvas.Canvas(str(temporary), pagesize=A4, pageCompression=1)
    c.setTitle("DiscoverPhysics研究笔记：公开11宇宙与草稿对照")
    c.setAuthor("DiscoverPhysics 项目研究整理")
    c.setSubject("实验、观测、源强、响应、惯性、拟合、评估与赛题接口")
    c.setCreator("scripts/build_dp_reference.py / ReportLab")
    manifest = [draw_page(c, i, len(pages), page, args.date) for i, page in enumerate(pages, 1)]
    c.save()
    temporary.replace(args.output)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "pages": len(pages), "bytes": args.output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
