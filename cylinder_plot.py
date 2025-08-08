import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import legend
from matplotlib.ticker import MaxNLocator
import math

# 设置中文字体路径
zh_font = FontProperties(fname=r"C:\Users\neilj\AppData\Local\Microsoft\Windows\Fonts\方正小标宋_GBK.TTF", size=12)

def plot_cylinder_type_by_company(df_cylinder, df_type, company_mapping, txt_path="./gas_cylinder/各企业气瓶状态数量统计.txt"):
    """
    绘制每个企业各类气瓶数量的堆叠横向柱状图，并标注总数与各类占比。
    图表底部展示总气瓶数量及各类气瓶数量。

    参数：
        df_type: 包含 ['companyCode', 'status', 'count']
        company_mapping: 包含 ['companyCode', 'company']
    """
    if df_cylinder.empty:
        return

    # 状态映射
    status_mapping = {0: "在用", 1: "报废", 2: "注销", 3: "停用", 4: "其它"}

    # 合并中文企业名
    df_type = df_type.merge(company_mapping, on='companyCode', how='left')

    # 处理、记录无法匹配的气瓶
    # 从 df_cylinder 中筛选出无法匹配的 companyCode 所对应的记录
    unmatched_codes = df_type[df_type['company'].isna()]['companyCode'].tolist()
    df_unmatched = df_cylinder[df_cylinder['companyCode'].isin(unmatched_codes)].copy()
    df_unmatched['company'] = df_unmatched['company'].fillna('未知企业')
    df_grouped = df_unmatched.groupby(['companyCode', 'company'])['serialCode'].count().reset_index()
    df_grouped = df_grouped.rename(columns={'serialCode': 'count'})
    parts = [f"{row['company']}（{row['companyCode']}）：{row['count']}个" for _, row in df_grouped.iterrows()]
    summary_str = "气瓶库表中企业编码与企业库表无法匹配，对应的企业名称（编码）及气瓶数量为：" + "；".join(parts)

    # 构建透视表
    pivot_df = df_type.pivot_table(
        index='company',
        columns='status',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns))
    pivot_df["total"] = pivot_df.sum(axis=1)
    pivot_df = pivot_df.sort_values(by="total", ascending=True)
    pivot_df = pivot_df.drop(columns=["total"])

    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot_df) * 0.5)))
    pivot_df.plot(kind='barh', stacked=True, ax=ax)

    plt.title(f"各企业气瓶状态数量统计", fontproperties=zh_font)
    plt.xlabel("数量", fontproperties=zh_font)
    plt.ylabel("企业名称", fontproperties=zh_font)
    plt.yticks(fontproperties=zh_font, fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注每个条块数值与比例
    for bar_idx, (company_name, row) in enumerate(pivot_df.iterrows()):
        total = row.sum()
        width_accum = 0
        for j, (status, value) in enumerate(row.items()):
            if value > 0:
                bar = ax.patches[bar_idx + j * len(pivot_df)]
                width = bar.get_width()
                ax.text(
                    width_accum + width / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f'{int(value)} ({value / total:.2%})',
                    ha='center',
                    va='center',
                    color='red',
                    fontsize=12,
                    fontproperties=zh_font
                )
                width_accum += width
        ax.text(
            total + 0.01 * total,
            bar.get_y() + bar.get_height() / 2,
            f'{int(total)}',
            ha='left',
            va='center',
            fontsize=10,
            fontproperties=zh_font,
            fontweight='bold'
        )

    # 图例设置
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [status_mapping.get(int(l), l) for l in labels]
    ax.legend(
        handles,
        new_labels,
        title="状态",
        fontsize=12,
        prop=zh_font,
        title_fontproperties=zh_font,
        loc='lower right'
    )

    # 汇总信息写入 x 轴标题（每张图相同）
    total_counts = df_type.groupby("status")["count"].sum().to_dict()
    total_all = sum(total_counts.values())
    status_texts = [f'{status_mapping.get(k)}：{total_counts.get(k, 0)}' for k in sorted(status_mapping)]
    summary_text = f'各地市在库气瓶总数：{total_all}个; ' + '个, '.join(status_texts)
    plt.xlabel(summary_text, fontproperties=zh_font, fontsize=16, color='red')

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"./gas_cylinder/各企业气瓶状态数量统计.png", dpi=300)
    plt.close()

    # 导出 TXT 文件
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("【气瓶状态汇总统计】\n")
        f.write(summary_str + "\n\n")
        f.write(summary_text + "\n\n")
        f.write("【各企业气瓶状态统计明细】\n")
        f.write("企业名称\t" + "\t".join([status_mapping.get(s) for s in sorted(pivot_df.columns)]) + "\n")
        for company, row in pivot_df.iterrows():
            values = "\t".join([str(int(row.get(s, 0))) for s in sorted(pivot_df.columns)])
            f.write(f"{company}\t{values}\n")

def plot_overdue_check(df_check, df_total, company_mapping, txt_path="./gas_cylinder/各企业已超核验时间但仍被标记为在用气瓶的数量明细.txt"):
    """
    绘制各站点超过下次检验日期的气瓶数量柱状图（只标注数量），并在x轴标题显示总数与前三名公司。

    参数：
        df_check: DataFrame，包含 ['companyCode', 'overdue_check_count']
        company_mapping: DataFrame，包含 ['companyCode', 'company']
    """
    # 合并数据
    df_plot = df_total.merge(df_check, on='companyCode', how='left')
    # 用 0 填充数量类字段
    df_plot['overdue_check_count'] = df_plot['overdue_check_count'].fillna(0).astype(int)
    # 用空列表填充列表类字段（必须用 apply）
    df_plot['overdue_check_gasids'] = df_plot['overdue_check_gasids'].apply(lambda x: x if isinstance(x, list) else [])
    df_plot = df_plot.merge(company_mapping, on='companyCode', how='left')
    df_plot['overdue_ratio'] = df_plot['overdue_check_count'] / df_plot['total_count']
    # 剔除缺失值
    df_plot = df_plot.dropna(subset=['company'])
    # 按总数升序排列，保持横向柱状图一致性
    df_plot = df_plot.sort_values(by='total_count', ascending=True).reset_index(drop=True)

    # 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(14, max(6, len(df_plot) * 0.4)))

    # 绘制总数与超期数量的双层横向条形图
    ax.barh(df_plot['company'], df_plot['total_count'], color='#4E79A7', label='在用状态气瓶数')
    ax.barh(df_plot['company'], df_plot['overdue_check_count'], color='#F28E2B', label='核验超期气瓶数')

    # 添加每个柱体末端的文本标注（数量/总数，占比）
    for i, row in df_plot.iterrows():
        total = row['total_count']
        overdue = row['overdue_check_count']
        ratio = f"{int(overdue)}/{int(total)} ({(overdue / total):.2%})" if total > 0 else "0/0"
        ax.text(
            overdue + 1, i,
            ratio,
            va='center',
            fontsize=12,
            fontproperties=zh_font,
            color='black'
        )

    # 图形样式设置
    ax.set_xlabel("气瓶数量", fontproperties=zh_font)
    ax.set_title("各企业在用气瓶总数与超期检验数量（含占比）", fontproperties=zh_font)
    plt.xticks(fontproperties=zh_font)
    plt.yticks(fontproperties=zh_font, fontsize=9)
    ax.legend(prop=zh_font)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 总数与前三公司文字
    total_overdue = df_plot['overdue_check_count'].sum()
    top3 = df_plot.sort_values(by='overdue_check_count', ascending=False).head(3)
    top3_text = "，".join([
        f"{row['company']}（{int(row['overdue_check_count'])}个）"
        for _, row in top3.iterrows()
    ])
    bottom_text = f"各地市已超下次检验日期的在库气瓶总数：{int(total_overdue)}；前三名：{top3_text}"

    # 添加底部说明文字
    fig.text(
        0.5, -0, bottom_text,
        ha='center',
        va='top',
        fontsize=12,
        fontproperties=zh_font,
        color='red'
    )

    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig("./gas_cylinder/超期检验气瓶统计.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 导出 TXT 文件
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("【超过下次检验日期的气瓶总数】\n")
        f.write(f"{bottom_text} \n")
        f.write(f"总超期气瓶数：{total}\n\n")

        f.write("【超期气瓶排名前三】\n")
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            f.write(f"{i}. {row['company']}：{int(row['overdue_check_count'])}\n")
        f.write("\n")

        f.write("【各公司超期气瓶数量明细】\n")
        f.write("公司名称\t超期数量\t气瓶编号列表\n")
        for _, row in df_plot.iterrows():
            ids = ','.join(row['overdue_check_gasids'])
            f.write(f"{row['company']}\t{int(row['overdue_check_count'])}\t{ids}\n")

def plot_overdue_scrap(df_scrap, df_total, company_mapping, txt_path="./gas_cylinder/各企业已过报废时间但仍被标记为在用气瓶的数量明细.txt"):
    """
    绘制各站点超过报废年月的气瓶数量柱状图（只标注数量）。
    参数：
        df_scrap: DataFrame，包含 ['station_id', 'overdue_scrap_count']
        station_mapping: DataFrame，包含 ['station_id', 'station_name']
    """
    # 合并数据
    df_plot = df_total.merge(df_scrap, on='companyCode', how='left')
    # 用 0 填充数量类字段
    df_plot['overdue_scrap_count'] = df_plot['overdue_scrap_count'].fillna(0).astype(int)
    # 用空列表填充列表类字段（必须用 apply）
    df_plot['overdue_scrap_gasids'] = df_plot['overdue_scrap_gasids'].apply(lambda x: x if isinstance(x, list) else [])
    df_plot = df_plot.merge(company_mapping, on='companyCode', how='left')
    df_plot['overdue_ratio'] = df_plot['overdue_scrap_count'] / df_plot['total_count']
    df_plot = df_plot.sort_values(by='total_count', ascending=True)
    # 剔除缺失值
    df_plot = df_plot.dropna(subset=['company'])
    # 按总数升序排列，保持横向柱状图一致性
    df_plot = df_plot.sort_values(by='total_count', ascending=True).reset_index(drop=True)

    # 创建图像和坐标轴
    fig, ax = plt.subplots(figsize=(14, max(6, len(df_plot) * 0.4)))

    # 绘制总数与超期数量的双层横向条形图
    ax.barh(df_plot['company'], df_plot['total_count'], color='#4E79A7', label='在用状态气瓶数')
    ax.barh(df_plot['company'], df_plot['overdue_scrap_count'], color='#F28E2B', label='超期报废气瓶数')

    # 添加每个柱体末端的文本标注（数量/总数，占比）
    for i, row in df_plot.iterrows():
        total = row['total_count']
        overdue = row['overdue_scrap_count']
        ratio = f"{int(overdue)}/{int(total)} ({(overdue / total):.2%})" if total > 0 else "0/0"
        ax.text(
            overdue + 1, i,
            ratio,
            va='center',
            fontsize=12,
            fontproperties=zh_font,
            color='black'
        )

    # 图形样式设置
    ax.set_xlabel("气瓶数量", fontproperties=zh_font)
    ax.set_title("各企业在用气瓶总数与超期报废数量（含占比）", fontproperties=zh_font)
    plt.xticks(fontproperties=zh_font)
    plt.yticks(fontproperties=zh_font, fontsize=9)
    ax.legend(prop=zh_font)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 总数与前三公司文字
    total_overdue = df_plot['overdue_scrap_count'].sum()
    top3 = df_plot.sort_values(by='overdue_scrap_count', ascending=False).head(3)
    top3_text = "，".join([
        f"{row['company']}（{int(row['overdue_scrap_count'])}个）"
        for _, row in top3.iterrows()
    ])
    bottom_text = f"各地市已超报废日期的在库气瓶总数：{int(total_overdue)}；前三名：{top3_text}"

    # 添加底部说明文字
    fig.text(
        0.5, -0, bottom_text,
        ha='center',
        va='top',
        fontsize=12,
        fontproperties=zh_font,
        color='red'
    )

    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig("./gas_cylinder/超期报废气瓶统计.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 输出 TXT 文件（详细）
    with open(txt_path, "w", encoding="utf-8") as f:
        total = df_plot['overdue_scrap_count'].sum()
        f.write(f"{bottom_text} \n")
        f.write("【超期报废气瓶总数】\n")
        f.write(f"总超期报废气瓶数：{total}\n\n")

        f.write("【超期报废气瓶排名前三】\n")
        top3 = df_plot.head(3)
        for i, (_, row) in enumerate(top3.iterrows(), 1):
            f.write(f"{i}. {row['company']}：{int(row['overdue_scrap_count'])}\n")
        f.write("\n")

        f.write("【各公司超期报废气瓶数量明细】\n")
        f.write("公司名称\t超期数量\t气瓶编号列表\n")
        for _, row in df_plot.iterrows():
            ids = ','.join(row['overdue_scrap_gasids'])
            f.write(f"{row['company']}\t{int(row['overdue_scrap_count'])}\t{ids}\n")

def plot_all_pie_charts(df, df_type, current_time):
    """
    生成三个饼图合并为一张图，优化标注不重叠问题：
    1. 各类型气瓶占比
    2. 在用气瓶中超过下次检验日期占比
    3. 在用气瓶中超过报废日期占比
    """

    if current_time is None:
        current_time = pd.Timestamp.now()

    status_mapping = {0: "在用", 1: "报废", 2: "注销", 3: "停用", 4: "其它"}

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    def draw_pie(ax, data, labels, title, highlight_top_n=None):
        total = sum(data)
        wedges, _ = ax.pie(
            data,
            startangle=90,
            wedgeprops=dict(width=0.55),
            labels=None
        )

        # 排序确定需标注的索引
        if highlight_top_n is not None:
            sorted_indices = np.argsort(data)[::-1][:highlight_top_n]
        else:
            sorted_indices = range(len(data))

        for i, (wedge, label, value) in enumerate(zip(wedges, labels, data)):
            if i not in sorted_indices:
                continue  # 跳过不需要标注的

            angle = (wedge.theta2 + wedge.theta1) / 2
            x = np.cos(np.deg2rad(angle))
            y = np.sin(np.deg2rad(angle))

            label_x = 1.2 * x
            label_y = 1.2 * y
            ha = 'left' if i%2!=0 else 'right'
            annotation = f"{label}: {value:,}（{value / total:.2%}）"

            ax.annotate(
                annotation,
                xy=(x * 1, y * 1),
                xytext=(label_x, label_y),
                ha=ha,
                va='center',
                fontsize=12,
                fontproperties=zh_font,
                textcoords='data',
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.8),
                # bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.3)
            )

        ax.set_title(title, fontproperties=zh_font, fontsize=12)

    # 图1：仅标注最多的两个状态
    df_sum = df_type.groupby('status')['count'].sum().reset_index()
    df_sum['status_name'] = df_sum['status'].map(status_mapping)
    draw_pie(
        axs[0],
        df_sum['count'].values,
        df_sum['status_name'].values,
        " ",
        # "各地市在库气瓶类型占比统计",
        highlight_top_n=4
    )

    # 图2：在用气瓶超下次检验
    df_use_check = df[df['status'] == 0].copy()
    df_use_check['nextCheckDate'] = pd.to_datetime(df_use_check['nextCheckDate'], errors='coerce')
    total_check = df_use_check['serialCode'].count()
    overdue_check = df_use_check[df_use_check['nextCheckDate'] < current_time]['serialCode'].count()
    valid_check = total_check - overdue_check
    draw_pie(
        axs[1],
        [overdue_check, valid_check],
        ['已超期', '未超期'],
        " ",
        # "各地市在库气瓶超下次核验时间占比统计"
    )

    # 图3：在用气瓶超报废日期
    df_use_scrap = df[df['status'] == 0].copy()
    df_use_scrap['scrapDate'] = pd.to_datetime(df_use_scrap['scrapDate'], errors='coerce') + pd.offsets.MonthEnd(0)
    total_scrap = df_use_scrap['serialCode'].count()
    overdue_scrap = df_use_scrap[df_use_scrap['scrapDate'] < current_time]['serialCode'].count()
    valid_scrap = total_scrap - overdue_scrap
    draw_pie(
        axs[2],
        [overdue_scrap, valid_scrap],
        ['已超期', '未超期'],
        " ",
        # "各地市在库气瓶超报废时间占比统计"
    )

    # 总标题与布局
    fig.suptitle("各地市在库气瓶状态统计", fontproperties=zh_font, fontsize=16, y=0.95)
    plt.subplots_adjust(top=0.88, bottom=0.1, wspace=0.3)
    plt.savefig("./gas_cylinder/各地市在库气瓶状态统计.png", dpi=300, bbox_inches='tight')
    plt.close()