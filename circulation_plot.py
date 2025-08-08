import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import legend
from matplotlib.ticker import MaxNLocator
from networkx.algorithms.bipartite.basic import color
import math
import os

status_mapping = {
    '1': '充装前检查',
    '2': '充装',
    '3': '充装后检查',
    '4': '出站',
    '5': '自提',
    '6': '开始配送',
    '7': '配送到户',
    '8': '收回',
    '9': '入站',
    '10': '回收',
    '20': '气瓶损坏'
}

# 设置中文字体路径
zh_font = FontProperties(fname=r"C:\Users\neilj\AppData\Local\Microsoft\Windows\Fonts\方正小标宋_GBK.TTF", size=12)


def extract_five_hours(hourly_stats):
    """
    提取每天 00:00 - 05:00 的跳转数据。

    参数：
        hourly_stats: 包含 ['hour_start', 'hour_stop', 'count_hourly'] 的 DataFrame

    返回：
        DataFrame：仅包含每天 0 点至 5 点的记录
    """
    hourly_stats_filtered = hourly_stats.copy()
    hourly_stats_filtered["hour"] = hourly_stats_filtered["hour_start"].dt.hour

    # 保留 hour ∈ [0, 5] 的行
    night_stats = hourly_stats_filtered[hourly_stats_filtered["hour"].between(0, 5)]

    return night_stats.drop(columns=["hour"])

def plot_abnormal_cylinders_in_use(df, company_mapping, output_txt_path="./gas_circulation/异常气瓶在流转统计明细.txt"):
    """
    绘制异常气瓶记录的公司分布图（横向柱状图）

    参数：
        df_abnormal: 包含异常气瓶记录的 DataFrame，需包含字段 ['company_code']
        company_mapping: 公司映射表 DataFrame，需包含字段 ['company_code', 'company_name']

    输出：
        保存横向柱状图 "异常气瓶企业分布图.png"
    """

    # 按公司统计异常数量及编号列表
    company_count = df.groupby("company_code").agg(
        abnormal_count=("gas_id", "count"),
        cylinder_list=("gas_id", lambda x: list(x))
    ).reset_index()

    # 合并公司名称
    company_count = company_count.merge(company_mapping, left_on='company_code', right_on='companyCode', how='left')
    company_count = company_count.sort_values("abnormal_count", ascending=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(company_count["company"], company_count["abnormal_count"], color="blue")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                 va='center', fontsize=9, fontproperties=zh_font)

    # 设置轴和标题
    ax = plt.gca()
    ax.set_xlabel("数量", fontproperties=zh_font, fontsize=9)
    ax.set_ylabel("企业", fontproperties=zh_font, fontsize=9)
    ax.set_title("异常气瓶在用数量分布图", fontproperties=zh_font, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=zh_font, fontsize=9)

    # 美观处理
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/异常气瓶(含超报废时间&超核验时间)在用分布图.png", dpi=300)
    plt.close()

    # 导出 TXT 文件
    total = company_count["abnormal_count"].sum()
    top3 = company_count.sort_values("abnormal_count", ascending=False).head(3)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("【异常气瓶在用总数】\n")
        f.write(f"异常气瓶在用总数：{total}\n\n")

        f.write("【异常气瓶在用企业排名前三】\n")
        for i, row in top3.iterrows():
            f.write(f"{i + 1}. {row['company']}：{row['abnormal_count']}\n")
        f.write("\n")

        f.write("【各公司异常气瓶在用数量明细】\n")
        f.write("公司名称\t异常扫描次数\t涉及气瓶编号列表\n")
        for _, row in company_count.iterrows():
            cylinders = ", ".join(map(str, set(row["cylinder_list"])))
            f.write(f"{row['company']}\t{row['abnormal_count']}\t{cylinders}\n")


def plot_check_cylinders_in_use(df, company_mapping, output_txt_path="./gas_circulation/异常气瓶(超核验)在流转统计明细.txt"):
    """
    绘制异常气瓶记录的公司分布图（横向柱状图）

    参数：
        df_abnormal: 包含异常气瓶记录的 DataFrame，需包含字段 ['company_code']
        company_mapping: 公司映射表 DataFrame，需包含字段 ['company_code', 'company_name']

    输出：
        保存横向柱状图 "异常气瓶企业分布图.png"
    """
    if df.empty:
        return

    # 按公司统计异常数量及编号列表
    company_count = df.groupby("company_code").agg(
        abnormal_count=("gas_id", "count"),
        cylinder_list=("gas_id", lambda x: list(x))
    ).reset_index()

    # 合并公司名称
    company_count = company_count.merge(company_mapping, left_on='company_code', right_on='companyCode', how='left')
    company_count = company_count.sort_values("abnormal_count", ascending=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(company_count["company"], company_count["abnormal_count"], color="blue")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                 va='center', fontsize=9, fontproperties=zh_font)

    # 设置轴和标题
    ax = plt.gca()
    ax.set_xlabel("数量", fontproperties=zh_font, fontsize=9)
    ax.set_ylabel("企业", fontproperties=zh_font, fontsize=9)
    ax.set_title("异常气瓶在用数量分布图", fontproperties=zh_font, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=zh_font, fontsize=9)

    # 美观处理
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/异常气瓶(超核验)在用分布图.png", dpi=300)
    plt.close()

    # 导出 TXT 文件
    total = company_count["abnormal_count"].sum()
    top3 = company_count.sort_values("abnormal_count", ascending=False).head(3)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("【异常气瓶(超核验)在用总数】\n")
        f.write(f"异常气瓶(超核验)在用总数：{total}\n\n")

        f.write("【异常气瓶(超核验)在用企业排名前三】\n")
        for i, row in top3.iterrows():
            f.write(f"{i + 1}. {row['company']}：{row['abnormal_count']}\n")
        f.write("\n")

        f.write("【各公司异常气瓶(超核验)在用数量明细】\n")
        f.write("公司名称\t异常扫描次数\t涉及气瓶编号列表\n")
        for _, row in company_count.iterrows():
            cylinders = ", ".join(map(str, set(row["cylinder_list"])))
            f.write(f"{row['company']}\t{row['abnormal_count']}\t{cylinders}\n")



def plot_scrap_cylinders_in_use(df, company_mapping, output_txt_path="./gas_circulation/异常气瓶(超报废)在流转统计明细.txt"):
    """
    绘制异常气瓶记录的公司分布图（横向柱状图）

    参数：
        df_abnormal: 包含异常气瓶记录的 DataFrame，需包含字段 ['company_code']
        company_mapping: 公司映射表 DataFrame，需包含字段 ['company_code', 'company_name']

    输出：
        保存横向柱状图 "异常气瓶企业分布图.png"
    """
    if df.empty:
        return

    # 按公司统计异常数量及编号列表
    company_count = df.groupby("company_code").agg(
        abnormal_count=("gas_id", "count"),
        cylinder_list=("gas_id", lambda x: list(x))
    ).reset_index()

    # 合并公司名称
    company_count = company_count.merge(company_mapping, left_on='company_code', right_on='companyCode', how='left')
    company_count = company_count.sort_values("abnormal_count", ascending=True)

    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(company_count["company"], company_count["abnormal_count"], color="blue")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(width)}",
                 va='center', fontsize=9, fontproperties=zh_font)

    # 设置轴和标题
    ax = plt.gca()
    ax.set_xlabel("数量", fontproperties=zh_font, fontsize=9)
    ax.set_ylabel("企业", fontproperties=zh_font, fontsize=9)
    ax.set_title("异常气瓶在用数量分布图", fontproperties=zh_font, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=zh_font, fontsize=9)

    # 美观处理
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/异常气瓶(超报废)在用分布图.png", dpi=300)
    plt.close()

    # 导出 TXT 文件
    total = company_count["abnormal_count"].sum()
    top3 = company_count.sort_values("abnormal_count", ascending=False).head(3)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("【异常气瓶(超报废)在用总数】\n")
        f.write(f"异常气瓶(超报废)在用总数：{total}\n\n")

        f.write("【异常气瓶(超报废)在用企业排名前三】\n")
        for i, row in top3.iterrows():
            f.write(f"{i + 1}. {row['company']}：{row['abnormal_count']}\n")
        f.write("\n")

        f.write("【各公司异常气瓶(超报废)在用数量明细】\n")
        f.write("公司名称\t异常扫描次数\t涉及气瓶编号列表\n")
        for _, row in company_count.iterrows():
            cylinders = ", ".join(map(str, set(row["cylinder_list"])))
            f.write(f"{row['company']}\t{row['abnormal_count']}\t{cylinders}\n")


def plot_hourly_transition_bars(hourly_stats):
    """
    绘制全局每小时状态跳转数的横向柱状图，并标注每根柱子的数值。

    参数：
        hourly_stats: DataFrame，包含 ['hour_start', 'count_hourly']

    输出：
        横向柱状图保存为 "每小时操作量.png"
    """
    # 获取全部涉及的日期
    hourly_stats = extract_five_hours(hourly_stats)

    # 构造完整时间段列（从每日 00:00 到 05:00，每小时一次）
    min_time = hourly_stats["hour_start"].min().normalize()
    max_time = hourly_stats["hour_start"].max().normalize() + pd.Timedelta(hours=5)

    full_index = pd.date_range(start=min_time, end=max_time, freq='H')
    full_index = full_index[full_index.hour.isin([0, 1, 2, 3, 4, 5])]

    # 构建 DataFrame
    full_df = pd.DataFrame({'hour_start': full_index})

    # 合并原数据（以 hour_start 为键），缺失 count_hourly 补 0
    hourly_stats = full_df.merge(hourly_stats, on='hour_start', how='left')
    hourly_stats['count_hourly'] = hourly_stats['count_hourly'].fillna(0).astype(int)
    hourly_stats['hour_stop'] = hourly_stats['hour_start'] + pd.Timedelta(hours=1)

    total = len(hourly_stats)
    chunk_size = math.ceil(total / 3)

    # 获取最大操作量，用于统一坐标轴
    max_value = hourly_stats["count_hourly"].max()
    x_limit = math.ceil(max_value * 1.1)  # 放大 10%，避免文字重叠

    for i in range(3):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk = hourly_stats.iloc[start:end]

        fig_height = max(6, len(chunk) * 0.3)
        plt.figure(figsize=(10, fig_height))
        ax = plt.gca()

        bars = ax.barh(
            chunk['hour_start'].astype(str),
            chunk['count_hourly'],
            color='steelblue'
        )

        # 设置统一 X 坐标轴范围
        ax.set_xlim(0, x_limit)

        # 计算统一右侧标注位置（最大值 + 偏移）
        max_width = hourly_stats['count_hourly'].max()
        label_x = max_width + 20  # 统一偏移

        for bar in bars:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(
                label_x,
                y,
                f"{int(width)}",
                va='center',
                ha='right',  # ← 右对齐
                fontsize=12,
                fontproperties=zh_font
            )

        # 添加红色虚线分隔：按日期
        prev_day = None
        for idx, hour in enumerate(chunk['hour_start']):
            current_day = hour.date()
            if prev_day is not None and current_day != prev_day:
                ax.axhline(idx - 0.5, color='red', linestyle='--', linewidth=1)
            prev_day = current_day

        ax.set_title(f"各地市气瓶流转操作量(小时)（第{i + 1}张）", fontproperties=zh_font, fontsize=12)
        ax.set_xlabel("操作量", fontproperties=zh_font)
        ax.set_ylabel("时间", fontproperties=zh_font)

        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(f"./gas_circulation/各地市气瓶流转操作量(小时)_第{i + 1}张.png", dpi=300)
        plt.close()

def plot_hourly_transition_bars_other(hourly_stats):
    """
    绘制全局每小时状态跳转数的横向柱状图，并标注每根柱子的数值。

    参数：
        hourly_stats: DataFrame，包含 ['hour_start', 'count_hourly']

    输出：
        横向柱状图保存为 "每小时操作量_1-4.png"
    """
    # 获取全部涉及的日期
    hourly_stats = extract_five_hours(hourly_stats)

    # 构造完整时间段列（从每日 00:00 到 05:00，每小时一次）
    min_time = hourly_stats["hour_start"].min().normalize()
    max_time = hourly_stats["hour_start"].max().normalize() + + pd.Timedelta(hours=5)

    full_index = pd.date_range(start=min_time, end=max_time, freq='H')
    full_index = full_index[full_index.hour.isin([0, 1, 2, 3, 4, 5])]

    # 构建 DataFrame
    full_df = pd.DataFrame({'hour_start': full_index})

    # 合并原数据（以 hour_start 为键），缺失 count_hourly 补 0
    hourly_stats = full_df.merge(hourly_stats, on='hour_start', how='left')
    hourly_stats['count_hourly'] = hourly_stats['count_hourly'].fillna(0).astype(int)
    hourly_stats['hour_stop'] = hourly_stats['hour_start'] + pd.Timedelta(hours=1)

    total = len(hourly_stats)
    chunk_size = math.ceil(total / 3)

    # 获取最大操作量，用于统一坐标轴
    max_value = hourly_stats["count_hourly"].max()
    x_limit = math.ceil(max_value * 1.1)  # 放大 10%，避免文字重叠

    for i in range(3):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk = hourly_stats.iloc[start:end]

        fig_height = max(6, len(chunk) * 0.3)
        plt.figure(figsize=(10, fig_height))
        ax = plt.gca()

        bars = ax.barh(
            chunk['hour_start'].astype(str),
            chunk['count_hourly'],
            color='steelblue'
        )

        # 设置统一 X 坐标轴范围
        ax.set_xlim(0, x_limit)

        # 计算统一右侧标注位置（最大值 + 偏移）
        max_width = hourly_stats['count_hourly'].max()
        label_x = max_width + 20  # 统一偏移

        for bar in bars:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(
                label_x,
                y,
                f"{int(width)}",
                va='center',
                ha='right',  # ← 右对齐
                fontsize=12,
                fontproperties=zh_font
            )

        # 添加红色虚线分隔：按日期
        prev_day = None
        for idx, hour in enumerate(chunk['hour_start']):
            current_day = hour.date()
            if prev_day is not None and current_day != prev_day:
                ax.axhline(idx - 0.5, color='red', linestyle='--', linewidth=1)
            prev_day = current_day

        # 标题与标签
        ax.set_title(f"各地市气瓶流转(含充装前至出站)操作量(小时)（第{i + 1}张）", fontproperties=zh_font, fontsize=12)
        ax.set_xlabel("操作量", fontproperties=zh_font)
        ax.set_ylabel("时间", fontproperties=zh_font)

        # y 轴字体
        ax.set_yticklabels(
            chunk['hour_start'].astype(str),
            fontproperties=zh_font,
            fontsize=12
        )

        # 美观处理
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(f"./gas_circulation/各地市气瓶流转(含充装前至出站)操作量(小时)（第{i + 1}张）.png", dpi=300)
        plt.close()
def plot_company_transition_heatmap_and_stacked(company_hourly_stats, company_mapping):
    """
    绘制公司-小时状态跳转热力图 和 堆叠柱状图（含中文公司名与时间格式化）
    特别规则：若 20:00~次日5:00 的小时段数值 >10，则数字标红

    参数：
        company_hourly_stats: DataFrame，包含 ['company_code', 'hour_start', '状态跳转数']
        company_mapping: DataFrame，包含 ['companyCode', 'company']
    输出：
        - 公司状态跳转_热力图.png
        - 公司状态跳转_堆叠柱状图.png
    """
    # 构造中文名映射
    company_dict = dict(zip(company_mapping['companyCode'], company_mapping['company']))

    # 提取每天0:00–6:00段数据
    company_hourly_stats = extract_five_hours(company_hourly_stats)

    # 构造完整时间段列（从每日 00:00 到 05:00，每小时一次）
    min_time = company_hourly_stats["hour_start"].min().normalize()
    max_time = company_hourly_stats["hour_start"].max().normalize() + pd.Timedelta(hours=5)

    full_times = pd.date_range(start=min_time, end=max_time, freq='H')
    full_times = full_times[full_times.hour.isin([0, 1, 2, 3, 4, 5])]

    # === 构造完整公司 × 时间的 MultiIndex ===
    full_index = pd.MultiIndex.from_product(
        [company_dict, full_times],
        names=["company_code", "hour_start"]
    )

    # 将原始数据设置为 MultiIndex 结构并重建 DataFrame
    company_hourly_stats = company_hourly_stats.set_index(["company_code", "hour_start"]).reindex(full_index).fillna(
        0).reset_index()

    # 构造透视表
    heatmap_data = company_hourly_stats.pivot_table(
        index='company_code', columns='hour_start', values='count_hourly', fill_value=0
    )
    heatmap_data.index = heatmap_data.index.to_series().replace(company_dict)

    # 格式化时间列
    heatmap_data.columns = pd.to_datetime(heatmap_data.columns)
    heatmap_data = heatmap_data.sort_index(axis=1)  # 按时间排序
    all_columns = heatmap_data.columns
    total_cols = len(all_columns)
    split_size = (total_cols + 2) // 3

    for i in range(3):
        sub_cols = all_columns[i * split_size: min((i + 1) * split_size, total_cols)]
        sub_data = heatmap_data[sub_cols]

        fig_width = max(12, len(sub_cols) * 0.3)
        fig_height = max(8, sub_data.shape[0] * 0.4)
        plt.figure(figsize=(fig_width, fig_height))

        ax = sns.heatmap(
            sub_data.astype(int),
            cmap="YlGnBu",
            annot=False,
            fmt='d',
            cbar=True,
            linewidths=0.3
        )

        # 标注数值（夜间红色）
        for y, company in enumerate(sub_data.index):
            for x, time_obj in enumerate(sub_data.columns):
                value = int(sub_data.iloc[y, x])
                hour = time_obj.hour
                is_night = (hour >= 20 or hour < 5)
                color = 'red' if is_night and value > 50 else 'black'
                ax.text(x + 0.5, y + 0.5, f"{value}",
                        ha='center', va='center',
                        fontsize=12, fontproperties=zh_font,
                        color=color)

        # 轴设置
        ax.set_title(f"各地市流转(全流程)操作统计量热力图（第{i + 1}张）", fontproperties=zh_font)
        ax.set_xlabel("时间", fontproperties=zh_font)
        ax.set_ylabel("企业", fontproperties=zh_font)
        ax.set_xticklabels([d.strftime('%m/%d %H:%M') for d in sub_data.columns],
                           fontproperties=zh_font, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./gas_circulation/各地市流转(全流程)操作统计量热力图_第{i + 1}张.png", dpi=300)
        plt.close()

def plot_company_transition_heatmap_and_stacked_specify_steps(company_hourly_stats, company_mapping):
    """
    绘制公司-小时状态跳转热力图 和 堆叠柱状图（含中文公司名与时间格式化）
    特别规则：若 20:00~次日5:00 的小时段数值 >10，则数字标红

    参数：
        company_hourly_stats: DataFrame，包含 ['company_code', 'hour_start', '状态跳转数']
        company_mapping: DataFrame，包含 ['companyCode', 'company']
    输出：
        - 公司状态跳转_热力图.png
        - 公司状态跳转_堆叠柱状图.png
    """
    # 构造中文名映射
    company_dict = dict(zip(company_mapping['companyCode'], company_mapping['company']))

    # 提取每天0:00–6:00段数据
    company_hourly_stats = extract_five_hours(company_hourly_stats)

    # 构造完整时间段列（从每日 00:00 到 05:00，每小时一次）
    min_time = company_hourly_stats["hour_start"].min().normalize()
    max_time = company_hourly_stats["hour_start"].max().normalize() + pd.Timedelta(hours=5)

    full_times = pd.date_range(start=min_time, end=max_time, freq='H')
    full_times = full_times[full_times.hour.isin([0, 1, 2, 3, 4, 5])]

    # === 构造完整公司 × 时间的 MultiIndex ===
    full_index = pd.MultiIndex.from_product(
        [company_dict, full_times],
        names=["company_code", "hour_start"]
    )

    # 将原始数据设置为 MultiIndex 结构并重建 DataFrame
    company_hourly_stats = company_hourly_stats.set_index(["company_code", "hour_start"]).reindex(full_index).fillna(
        0).reset_index()

    # 构造透视表
    heatmap_data = company_hourly_stats.pivot_table(
        index='company_code', columns='hour_start', values='count_hourly', fill_value=0
    )
    heatmap_data.index = heatmap_data.index.to_series().replace(company_dict)
    heatmap_data.columns = pd.to_datetime(heatmap_data.columns)
    heatmap_data = heatmap_data.sort_index(axis=1)  # 确保列时间顺序

    all_cols = heatmap_data.columns
    total_cols = len(all_cols)
    split_size = (total_cols + 2) // 3

    for i in range(3):
        sub_cols = all_cols[i * split_size: min((i + 1) * split_size, total_cols)]
        sub_data = heatmap_data[sub_cols]

        fig_width = max(12, len(sub_cols) * 0.35)
        fig_height = max(8, sub_data.shape[0] * 0.4)
        plt.figure(figsize=(fig_width, fig_height))

        ax = sns.heatmap(
            sub_data.astype(int),
            cmap="YlGnBu",
            annot=False,
            fmt='d',
            cbar=True,
            linewidths=0.3
        )

        # 仅标注高值：夜间（20:00~次日5:00）且值 > 50 红色，其余黑色
        for y, company in enumerate(sub_data.index):
            for x, time_obj in enumerate(sub_data.columns):
                value = int(sub_data.iloc[y, x])
                hour = time_obj.hour
                is_night = (hour >= 20 or hour < 5)
                color = 'red' if is_night and value > 50 else 'black'
                ax.text(x + 0.5, y + 0.5, f"{value}",
                        ha='center', va='center',
                        fontsize=12, fontproperties=zh_font,
                        color=color)

        # 添加红色虚线分隔：按日期
        prev_day = None
        for idx, hour in enumerate(company_hourly_stats['hour_start']):
            current_day = hour.date()
            if prev_day is not None and current_day != prev_day:
                ax.axvline(idx - 0.5, color='red', linestyle='--', linewidth=1)
            prev_day = current_day

        ax.set_title(f"各地市流转操作(含充装前至出站)统计量热力图（第{i + 1}张）", fontproperties=zh_font)
        ax.set_xlabel("时间", fontproperties=zh_font)
        ax.set_ylabel("企业", fontproperties=zh_font)
        ax.set_xticklabels([d.strftime('%m/%d %H:%M') for d in sub_data.columns],
                           fontproperties=zh_font, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=12)

        plt.tight_layout()
        plt.savefig(f"./gas_circulation/各地市流转操作(含充装前至出站)统计量热力图_第{i + 1}张.png", dpi=300)
        plt.close()


def plot_time_interval_statistics(df_transitions):
    """
        绘制状态跳转对 (from_status -> to_status) 的横向柱状图。

        参数：
            df_transitions: 包含 from_status 和 to_status 的 DataFrame
            output_path: 图片保存路径（默认为 '状态跳转统计图.png'）
        """
    if df_transitions.empty:
        return

    # 转换为字符串后进行映射
    df_transitions["from_status_str"] = df_transitions["from_status"].astype(str)
    df_transitions["to_status_str"] = df_transitions["to_status"].astype(str)

    # 构建状态跳转对标签
    df_transitions["transition_pair"] = df_transitions["from_status_str"].map(status_mapping) + " 至 " + \
                                        df_transitions["to_status_str"].map(status_mapping)

    # 统计每种跳转对的数量
    pair_counts = df_transitions["transition_pair"].value_counts()

    # 自定义顺序列表（根据业务流程顺序）
    ordered_pairs = []
    for i in range(1, 21):
        for j in range(1, 21):
            key = f"{status_mapping.get(str(i), str(i))} → {status_mapping.get(str(j), str(j))}"
            if key in pair_counts:
                ordered_pairs.append(key)

    # 构造排序后的 Series
    pair_counts = pair_counts[ordered_pairs]

    # 绘图
    plt.figure(figsize=(10, 8))
    bars = plt.barh(pair_counts.index, pair_counts.values, color="blue")

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 2, bar.get_y() + bar.get_height() / 2, str(int(width)),
                 va="center", fontsize=12, fontproperties=zh_font)

    # 图表美化
    plt.xlabel("异常次数", fontproperties=zh_font)
    plt.ylabel("异常操作", fontproperties=zh_font)
    plt.title("异常操作统计量", fontproperties=zh_font, fontsize=12)
    plt.xticks(fontproperties=zh_font)
    plt.yticks(fontproperties=zh_font)

    # 去掉顶部和右侧边框
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("./gas_circulation/异常操作统计量.png", dpi=300)
    plt.close()

def plot_abnormal_distribution(company_exception, employee_exception, df_cylinder, company_mapping, abnormal_distribution, txt_path_company="./gas_circulation/出站、配送未在同一天进行的统计明细_company.txt"):
    """
    绘制企业异常处理统计图（横向柱状图）、绘制员工异常处理统计图（横向柱状图）

    参数：
        company_exception: DataFrame，字段包含 ['company', 'abnormal_count']
        employee_exception: DataFrame，字段包含 ['name', 'company', 'abnormal_count']

    输出：
        分别保存为 ./gas_circulation/企业异常处理统计.png、./gas_circulation/员工异常处理统计.png
    """
    if company_exception.empty:
        return
    # 类型转换与数据清洗
    company_exception['abnormal_count'] = pd.to_numeric(company_exception['abnormal_count'], errors='coerce').fillna(0).astype(int)
    company_df = company_exception[company_exception['abnormal_count'] > 0].sort_values('abnormal_count', ascending=True)

    # 动态调整高度
    fig_height = max(6, len(company_df) * 0.35)
    plt.figure(figsize=(10, fig_height))
    ax = plt.gca()

    bars = ax.barh(company_df['company'], company_df['abnormal_count'], color='blue')

    # 数值标注
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                va='center', fontsize=12, fontproperties=zh_font)

    # 图表设置
    ax.set_title("企业异常处理次数", fontproperties=zh_font, fontsize=12)
    ax.set_xlabel("异常次数", fontproperties=zh_font)
    ax.set_ylabel("企业名称", fontproperties=zh_font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/企业异常处理统计.png", dpi=300)
    plt.close()

    abnormal_with_company = abnormal_distribution.merge(df_cylinder, left_on='gas_id', right_on='id', how='left')

    # 文本输出(详细)

    txt_lines = ["【各公司明细】\n\n"]
    grouped = abnormal_with_company.groupby(['companyCode'])

    for company_code, group in grouped:
        company_name = dict(zip(company_mapping['companyCode'], company_mapping['company'])).get(company_code[0],
                                                                                                 '未知公司')
        gas_ids = group['gas_id'].dropna().astype(str).tolist()
        line = f"{company_name} {len(gas_ids)} " + ", ".join(gas_ids)
        txt_lines.append(line)

    with open(txt_path_company, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    if employee_exception.empty:
        return
    # 类型转换与数据清洗
    employee_exception['abnormal_count'] = pd.to_numeric(employee_exception['abnormal_count'], errors='coerce').fillna(
        0).astype(int)
    employee_df = (
        employee_exception[employee_exception['abnormal_count'] > 0]
        .sort_values('abnormal_count', ascending=False)
        .head(20) )
    employee_df['label'] = employee_df['name'].fillna('未知员工') + '（' + employee_df['company'].fillna(
        '未知单位') + '）'

    # 动态调整宽度
    fig_width = max(6, len(employee_df) * 0.4)
    plt.figure(figsize=(fig_width, 6))
    ax = plt.gca()

    bars = ax.bar(employee_df['label'], employee_df['abnormal_count'], color='blue')

    # 数值标注（标注在柱体上方）
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=12,
            fontproperties=zh_font
        )

    # 图表设置
    ax.set_title("员工异常处理次数", fontproperties=zh_font, fontsize=9)
    ax.set_ylabel("异常次数", fontproperties=zh_font)
    ax.set_xlabel("员工（所属企业）", fontproperties=zh_font)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=zh_font, fontsize=9, rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/员工异常处理统计(隔天配送).png", dpi=300)
    plt.close()

def plot_inflator_bottle(df_cumulative, top_n):
    """
    绘制充气工分钟级持瓶数量折线图（默认显示前N位最大值持瓶人）

    参数：
        df_cumulative: DataFrame，index 为分钟，columns 为“充气工姓名（公司）”，值为持瓶数量
        top_n: int，绘制持瓶峰值最大的前N个员工
        save_path: str，保存路径，若为None则不保存

    输出：
        折线图显示/保存
    """
    if df_cumulative.empty:
        print("数据为空，无法绘图")
        return

    # 筛选持瓶最大值排名前 N 的充气工
    max_vals = df_cumulative.max().sort_values(ascending=False)
    top_columns = max_vals.head(top_n).index.tolist()
    df_plot = df_cumulative[top_columns]

    # 绘图
    plt.figure(figsize=(16, 6))
    for col in df_plot.columns:
        plt.plot(df_plot.index, df_plot[col], label=col)

    plt.title("充气工分钟级气瓶持有量趋势", fontproperties=zh_font, fontsize=12)
    plt.xlabel("时间", fontproperties=zh_font)
    plt.ylabel("持瓶数量", fontproperties=zh_font)
    plt.xticks(rotation=45)
    plt.legend(prop=zh_font, fontsize=12)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.tight_layout()
    plt.savefig("./gas_circulation/充气工持瓶状态.png", dpi=300)
    plt.close()

def plot_aspirator_bottle(df_cumulative, df_violation, top_n):
    """
    绘制充气工分钟级持瓶数量折线图（默认显示前N位最大值持瓶人）

    参数：
        df_cumulative: DataFrame，index 为分钟，columns 为“充气工姓名（公司）”，值为持瓶数量
        top_n: int，绘制持瓶峰值最大的前N个员工
        save_path: str，保存路径，若为None则不保存

    输出：
        折线图显示/保存
    """
    if df_cumulative.empty:
        return

    # 筛选持瓶最大值排名前 N 的充气工
    max_vals = df_cumulative.max().sort_values(ascending=False)
    top_columns = max_vals.head(top_n).index.tolist()
    df_plot = df_cumulative[top_columns]

    # 绘图
    plt.figure(figsize=(16, 6))
    for col in df_plot.columns:
        plt.plot(df_plot.index, df_plot[col], label=col)

    plt.title("送气工分钟级气瓶持有量趋势", fontproperties=zh_font, fontsize=12)
    plt.xlabel("时间", fontproperties=zh_font)
    plt.ylabel("持瓶数量", fontproperties=zh_font)
    plt.xticks(rotation=45)
    plt.legend(prop=zh_font, fontsize=12)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.tight_layout()
    plt.savefig("./gas_circulation/送气工持瓶状态.png", dpi=300)
    plt.close()

    if df_violation.empty:
        return

    # 取前 N 条
    df_plot = df_violation.sort_values(by="count", ascending=False).head(top_n)

    # 图像高度动态调整
    fig_height = max(4, len(df_plot) * 0.5)
    plt.figure(figsize=(12, fig_height))
    ax = sns.barplot(x="count", y="label", data=df_plot, palette="Reds_d")

    # 添加数值标注（紧贴柱体末端）
    for bar in ax.patches:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            width + 0.5, y,
            f'{int(width)}',
            va='center',
            ha='left',
            fontsize=12,
            fontproperties=zh_font
        )

    # 样式与标签美化
    plt.title("送气工违规操作次数 Top{}".format(top_n), fontproperties=zh_font, fontsize=12)
    plt.xlabel("违规次数", fontproperties=zh_font)
    plt.ylabel("送气工（所属企业）", fontproperties=zh_font)
    plt.xticks(fontproperties=zh_font)
    plt.yticks(fontproperties=zh_font, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("./gas_circulation/送气工违规行为统计.png", dpi=300)
    plt.close()

def plot_abnormal_flow(abnormal_flow, df_cylinder, company_mapping, save_dir="./gas_circulation", txt_path='./gas_circulation/'):
    """
    绘制横向柱状图，展示气瓶异常操作（报废、检验）情况。

    参数：
        abnormal_flow: dict，结构如 {'scrap': {...}, 'check': {...}}
        save_dir: 保存图片的目录路径
    """
    for flow_type in ['scrap', 'check']:
        data = pd.DataFrame(abnormal_flow[flow_type]['abnormal_df'], columns=['gas_id'])

        if data.empty:
            continue

        abnormal_with_company = data.merge(df_cylinder, left_on='gas_id', right_on='id', how='left')

        # Step 3：统计每个 company_code/company 的异常次数
        company_abnormal_count = (
            abnormal_with_company
            .groupby(['companyCode'])
            .size()
            .reset_index(name='abnormal_count')
            .sort_values('abnormal_count', ascending=False)
        )

        # Step 3：合并公司名称
        company_abnormal_count = company_abnormal_count[::-1].merge(company_mapping, on='companyCode', how='left')

        # Step 4：绘图（横向柱状图）
        fig_height = max(6, len(company_abnormal_count) * 0.35)
        plt.figure(figsize=(10, fig_height))
        ax = plt.gca()

        bars = ax.barh(company_abnormal_count['company'], company_abnormal_count['abnormal_count'], color='blue')

        # 数值标注
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                    va='center', fontsize=12, fontproperties=zh_font)

        # 图表设置
        title = "超报废日期操作气瓶数" if flow_type == 'scrap' else "超下次检验日期操作气瓶数"
        ax.set_title(f"各企业{title}", fontproperties=zh_font, fontsize=12)
        ax.set_xlabel("异常气瓶数量", fontproperties=zh_font)
        ax.set_ylabel("企业名称", fontproperties=zh_font)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{flow_type}_异常操作统计.png"), dpi=300)
        plt.close()

        # 文本输出(详细)
        txt_output_path = os.path.join(txt_path, f"{flow_type}_异常操作明细.txt")
        txt_lines = ["【各公司明细】\n\n"]
        grouped = abnormal_with_company.groupby(['companyCode'])

        for company_code, group in grouped:
            company_name = dict(zip(company_mapping['companyCode'], company_mapping['company'])).get(company_code[0],
                                                                                                     '未知公司')
            gas_ids = group['gas_id'].dropna().astype(str).tolist()
            line = f"{company_name} {len(gas_ids)} " + ", ".join(gas_ids)
            txt_lines.append(line)

        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))

def plot_abnormal_flow_other(abnormal_flow, df_cylinder, company_mapping, save_dir="./gas_circulation", txt_path='./gas_circulation/4、6、7、8在一小时内完成操作统计.txt'):
    """
    绘制横向柱状图，展示气瓶异常操作（报废、检验）情况。

    参数：
        abnormal_flow: dict，结构如 {'scrap': {...}, 'check': {...}}
        save_dir: 保存图片的目录路径
    """
    data = abnormal_flow['abnormal_df']
    abnormal_with_company = data.merge(df_cylinder, left_on='gas_id', right_on='id', how='left')

    # Step 3：统计每个 company_code/company 的异常次数
    company_abnormal_count = (
        abnormal_with_company
        .groupby(['companyCode'])
        .size()
        .reset_index(name='abnormal_count')
        .sort_values('abnormal_count', ascending=False)
    )

    # Step 3：合并公司名称
    company_abnormal_count = company_abnormal_count[::-1].merge(company_mapping, on='companyCode', how='left')

    # Step 4：绘图（横向柱状图）
    fig_height = max(6, len(company_abnormal_count) * 0.35)
    plt.figure(figsize=(10, fig_height))
    ax = plt.gca()

    bars = ax.barh(company_abnormal_count['company'], company_abnormal_count['abnormal_count'], color='blue')

    # 数值标注
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                va='center', fontsize=12, fontproperties=zh_font)

    # 图表设置
    ax.set_title(f"各企业4、6、7、8在一小时内完成操作统计", fontproperties=zh_font, fontsize=14)
    ax.set_xlabel("异常气瓶数量", fontproperties=zh_font)
    ax.set_ylabel("企业名称", fontproperties=zh_font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/4、6、7、8在一小时内完成操作统计_AAAAA.png", dpi=300)
    plt.close()

    # Step 5：输出 TXT 气瓶编号明细
    txt_lines = ["【各公司明细】\n\n"]
    grouped = abnormal_with_company.groupby(['companyCode'])

    for company_code, group in grouped:
        company_name = dict(zip(company_mapping['companyCode'], company_mapping['company'])).get(company_code[0],
                                                                                                 '未知公司')
        gas_ids = group['gas_id'].dropna().astype(str).tolist()
        line = f"{company_name} {len(gas_ids)} " + ", ".join(gas_ids)
        txt_lines.append(line)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

def plot_fake_data(company_exception, employee_exception, df_cylinder,  fake_data_cylinder, company_mapping, txt_path_company='./gas_circulation/企业上报数据异常明细.txt', txt_path_employee='./gas_circulation/员工上报数据异常明细.txt'):

    if company_exception.empty:
        return

    # 将企业信息合并到异常记录中
    abnormal_with_company = fake_data_cylinder.merge(df_cylinder, left_on='gas_id', right_on='id', how='left')

    # Step 3：统计每个 company_code/company 的异常次数
    company_abnormal_count = (
        abnormal_with_company
        .groupby(['companyCode'])
        .size()
        .reset_index(name='abnormal_count')
        .sort_values('abnormal_count', ascending=False)
    )

    # Step 3：合并公司名称
    company_abnormal_count = company_abnormal_count[::-1].merge(company_mapping, on='companyCode', how='left')

    # 类型转换与数据清洗
    company_exception['abnormal_count'] = pd.to_numeric(company_exception['abnormal_count'], errors='coerce').fillna(
        0).astype(int)
    company_df = company_exception[company_exception['abnormal_count'] > 0].sort_values('abnormal_count',
                                                                                        ascending=True)

    # Step 4：绘图（横向柱状图）
    fig_height = max(6, len(company_abnormal_count) * 0.35)
    plt.figure(figsize=(10, fig_height))
    ax = plt.gca()

    bars = ax.barh(company_abnormal_count['company'], company_abnormal_count['abnormal_count'], color='blue')

    # 数值标注
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                va='center', fontsize=12, fontproperties=zh_font)

    # 图表设置
    ax.set_title("企业上报数据异常次数", fontproperties=zh_font, fontsize=12)
    ax.set_xlabel("异常次数", fontproperties=zh_font)
    ax.set_ylabel("企业名称", fontproperties=zh_font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/企业上报数据异常次数.png", dpi=300)
    plt.close()

    # Step 5：输出 TXT 气瓶编号明细
    txt_lines = ["【各公司超期未配送气瓶数据明细】", "公司名称 超期数量 气瓶编号列表"]
    grouped = abnormal_with_company.groupby(['companyCode'])


    for company_code, group in grouped:
        company_name = dict(zip(company_mapping['companyCode'], company_mapping['company'])).get(company_code[0], '未知公司')
        gas_ids = group['gas_id'].dropna().astype(str).tolist()
        line = f"{company_name} {len(gas_ids)} " + ", ".join(gas_ids)
        txt_lines.append(line)

    with open(txt_path_company, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))


    if employee_exception.empty:
        return
    # 类型转换与数据清洗
    employee_exception['abnormal_count'] = pd.to_numeric(
        employee_exception['abnormal_count'], errors='coerce'
    ).fillna(0).astype(int)

    df_employee = employee_exception.sort_values(by="abnormal_count", ascending=False).head(20).sort_values('abnormal_count',
                                                                                        ascending=True)

    df_employee['label'] = df_employee['name'].fillna('未知员工') + '（' + employee_exception['company'].fillna('未知单位') + '）'

    # 动态调整高度
    fig_height = max(6, len(df_employee) * 0.35)
    plt.figure(figsize=(10, fig_height))
    ax = plt.gca()

    bars = ax.barh(df_employee['label'], df_employee['abnormal_count'], color='blue')

    # 数值标注
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                va='center', fontsize=12, fontproperties=zh_font)

    # 图表设置
    ax.set_title(f"员工上报数据异常次数 Top20", fontproperties=zh_font, fontsize=14)
    ax.set_xlabel("异常次数", fontproperties=zh_font)
    ax.set_ylabel("员工（所属企业）", fontproperties=zh_font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/员工上报数据异常次数.png", dpi=300)
    plt.close()

    # Step 5：输出 TXT 气瓶编号明细
    txt_lines = ["【各公司超期未配送气瓶数据明细】", "姓名（所属企业） 超期数量 气瓶编号列表"]
    grouped = abnormal_with_company.groupby(['companyCode'])


    for company_code, group in grouped:
        company_name = dict(zip(company_mapping['companyCode'], company_mapping['company'])).get(company_code[0], '未知公司')
        gas_ids = group['gas_id'].dropna().astype(str).tolist()
        line = f"{company_name} {len(gas_ids)} " + ", ".join(gas_ids)
        txt_lines.append(line)

    with open(txt_path_company, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

def plot_no_delivered_specified_time(df, df_cylinder, company_mapping, txt_path='./gas_circulation/超时未配送的气瓶详细记录.txt'):
    if df.empty:
        return pd.DataFrame()

    # Step 2：将企业信息合并到异常记录中
    abnormal_with_company = df.merge(df_cylinder, left_on='gas_id', right_on='id', how='left')

    # Step 3：统计每个 company_code/company 的异常次数
    company_abnormal_count = (
        abnormal_with_company
        .groupby(['companyCode'])
        .size()
        .reset_index(name='abnormal_count')
        .sort_values('abnormal_count', ascending=False)
    )

    # Step 3：合并公司名称
    company_abnormal_count = company_abnormal_count[::-1].merge(company_mapping, on='companyCode', how='left')

    # Step 4：绘图（横向柱状图）
    fig_height = max(6, len(company_abnormal_count) * 0.35)
    plt.figure(figsize=(10, fig_height))
    ax = plt.gca()

    bars = ax.barh(company_abnormal_count['company'], company_abnormal_count['abnormal_count'], color='blue')

    # 数值标注
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, str(int(width)),
                va='center', fontsize=12, fontproperties=zh_font)

    # 图表设置
    ax.set_title(f"各公司充装后超期未配送气瓶统计", fontproperties=zh_font, fontsize=14)
    ax.set_xlabel("超期未配送气瓶数量", fontproperties=zh_font)
    ax.set_ylabel("企业名称", fontproperties=zh_font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=zh_font, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig("./gas_circulation/超期未配送气瓶_企业柱状图.png", dpi=300)
    plt.close()

    # Step 5：输出 TXT 气瓶编号明细
    txt_lines = ["【各公司超期未配送气瓶数据明细】", "公司名称 超期数量 气瓶编号列表"]
    grouped = abnormal_with_company.groupby(['companyCode'])


    for company_code, group in grouped:
        company_name = dict(zip(company_mapping['companyCode'], company_mapping['company'])).get(company_code[0], '未知公司')
        gas_ids = group['gas_id'].dropna().astype(str).tolist()
        line = f"{company_name} {len(gas_ids)} " + ", ".join(gas_ids)
        txt_lines.append(line)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    return company_abnormal_count

def plot_user_recycled_long_time():
    pass