import pandas as pd
from datetime import timedelta
from collections import Counter
import ast
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 定义状态跳转合法性映射（来自图中规则）
valid_map = {
    '1': {'1', '2', '10'},
    '2': {'1', '2', '3', '10'},
    '3': {'3', '4', '5', '6', '10'},
    '4': {'3', '4', '5', '6', '9', '10'},
    '5': {'5', '8', '10'},
    '6': {'3', '4', '6', '7', '9', '10'},
    '7': {'7', '8'},
    '8': {'1', '8', '9'},
    '9': {'1', '4', '8', '9', '10'},
    '10': {'1', '9'}
}

# 定义状态跳转合法性映射（来自图中规则）
valid_map = {
    '1': {'2', '10'},
    '2': {'1', '3', '10'},
    '3': {'4', '5', '6', '10'},
    '4': {'3', '5', '6', '9', '10'},
    '5': {'8', '10'},
    '6': {'3', '4', '7', '9', '10'},
    '7': {'7', '8'},
    '8': {'1', '8', '9'},
    '9': {'1', '4', '8', '9', '10'},
    '10': {'1', '9'}
}

# 定义步骤间时间规则
day_limit_rules = {
    ('1', '2'): 0,  # 当天
    ('2', '3'): 0,  # 当天
    ('3', '4'): 1,  # 第二天
    ('4', '5'): 0,  # 当天
    ('5', '6'): 0,  # 当天
    ('6', '7'): 29,  # 暂定第30天
    ('7', '8'): 1,  # 第二天
    ('8', '9'): 0,  # 当天
    "default": None,  # 不限制
    ('4', '3'): 0,  # 当天
    # 同状态跳转默认当天
    ('1', '1'): None, ('2', 2): None, ('3', '3'): None, ('4', '4'): None, ('5', '5'): None,
    ('6', '6'): None, ('7', '7'): None, ('8', '8'): None, ('9', '9'): None, ('10', '10'): None,
}

def extract_status_paths(df):
    """
    从 DataFrame 中提取每个 gas_id 的状态迁移路径和时间序列（按 opTime 排序）
    返回：
        - status_path_df: DataFrame，包含 gas_id、status_path、time_path
        - status_path_dict: dict 格式，gas_id -> 状态序列
    """
    df_sorted = df.copy()
    df_sorted = df_sorted.sort_values(by=["gas_id", "opTime", "flow", "id"], ignore_index=True)
    df_sorted["opTime"] = pd.to_datetime(df_sorted["opTime"], unit='s', errors='coerce')

    grouped = df_sorted.groupby("gas_id").agg({
        "flow": list,
        "opTime": list,
        "employee_Id": list,
        "customer_id": list,
    }).reset_index()
    grouped.columns = ["gas_id", "status_path", "time_path", "employee_Id", "customer_id"]

    status_path_dict = dict(zip(grouped["gas_id"], grouped["status_path"]))

    return grouped, status_path_dict


def detect_abnormal_status_paths(status_path_df):
    """
    检查每个气瓶的状态路径中是否存在非法跳转，输出包含异常的 gas_id 及异常段。

    参数：
        status_path_df: 包含 ['gas_id', 'status_path'] 的 DataFrame，来自 extract_status_paths()

    返回：
        abnormal_df: 包含 gas_id、非法跳转列表、状态路径等字段的 DataFrame
    """
    records = []

    for _, row in status_path_df.iterrows():
        gas_id = row['gas_id']
        status_path = row['status_path']
        illegal_jumps = identify_illegal_jumps(status_path)

        if illegal_jumps:  # 若存在非法跳转
            records.append({
                'gas_id': gas_id,
                'status_path': status_path,
                'illegal_jumps': illegal_jumps,
                'is_abnormal': True
            })

    return pd.DataFrame(records)


def identify_illegal_jumps(status_path):
    """
    识别给定状态路径中是否存在非法跳转。

    参数：
        status_path: List[str]，表示一个气瓶的状态编号序列

    返回：
        List[Tuple[str, str]]，包含所有非法跳转对（from_status, to_status）
    """
    illegal_jumps = []

    for i in range(1, len(status_path)):
        prev = status_path[i - 1]
        curr = status_path[i]
        if curr not in valid_map.get(prev, set()):
            illegal_jumps.append((prev, curr))

    return illegal_jumps


def extract_abnormal_cylinders_in_use(abnormal_list, circulation_list, info_circulation):
    """
    返回两个列表的交集，并在info_circulation中查询这些交集元素对应的记录

    参数:
        abnormal_list: 包含异常气瓶ID的列表
        circulation_list: 包含流通中气瓶ID的列表
        info_circulation: 包含气瓶详细信息的字典，键为gas_id，值为气瓶信息

    返回:
        包含交集气瓶详细信息的列表
    """
    # 计算两个列表的交集（使用集合操作提高效率）
    intersection_ids = set(abnormal_list) & set(circulation_list)

    # 如果交集为空，直接返回空DataFrame
    if not intersection_ids:
        return pd.DataFrame(columns=info_circulation.columns)

    # 使用isin()方法筛选DataFrame
    result_df = info_circulation[info_circulation['gas_id'].isin(intersection_ids)]

    return result_df


def count_status_transitions_per_hour(df):
    """
    统计每小时的状态跳转数量 + 每个公司每小时跳转数量（含补零时间段）。

    参数：
        df: 原始气瓶状态数据，应包含字段 ['gas_id', 'flow', 'opTime', 'company_code']

    返回：
        - hourly_stats: DataFrame，包含每小时的跳转数量（总量）
        - company_hourly_stats: DataFrame，包含每公司每小时跳转数量
    """
    df_time = df.copy()
    df_time["opTime"] = pd.to_datetime(df_time["opTime"], unit='s', errors='coerce')
    df_time["hour_start"] = df_time["opTime"].dt.floor("h")
    df_time["hour_stop"] = df_time["hour_start"] + pd.Timedelta(hours=1)

    # 获取完整时间段范围
    time_range = pd.date_range(
        start=df_time["hour_start"].min(),
        end=df_time["hour_start"].max(),
        freq='H'
    )
    time_df = pd.DataFrame({
        "hour_start": time_range,
        "hour_stop": time_range + pd.Timedelta(hours=1)
    })

    # 1. 全局每小时跳转统计
    hourly_stats = df_time.groupby(["hour_start"]).size().reset_index(name="count_hourly")
    hourly_stats = time_df.merge(hourly_stats, on="hour_start", how="left").fillna(0)
    hourly_stats["count_hourly"] = hourly_stats["count_hourly"].astype(int)

    # 2. 每个公司每小时统计
    if "company_code" in df_time.columns:
        companies = df_time["company_code"].unique()
        company_time_df = pd.MultiIndex.from_product(
            [companies, time_range],
            names=["company_code", "hour_start"]
        ).to_frame(index=False)
        company_time_df["hour_stop"] = company_time_df["hour_start"] + pd.Timedelta(hours=1)

        company_hourly_stats = df_time.groupby(["company_code", "hour_start"]).size().reset_index(name="count_hourly")
        company_hourly_stats = company_time_df.merge(company_hourly_stats, on=["company_code", "hour_start"], how="left").fillna(0)
        company_hourly_stats["count_hourly"] = company_hourly_stats["count_hourly"].astype(int)
    else:
        company_hourly_stats = pd.DataFrame(columns=["company_code", "hour_start", "hour_stop", "count_hourly"])

    return hourly_stats, company_hourly_stats

def apply_split_status_rules(status_path_df):
    """
    对每条气瓶状态路径应用跳转规则，拆分为多个合法路径段。

    参数：
        status_path_df: DataFrame，包含 ['gas_id', 'status_path']，由 extract_status_paths 提供
        valid_map: 字典，合法状态跳转规则，如 {1: {2, 3}, 2: {3, 4}, ...}

    返回：
        DataFrame，包含：
            - gas_id: 气瓶唯一标识
            - segment_index: 第几段（从0开始）
            - segment_path: 合法的状态路径段（list）
    """
    rows = []

    for _, row in status_path_df.iterrows():
        gas_id = row["gas_id"]
        status_path = row["status_path"]

        # 拆分状态路径为多个合法段
        segments = []
        current = []

        for s in status_path:
            if not current:
                current.append(s)
            else:
                prev = current[-1]
                if s in valid_map.get(prev, set()):
                    current.append(s)
                else:
                    segments.append(current)
                    current = [s]

        if current:
            segments.append(current)

        # 记录每段
        for idx, seg in enumerate(segments):
            rows.append({
                "gas_id": gas_id,
                "segment_index": idx,
                "segment_path": seg
            })

    return pd.DataFrame(rows)


def build_transition_records_from_segments(segment_df, status_path_df, current_time):
    """
    构建合法状态段的跳转对，并附带时间信息。

    参数：
        segment_df: DataFrame，包含 gas_id, segment_index, segment_path（apply_split_status_rules 输出）
        status_path_df: DataFrame，包含 gas_id, status_path, time_path（extract_status_paths 输出）

    返回：
        transitions_df: DataFrame，包含每次跳转的详细信息，包括：
            - gas_id
            - from_status
            - to_status
            - from_time
            - to_time
    """
    if current_time is None:
        current_time = pd.Timestamp.now()

    # 构建 gas_id -> time_path 的映射
    time_path_dict = dict(zip(status_path_df["gas_id"], status_path_df["time_path"]))
    records = []

    for _, row in segment_df.iterrows():
        gas_id = row["gas_id"]
        segment = row["segment_path"]
        time_list = time_path_dict.get(gas_id, [])

        for i in range(1, len(segment)):
            try:
                records.append({
                    "gas_id": gas_id,
                    "from_status": segment[i - 1],
                    "to_status": segment[i],
                    "from_time": time_list[i - 1],
                    "to_time": time_list[i]
                })
            except IndexError:
                continue  # 安全处理异常

        # 添加最后一个状态到“当前时间”的跳转判断
        if len(segment) >= 1 and len(time_list) >= len(segment):
            records.append({
                "gas_id": gas_id,
                "from_status": segment[-1],
                "to_status": None,
                "from_time": time_list[-1],
                "to_time": current_time
            })

    return pd.DataFrame(records)

def filter_illegal_transitions_by_day_cutoff(df):
    """
    对每条状态跳转判断是否在规定日期内完成。
    参数：
        df: 包含 from_status, to_status, from_time, to_time
        day_limit_rules: 跳转对 -> 截止天数（0 表示当天 23:59:59）

    返回：
        不合法的跳转 DataFrame
    """
    illegal = []

    for _, row in df.iterrows():
        key = (row["from_status"], row["to_status"])
        day_limit = day_limit_rules.get(key, day_limit_rules.get("default"))

        if day_limit is None:
            continue  # 不限制

        # 截止时间 = from_time.date + 天数 + 23:59:59
        deadline = (row["from_time"].normalize() + pd.Timedelta(days=day_limit + 1)) - pd.Timedelta(seconds=1)

        if row["to_time"] > deadline:
            illegal.append(row)

    return pd.DataFrame(illegal)

def detect_fast_transitions(status_path_df, max_seconds=20, output_path="./gas_circulation/异常跳转(出站至配送入户)汇总.txt"):
    """
    检测同时包含 4→6→7 操作路径，且整体耗时小于 max_seconds 的异常记录。
    输出整合为一个 TXT 文件，含异常 gas_id 列表与对应详细记录。

    参数：
        status_path_df: DataFrame，包含 ['gas_id', 'status_path', 'time_path']
        max_seconds: 最大允许的状态 4→6→7 总耗时（单位：秒）
        output_path: 输出 TXT 文件路径

    返回：
        result_df: 所有跳转记录（含是否异常标记）
    """
    records = []
    abnormal_gas_ids = set()

    for _, row in status_path_df.iterrows():
        gas_id = row['gas_id']
        path = row['status_path']
        times = row['time_path']

        if not (set([4, 6, 7]).issubset(set(path))):
            continue

        try:
            idx_4 = next(i for i, s in enumerate(path) if s == 4)
            idx_6 = next(i for i in range(idx_4 + 1, len(path)) if path[i] == 6)
            idx_7 = next(i for i in range(idx_6 + 1, len(path)) if path[i] == 7)
        except StopIteration:
            continue

        t_start = times[idx_4]
        t_end = times[idx_7]

        if pd.isna(t_start) or pd.isna(t_end):
            continue

        total_time = (t_end - t_start).total_seconds()
        is_abnormal = total_time <= max_seconds

        if is_abnormal:
            abnormal_gas_ids.add(gas_id)

        for i in range(1, len(path)):
            from_status = path[i - 1]
            to_status = path[i]
            from_time = times[i - 1]
            to_time = times[i]

            if pd.isna(from_time) or pd.isna(to_time):
                continue

            time_diff_segment = (to_time - from_time).total_seconds()

            records.append({
                "gas_id": gas_id,
                "from_status": from_status,
                "to_status": to_status,
                "from_time": from_time,
                "to_time": to_time,
                "time_diff_sec": time_diff_segment,
                "is_abnormal": is_abnormal
            })

    result_df = pd.DataFrame(records)

    # ---- 写入合并 TXT 文件 ----
    if not abnormal_gas_ids:
        print("无异常记录")
        return result_df

    with open(output_path, "w", encoding="utf-8") as f:
        # 异常 gas_id 列表
        f.write("【异常气瓶 gas_id 列表】\n")
        for gid in sorted(abnormal_gas_ids):
            f.write(f"{gid}\n")

        f.write("\n【异常跳转明细】\n")
        result_df_abnormal = result_df[result_df['gas_id'].isin(abnormal_gas_ids)]

        for gid in sorted(abnormal_gas_ids):
            f.write(f"\n▶ 气瓶编号：{gid}\n")
            sub_df = result_df_abnormal[result_df_abnormal['gas_id'] == gid]
            for _, r in sub_df.iterrows():
                f.write(
                    f"  {r['from_status']} → {r['to_status']} | "
                    f"{r['from_time']} → {r['to_time']} | "
                    f"用时：{r['time_diff_sec']:.1f}s\n"
                )

    return result_df

def abnormal_distribution_gas_cylinders(df):
    # 存储异常记录
    abnormal_rows = []
    fake_data = []
    # 遍历每一行判断异常
    for idx, row in df.iterrows():
        status_path = row['status_path']
        time_path = row['time_path']
        employee = row['employee_Id']

        # 找所有的 4 和 7 的索引
        idx_7 = status_path.index('7')
        idx_4_candidates = [i for i in range(idx_7) if status_path[i] == '4']

        if not idx_4_candidates:
            continue  # 理论不会出现，因为前面已经筛选过

        idx_4 = idx_4_candidates[-1]  # 找最近的 '4'

        time_4 = pd.to_datetime(time_path[idx_4])
        time_7 = pd.to_datetime(time_path[idx_7])

        # 判断是否为同一天
        if time_4.date() != time_7.date():
            abnormal_rows.append({
                'gas_id': row['gas_id'],
                'status_path': status_path,
                'time_path': time_path,
                'step_4_time': time_4,
                'step_7_time': time_7,
                'employee': employee,
            })
        if time_4.date() == time_7.date():
            fake_data.append({
                'gas_id': row['gas_id'],
                'status_path': status_path,
                'time_path': time_path,
                'step_4_time': time_4,
                'step_7_time': time_7,
                'employee': employee,
            })

    # 构建异常 DataFrame
    return pd.DataFrame(abnormal_rows), pd.DataFrame(fake_data)

def statics_company_exception(df, df_cylinder, company_mapping):

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

    return company_abnormal_count.merge(company_mapping, on='companyCode', how='left')

def statics_employee_exception(df, df_employee, company_mapping):
    if df.empty:
        return pd.DataFrame()

    # 确保 status_path 和 employee 是列表格式
    df_filtered = df.copy()
    df_filtered['status_path'] = df_filtered['status_path'].apply(
        ast.literal_eval if isinstance(df_filtered['status_path'].iloc[0], str) else lambda x: x)
    df_filtered['employee'] = df_filtered['employee'].apply(ast.literal_eval if isinstance(df_filtered['employee'].iloc[0], str) else lambda x: x)

    # 提取每条记录中 step=7 的员工
    df_filtered['employee_7'] = df_filtered.apply(
        lambda row: row['employee'][row['status_path'].index('7')] if '7' in row['status_path'] else None,
        axis=1
    )

    # 删除没有7步骤的记录
    # df_filtered = df_filtered.dropna(subset=['employee_7'])

    # 统计员工异常次数, 合并员工基本信息/公司中文名
    employee_abnormal_count = (
        df_filtered['employee_7']
        .value_counts()
        .reset_index()
        .rename(columns={'count': 'abnormal_count', 'employee_7': 'employee_id'})
        .merge(df_employee[['id', 'name', 'companyCode']], left_on='employee_id', right_on='id', how='left')
        .merge(company_mapping[['companyCode', 'company']], on='companyCode', how='left')
    )

    return employee_abnormal_count

def statics_company_fake_data_exception(df, df_cylinder, company_mapping):

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

    return company_abnormal_count.merge(company_mapping, on='companyCode', how='left')

def statics_employee_fake_data_exception(df, df_employee, company_mapping):
    if df.empty:
        return pd.DataFrame()

    # 确保 status_path 和 employee 是列表格式
    df_filtered = df.copy()
    df_filtered['status_path'] = df_filtered['status_path'].apply(
        ast.literal_eval if isinstance(df_filtered['status_path'].iloc[0], str) else lambda x: x)
    df_filtered['employee'] = df_filtered['employee'].apply(
        ast.literal_eval if isinstance(df_filtered['employee'].iloc[0], str) else lambda x: x)

    # 提取每条记录中 step=7 的员工
    df_filtered['employee_7'] = df_filtered.apply(
        lambda row: row['employee'][row['status_path'].index('7')] if '7' in row['status_path'] else None,
        axis=1
    )

    # 删除没有7步骤的记录
    # df_filtered = df_filtered.dropna(subset=['employee_7'])

    # 统计员工异常次数, 合并员工基本信息/公司中文名
    employee_abnormal_count = (
        df_filtered['employee_7']
        .value_counts()
        .reset_index()
        .rename(columns={'count': 'abnormal_count', 'employee_7': 'employee_id'})
        .merge(df_employee[['id', 'name', 'companyCode']], left_on='employee_id', right_on='id', how='left')
        .merge(company_mapping[['companyCode', 'company']], on='companyCode', how='left')
    )

    return employee_abnormal_count

def extract_step2_step3_ownership(df_raw,  default_step3_time='2025-07-01 00:00:00'):
    """
    从原始状态路径中提取step2到step3期间的气瓶持有记录。

    输入：
        df_raw: DataFrame，包含字段 ['gas_id', 'status_path', 'time_path', 'employee_Id']，其中后三项为 list 或 str(list)

    返回：
        DataFrame，包含 ['gas_id', 'employee_id', 'step2_time', 'step3_time']
    """
    df = df_raw.copy()

    # 若字段为字符串形式的列表，先转为列表
    for col in ['status_path', 'time_path', 'employee_Id']:
        if df[col].apply(lambda x: isinstance(x, str)).all():
            df[col] = df[col].apply(ast.literal_eval)

    records = []

    for _, row in df.iterrows():
        try:
            status_list = row['status_path']
            time_list = pd.to_datetime(row['time_path'])
            emp_list = row['employee_Id']

            if '2' in status_list:
                idx2 = status_list.index('2')
                step2_time = time_list[idx2]
                emp = emp_list[idx2]

                if '3' in status_list:
                    idx3 = status_list.index('3')
                    if idx3 > idx2:
                        step3_time = time_list[idx3]
                    else:
                        continue  # 若3在2之前，不合理，跳过
                elif default_step3_time:
                    step3_time = pd.to_datetime(default_step3_time)
                else:
                    continue  # 没有step3也没有默认值，则跳过

                records.append({
                    'gas_id': row['gas_id'],
                    'employee_id': emp,
                    'step2_time': step2_time,
                    'step3_time': step3_time
                })
        except Exception:
            continue

    return pd.DataFrame(records)

def inflator_minute_level_ownership(df, df_employee, company_mapping, threshold=20):
    if df.empty:
        return pd.DataFrame()
        # 确保 status_path 和 employee 是列表格式

    """
    分钟级别统计每个充气工的气瓶持有数量。
    返回：DataFrame，索引为时间，列为充气工，值为其持瓶数量
    """
    df = df.copy()
    df['step2_time'] = pd.to_datetime(df['step2_time'])
    df['step3_time'] = pd.to_datetime(df['step3_time'])

    # +1 事件（step2），-1 事件（step3）
    df_plus = df[['employee_id', 'step2_time']].copy()
    df_plus['minute'] = df_plus['step2_time'].dt.floor('min')
    df_plus['delta'] = 1

    df_minus = df[['employee_id', 'step3_time']].copy()
    df_minus['minute'] = df_minus['step3_time'].dt.floor('min')
    df_minus['delta'] = -1

    df_events = pd.concat([
        df_plus[['employee_id', 'minute', 'delta']],
        df_minus[['employee_id', 'minute', 'delta']]
    ])

    # 按分钟聚合增量
    df_events = (
        df_events
        .groupby(['minute', 'employee_id'])['delta']
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    # 累加得到实际持瓶数
    df_cumulative = df_events.cumsum()

    # 过滤持瓶峰值超过阈值的员工
    col_max = np.asarray(df_cumulative.values).max(axis=0)
    keep_ids = df_cumulative.columns[col_max >= threshold]
    df_cumulative = df_cumulative[keep_ids]

    # 获取 id → 姓名、企业 映射
    emp_info = df_employee[['id', 'name', 'companyCode']].drop_duplicates()
    emp_info = emp_info.merge(company_mapping, on='companyCode', how='left')
    emp_info['label'] = emp_info['name'].fillna('未知') + '（' + emp_info['company'].fillna('未知企业') + '）'
    emp_label_map = dict(zip(emp_info['id'], emp_info['label']))

    # 替换列名为 姓名（企业）
    df_cumulative.columns = [emp_label_map.get(emp_id, emp_id) for emp_id in df_cumulative.columns]

    # ========== 记录每个超标充气工的变动节点 ==========
    txt_lines = ["【充气工持瓶数变动节点记录 - 阈值={}】\n".format(threshold)]

    for emp_id in keep_ids:
        emp_label = emp_label_map.get(emp_id, emp_id)
        series = df_cumulative[emp_label]

        # 识别持瓶数量变化的节点（差分不为0的时刻）
        diff_series = series.diff().fillna(series)
        change_points = diff_series[diff_series != 0]

        txt_lines.append(f"\n【{emp_label}】")
        for t, delta in change_points.items():
            current_count = series.loc[t]
            delta_sign = '+' if delta > 0 else ''
            txt_lines.append(f"{t} → {delta_sign}{int(delta)} → 当前持瓶：{int(current_count)}")

    # 保存为 txt 文件
    with open("./gas_circulation/充气工持瓶数变化节点记录.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    return df_cumulative

def extract_step6_step7_ownership(df_raw,  default_step7_time='2025-07-01 00:00:00'):
    """
    从原始状态路径中提取step6到step7期间的气瓶持有记录。

    输入：
        df_raw: DataFrame，包含字段 ['gas_id', 'status_path', 'time_path', 'employee_Id']，其中后三项为 list 或 str(list)

    返回：
        DataFrame，包含 ['gas_id', 'employee_id', 'step2_time', 'step3_time']
    """
    df = df_raw.copy()

    # 若字段为字符串形式的列表，先转为列表
    for col in ['status_path', 'time_path', 'employee_Id']:
        if df[col].apply(lambda x: isinstance(x, str)).all():
            df[col] = df[col].apply(ast.literal_eval)

    records = []

    for _, row in df.iterrows():
        try:
            status_list = row['status_path']
            time_list = pd.to_datetime(row['time_path'])
            emp_list = row['employee_Id']

            if '6' in status_list:
                idx6 = status_list.index('6')
                step6_time = time_list[idx6]
                emp = emp_list[idx6]

                if '7' in status_list:
                    idx7 = status_list.index('7')
                    if idx7 > idx6:
                        step7_time = time_list[idx7]
                    else:
                        continue  # 若7在6之前，不合理，跳过
                elif default_step7_time:
                    step7_time = pd.to_datetime(default_step7_time)
                else:
                    continue  # 没有step7也没有默认值，则跳过

                records.append({
                    'gas_id': row['gas_id'],
                    'employee_id': emp,
                    'step6_time': step6_time,
                    'step7_time': step7_time
                })
        except Exception:
            continue

    return pd.DataFrame(records)

def aspirator_minute_level_ownership(df, df_employee, company_mapping, threshold):
    if df.empty:
        return pd.DataFrame()
        # 确保 status_path 和 employee 是列表格式

    """
    分钟级别统计每个送气工的气瓶持有数量。
    返回：DataFrame，索引为时间，列为送气工，值为其持瓶数量
    """
    df = df.copy()
    df['step6_time'] = pd.to_datetime(df['step6_time'])
    df['step7_time'] = pd.to_datetime(df['step7_time'])

    # ==================== 异常行为统计（step6和step7在同一分钟） ====================
    # 根据实际查看数据得知, 如果出现一分钟内计算值为0, 说明在同一分钟内发生了6、7两步
    df['step6_min'] = df['step6_time'].dt.floor('min')
    df['step7_min'] = df['step7_time'].dt.floor('min')
    df['same_minute'] = df['step6_min'] == df['step7_min']

    # 统计每个 employee_id 的违规次数
    df_abnormal = df[df['same_minute']]
    abnormal_counts = df_abnormal['employee_id'].value_counts().rename("count").reset_index()
    abnormal_counts.columns = ['employee_id', 'count']

    # 映射姓名与公司
    emp_info = df_employee[['id', 'name', 'companyCode']].drop_duplicates()
    emp_info = emp_info.merge(company_mapping, on='companyCode', how='left')
    emp_info['label'] = emp_info['name'].fillna('未知') + '（' + emp_info['company'].fillna('未知企业') + '）'

    df_violation = abnormal_counts.merge(emp_info, left_on='employee_id', right_on='id', how='left')
    df_violation = df_violation[['employee_id', 'label', 'count']].sort_values(by='count', ascending=False)

    # ==================== 持瓶统计流程 ====================
    # +1 事件（step2），-1 事件（step3）
    df_plus = df[['employee_id', 'step6_time']].copy()
    df_plus['minute'] = df_plus['step6_time'].dt.floor('min')
    df_plus['delta'] = 1

    df_minus = df[['employee_id', 'step7_time']].copy()
    df_minus['minute'] = df_minus['step7_time'].dt.floor('min')
    df_minus['delta'] = -1

    df_events = pd.concat([
        df_plus[['employee_id', 'minute', 'delta']],
        df_minus[['employee_id', 'minute', 'delta']]
    ])

    # 按分钟聚合增量
    df_events = (
        df_events
        .groupby(['minute', 'employee_id'])['delta']
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    # 累加得到实际持瓶数
    df_cumulative = df_events.cumsum()

    # 过滤持瓶峰值超过阈值的员工
    col_max = np.asarray(df_cumulative.values).max(axis=0)
    keep_ids = df_cumulative.columns[col_max >= threshold]
    df_cumulative = df_cumulative[keep_ids]

    # 获取 id → 姓名、企业 映射
    emp_info = df_employee[['id', 'name', 'companyCode']].drop_duplicates()
    emp_info = emp_info.merge(company_mapping, on='companyCode', how='left')
    emp_info['label'] = emp_info['name'].fillna('未知') + '（' + emp_info['company'].fillna('未知企业') + '）'
    emp_label_map = dict(zip(emp_info['id'], emp_info['label']))

    # 替换列名为 姓名（企业）
    df_cumulative.columns = [emp_label_map.get(emp_id, emp_id) for emp_id in df_cumulative.columns]

    # ========== 记录每个超标送气工的变动节点 ==========
    txt_lines = ["【送气工持瓶数变动节点记录 - 阈值={}】\n".format(threshold)]

    for emp_id in keep_ids:
        emp_label = emp_label_map.get(emp_id, emp_id)
        series = df_cumulative[emp_label]

        # 识别持瓶数量变化的节点（差分不为0的时刻）
        diff_series = series.diff().fillna(series)
        change_points = diff_series[diff_series != 0]

        txt_lines.append(f"\n【{emp_label}】")
        for t, delta in change_points.items():
            current_count = series.loc[t]
            delta_sign = '+' if delta > 0 else ''
            txt_lines.append(f"{t} → {delta_sign}{int(delta)} → 当前持瓶：{int(current_count)}")

    # 保存为 txt 文件
    with open("./gas_circulation/送气工持瓶数变化节点记录.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    return df_cumulative, df_violation

def querying_fake_data():
    pass



def abnormal_flow_cylinders(df_cylinder, filtered_df, scrap_cylinders, check_cylinders, company_mapping):
    """
    检查气瓶在报废/下次检验日期后是否仍被执行操作（步骤2/6/7），并统计异常气瓶及对应企业。

    参数：
        df_cylinder: 包含 ['id', 'companyCode']
        filtered_df: 包含 ['gas_id', 'status_path', 'time_path']
        scrap_cylinders: 包含 ['id', 'scrapDate']
        check_cylinders: 包含 ['id', 'nextCheckDate']
        company_mapping: 包含 ['companyCode', 'company']

    返回：
        result_dict: {
            'scrap': {'scrap_cylinders': list, 'abnormal_count': int, 'company_stats': dict},
            'check': {'check_cylinders': list, 'abnormal_count': int, 'company_stats': dict}
        }
    """
    # 准备目标步骤集合
    target_steps = {'2', '6', '7'}

    # ========== 第一步：提前合并企业信息 ==========
    # 1. 关联 companyCode
    filtered_df = filtered_df.merge(
        df_cylinder[['id', 'companyCode']],
        left_on='gas_id', right_on='id', how='left'
    )

    # 2. 关联公司名称
    filtered_df = filtered_df.merge(
        company_mapping, on='companyCode', how='left'
    )
    filtered_df['company'] = filtered_df['company'].fillna('未知企业')

    # ========== 第二步：映射 scrapDate 和 nextCheckDate ==========
    scrap_date_dict = scrap_cylinders.set_index('id')['scrapDate'].to_dict()
    check_date_dict = check_cylinders.set_index('id')['nextCheckDate'].to_dict()

    filtered_df['scrapDate'] = filtered_df['gas_id'].map(scrap_date_dict)
    filtered_df['nextCheckDate'] = filtered_df['gas_id'].map(check_date_dict)

    # ========== 第三步：初始化结果记录 ==========
    scrap_abnormal = []
    scrap_company_counter = defaultdict(int)

    check_abnormal = []
    check_company_counter = defaultdict(int)

    # ========== 第四步：逐条判断 ==========
    for _, row in filtered_df.iterrows():
        gas_id = row['gas_id']
        company = row['company']

        try:
            status_list = eval(row['status_path']) if isinstance(row['status_path'], str) else row['status_path']
        except Exception:
            continue

        time_list = row['time_path']
        if not isinstance(time_list, list) or len(status_list) != len(time_list):
            continue

        # 检查报废
        scrap_time = pd.to_datetime(row['scrapDate'], errors='coerce')
        if pd.notna(scrap_time):
            for status, t in zip(status_list, time_list):
                if status in target_steps and pd.to_datetime(t) > scrap_time:
                    scrap_abnormal.append(gas_id)
                    scrap_company_counter[company] += 1
                    break

        # 检查检验
        check_time = pd.to_datetime(row['nextCheckDate'], errors='coerce')
        if pd.notna(check_time):
            for status, t in zip(status_list, time_list):
                if status in target_steps and pd.to_datetime(t) > check_time:
                    check_abnormal.append(gas_id)
                    check_company_counter[company] += 1
                    break

    # ========== 第五步：结果汇总 ==========
    return {
        'scrap': {
            'scrap_cylinders': scrap_abnormal,
            'abnormal_count': len(scrap_abnormal),
            'company_stats': dict(scrap_company_counter)
        },
        'check': {
            'check_cylinders': check_abnormal,
            'abnormal_count': len(check_abnormal),
            'company_stats': dict(check_company_counter)
        }
    }


def detect_fast_delivery(df_cylinder, filtered_df, company_mapping, max_hours=1):
    """
    判断气瓶操作流程中 '4-6-7-8' 四个步骤是否在 max_hours 小时内完成，若是则标为异常。

    参数：
        filtered_df: DataFrame，包含 'gas_id', 'status_path', 'time_path'
        df_cylinder: 气瓶主表，包含 'id' 和 'companyCode'
        company_mapping: 企业映射表，包含 'companyCode' 和 'company'
        max_hours: float，最大允许的完成时间（单位：小时）

    返回：
        result_dict = {
            'abnormal_df': 异常记录DataFrame,
            'abnormal_count': 异常气瓶数量,
            'company_stats': 各企业异常数量统计字典
        }
    """
    abnormal_rows = []
    company_counter = defaultdict(int)

    for idx, row in filtered_df.iterrows():
        gas_id = row['gas_id']
        status_path = eval(row['status_path']) if isinstance(row['status_path'], str) else row['status_path']
        time_path = row['time_path']

        if not isinstance(time_path, list) or len(status_path) != len(time_path):
            continue

        # 提取步骤 4,6,7,8 的时间
        steps = {'4': None, '6': None, '7': None, '8': None}
        for status, time in zip(status_path, time_path):
            if status in steps:
                steps[status] = pd.to_datetime(time, errors='coerce')

        # 若四步都存在，判断耗时
        if all(steps.values()):
            min_time = min(steps.values())
            max_time = max(steps.values())
            duration_hours = (max_time - min_time).total_seconds() / 3600

            if duration_hours <= max_hours:
                abnormal_rows.append(row)

                # 获取企业名称
                company_code_series = df_cylinder[df_cylinder['id'] == gas_id]['companyCode']
                if company_code_series.empty:
                    company = "未知企业"
                else:
                    code = company_code_series.iloc[0]
                    company_match = company_mapping[company_mapping['companyCode'] == code]['company']
                    company = company_match.iloc[0] if not company_match.empty else "未知企业"

                company_counter[company] += 1

    abnormal_df = pd.DataFrame(abnormal_rows)
    return {
        'abnormal_df': abnormal_df,
        'abnormal_count': len(abnormal_df),
        'company_stats': dict(company_counter)
    }