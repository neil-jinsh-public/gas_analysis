import pandas as pd
import pickle
from dateutil.relativedelta import relativedelta
from datetime import datetime
import database_model
from cylinder_model import count_cylinders_by_type, check_overdue_next_check_date, check_overdue_scrap_date
from cylinder_plot import plot_cylinder_type_by_company, plot_overdue_check, plot_overdue_scrap, plot_all_pie_charts
from circulation_model import (extract_status_paths, detect_abnormal_status_paths, extract_abnormal_cylinders_in_use,
                               count_status_transitions_per_hour, filter_illegal_transitions_by_day_cutoff,
                               build_transition_records_from_segments, apply_split_status_rules, detect_fast_transitions,
                               abnormal_distribution_gas_cylinders, statics_company_exception, statics_employee_exception,
                               extract_step2_step3_ownership, inflator_minute_level_ownership, extract_step6_step7_ownership,
                               aspirator_minute_level_ownership, querying_fake_data, abnormal_flow_cylinders, detect_fast_delivery,
                               statics_company_fake_data_exception, statics_employee_fake_data_exception)
from circulation_plot import (plot_abnormal_cylinders_in_use, plot_hourly_transition_bars, plot_company_transition_heatmap_and_stacked,
                              plot_time_interval_statistics, plot_company_transition_heatmap_and_stacked_specify_steps,
                              plot_hourly_transition_bars_other, plot_abnormal_distribution, plot_inflator_bottle,
                              plot_aspirator_bottle, plot_fake_distribution, plot_abnormal_flow, plot_abnormal_flow_other, plot_fake_data)
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    conn = database_model.db_conn()
    # 分析时间确定
    analysis_time = pd.to_datetime('2025-06-01 00:00:00')

    # 读取气瓶数据
    df_cylinder = database_model.get_gas_cylinders(conn)
    # 读取员工数据
    df_employee = database_model.get_employee(conn)
    # 读取气瓶流转状态数据
    # df_circulation = database_model.get_gas_status(conn)
    # 读取气瓶流转扫描日志数据
    df_scanning = database_model.get_gas_scanning(conn, analysis_time)
    # 读取瓶装燃气企业数据
    df_company = database_model.get_gas_company(conn)

    ### 以金华为统计对象进行气瓶底数统计
    # 企业id与企业name对应关系
    company_mapping = df_company[['code', 'companyName']].drop_duplicates().rename(columns={'code': 'companyCode', 'companyName': 'company'})

    # 各企业各类气瓶类型统计（0在用 1报废  2注销  3停用  4其它）
    gas_cylinder_type = count_cylinders_by_type(df_cylinder)
    plot_cylinder_type_by_company(df_cylinder, gas_cylinder_type, company_mapping)

    # 各场站在用状态下已超下次核验时间的气瓶统计(nextCheckDate 下次检验日期)
    df_total = df_cylinder[df_cylinder['status'] == 0].groupby('companyCode').agg(total_count=('id', 'count')).reset_index()
    df_check = check_overdue_next_check_date(df_cylinder, analysis_time)
    plot_overdue_check(df_check, df_total, company_mapping)

    # 各场站在用状态下已过超期报废时间的气瓶统计
    df_scrap = check_overdue_scrap_date(df_cylinder, analysis_time)
    plot_overdue_scrap(df_scrap, df_total, company_mapping)

    # 生成占比饼图
    plot_all_pie_charts(df_cylinder, gas_cylinder_type, analysis_time)

    # 获取在用状态下已超下次核验时间的气瓶编号以及在用状态下已过超期报废时间的气瓶编号,用于判断后续是否有进行流转
    list_abnormal_gas_cylinders = list(df_check['overdue_check_gasids']) + list(df_scrap['overdue_scrap_gasids'])
    # 整合为一个list
    flattened_list = set([item for sublist in list_abnormal_gas_cylinders for item in sublist])

    ########################################################################################################

    # 读取气瓶流转数据（状态数据 & 扫描日志数据）

    # 记录流转气瓶列表
    working_gas_list = set(df_scanning["gas_id"].to_list())

    # 获取仍在流转的异常气瓶记录以及详细流转信息
    abnormal_cylinders = extract_abnormal_cylinders_in_use(flattened_list, working_gas_list, df_scanning)
    plot_abnormal_cylinders_in_use(abnormal_cylinders, company_mapping)

    # 以小时统计状态跳转（即操作）数量
    hourly_statistics_all, hourly_statistics_company = count_status_transitions_per_hour(df_scanning)
    plot_hourly_transition_bars(hourly_statistics_all)
    plot_company_transition_heatmap_and_stacked(hourly_statistics_company, company_mapping)

    # 仅对1(充装前)/2(充装)/3(充装后)/4(出站)步骤进行统计
    filtered_df = df_scanning[df_scanning["flow"].isin(['1', '2', '3', '4'])]
    hourly_stats_1_4, hourly_stats_company_1_4 = count_status_transitions_per_hour(filtered_df)
    plot_hourly_transition_bars_other(hourly_stats_1_4)
    plot_company_transition_heatmap_and_stacked_specify_steps(hourly_stats_company_1_4, company_mapping)

    # 对不同阶段操作进行统计
    ###### pass

    # 提取气瓶状态路径
    status_path_df, status_path_dict = extract_status_paths(df_scanning)

    # 获取异常路径（指异常操作）记录
    abnormal_df = detect_abnormal_status_paths(status_path_df)

    # 切分出合规路径段（基于状态跳转合法性）【剔除异常路径（指异常操作）记录】
    normal_df = apply_split_status_rules(status_path_df)

    # 返回合法状态段的跳转对,并附带时间信息
    transitions_df = build_transition_records_from_segments(normal_df, status_path_df, analysis_time)

    # 正常路径时间间隔分析(因为上一步已经包含对最后一步完成时间的补充, 因此此处含最后一个步骤到当前时间的间隔判断)
    time_interval_statistics = filter_illegal_transitions_by_day_cutoff(transitions_df)
    plot_time_interval_statistics(time_interval_statistics)

    # 获取2(充装)/6(开始配送)/7(配送入户)步骤超过核验时间的气瓶统计
    df_cylinder['scrapDate'] = pd.to_datetime(df_cylinder['scrapDate'], errors='coerce')
    df_cylinder['nextCheckDate'] = pd.to_datetime(df_cylinder['nextCheckDate'], errors='coerce')
    # 筛选出包含充装、开始配送、配送入户的记录
    filtered_df = status_path_df[status_path_df['status_path'].apply(
        lambda path: '2' in path or '6' in path or '7' in path
    )]
    # 获取超过核验时间、超过报废时间的气瓶
    scrap_cylinders = df_cylinder[df_cylinder['scrapDate'] < pd.to_datetime(analysis_time + relativedelta(months=1))]
    check_cylinders = df_cylinder[df_cylinder['nextCheckDate'] < pd.to_datetime(analysis_time + relativedelta(months=1))]
    abnormal_flow = abnormal_flow_cylinders(df_cylinder, filtered_df, scrap_cylinders, check_cylinders, company_mapping)
    plot_abnormal_flow(abnormal_flow)

    # 对4(出站)、6(开始配送)、7(配送入户)、8（回收）四步1h内完成(此处同时完成暂定为1h)的流程进行筛选
    filtered_df = status_path_df[status_path_df['status_path'].apply(
        lambda path: '4' in path and '6' in path and '7' in path and '8' in path
    )]
    abnormal_flow = detect_fast_delivery(df_cylinder, filtered_df, company_mapping)
    plot_abnormal_flow_other(abnormal_flow)

    # 对于4(出站)、6(开始配送)后,在当天0点之前没有7(配送入户)且未(入站)的气瓶进行统计
    # 根据统计, 在7(配送入户)之前有{'1', '3', '4', '6', '7', '8', '9'}
    ##### filtered_df = df_scanning[df_scanning["flow"].isin(['7'])]
    # 筛选包含 '4' 和 '7' 的行, 且4在7前
    filtered_df = status_path_df[status_path_df['status_path'].apply(
        lambda path: '4' in path and '7' in path )]
    # 需要明确在7(配送入户)前的当天必须要有4(出站)状态, 否则为异常配送
    abnormal_distribution, fake_data = abnormal_distribution_gas_cylinders(filtered_df)

    # 统计企业异常次数、员工异常次数
    company_exception = statics_company_exception(abnormal_distribution, df_cylinder, company_mapping)
    employee_exception = statics_employee_exception(abnormal_distribution, df_employee, company_mapping)
    plot_abnormal_distribution(company_exception, employee_exception)

    # 送气工持瓶异常统计, 内部可配置同时持有气瓶的数量阈值
    # filtered_df = extract_step6_step7_ownership(status_path_df)
    # aspirator_bottle, df_violation = aspirator_minute_level_ownership(filtered_df, df_employee, company_mapping, threshold=12)
    # plot_aspirator_bottle(aspirator_bottle, df_violation, top_n=10)
    company_fake_data_exception = statics_company_fake_data_exception(fake_data, df_cylinder, company_mapping)
    employee_fake_data_exception = statics_employee_fake_data_exception(fake_data, df_employee, company_mapping)
    plot_fake_data(company_fake_data_exception, employee_fake_data_exception)

    print(f">>> {datetime.now()} Done")

