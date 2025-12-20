import os
import json
import pandas as pd
import glob
import re
import sys
import shutil
import numpy as np

# 设置控制台输出编码
sys.stdout.reconfigure(encoding='utf-8')

# ================= 配置区域 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(BASE_DIR, '1_Data')
WEB_TEMPLATE_DIR = os.path.join(BASE_DIR, '3_Web')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Report_Output')

# 原始数据源
SOURCE_RAW_DIR = r"D:\Onedrive\OFFICEVBA测试文档\002_WeChat_Automation\001A_DailyData_Tracking\01_raw_data"
SOURCE_SENTIMENT_PATH = os.path.join(SOURCE_RAW_DIR, "涨跌停数据.xlsx")
SOURCE_INDEX_RECENT = os.path.join(SOURCE_RAW_DIR, "指数跟踪第五版-提取表.xlsx")
SOURCE_INDEX_EARLY = os.path.join(SOURCE_RAW_DIR, "指数跟踪第五版 - 前期表.xlsx")

# 文件名常量
FILE_PCT = 'Pctchg_连续表.xlsx'
FILE_AMT = 'Amount_连续表.xlsx'
FILE_PRICE = 'Price_连续表.xlsx'
FILE_IND = '沪深A股-申万行业-20251216.xlsx'

# ================= 通用工具 =================
def safe_copy(source, dest):
    """安全复制文件，处理不存在或占用的情况"""
    try:
        shutil.copy2(source, dest)
        print(f"  - [复制成功] {os.path.basename(source)}")
        return True
    except FileNotFoundError:
        print(f"  - [忽略] 源文件不存在: {source}")
        return False
    except Exception as e:
        print(f"  - [警告] 复制失败: {e}")
        return False

def extract_date_from_filename(filename):
    match = re.search(r'(\d{4}[-年/]?\d{2}[-月/]?\d{2})', filename)
    if match: return match.group(1).replace('年','-').replace('月','-').replace('/','-')
    return "未知日期"

def read_file_smart(filepath):
    try:
        if filepath.endswith('.csv'): return pd.read_csv(filepath)
        return pd.read_excel(filepath)
    except: return None

# ================= 核心：统一数据加载 =================
def clean_numeric_series(series):
    """强制清洗数字列（处理千分位逗号、特殊字符）"""
    if series.dtype == 'object':
        return pd.to_numeric(series.astype(str).str.replace(r'[,\$￥]', '', regex=True), errors='coerce').fillna(0)
    return series.fillna(0)

def read_wide_table(path):
    """读取宽表通用逻辑"""
    if not os.path.exists(path): return None
    try:
        print(f"    - Reading {os.path.basename(path)}...")
        df = pd.read_excel(path)
        # 假设第1列是代码
        df = df.set_index(df.columns[0]) 
        
        # 识别日期列
        date_cols = []
        valid_cols = []
        for col in df.columns:
            try:
                d = pd.to_datetime(col)
                date_cols.append(d)
                valid_cols.append(col)
            except: pass
        
        df_final = df[valid_cols]
        df_final.columns = date_cols
        
        # 强制转为数字，处理空值
        df_final = df_final.apply(pd.to_numeric, errors='coerce').fillna(0)
        df_final.sort_index(axis=1, inplace=True)
        return df_final
    except Exception as e:
        print(f"    [Error] {e}")
        return None

def load_all_common_data():
    """【单次加载】读取所有公共大表"""
    print("=== [Stage 1] Loading Common Data ===")
    
    dest_folder = os.path.join(DATA_DIR, '000_common_data')
    if not os.path.exists(dest_folder): os.makedirs(dest_folder)
    
    # 1. 搬运文件
    for f in [FILE_PCT, FILE_AMT, FILE_PRICE, FILE_IND]:
        src = os.path.join(SOURCE_RAW_DIR, f)
        dst = os.path.join(dest_folder, f)
        if os.path.exists(src): safe_copy(src, dst)
    
    data_bundle = {'pct': None, 'amt': None, 'ind': None, 'dates': []}
    
    # 2. 读取数据
    # Pctchg
    path_pct = os.path.join(dest_folder, FILE_PCT)
    df_pct = read_wide_table(path_pct)
    if df_pct is not None:
        # 停牌修正
        df_pct[df_pct == -100] = 0
        data_bundle['pct'] = df_pct
        
    # Amount
    path_amt = os.path.join(dest_folder, FILE_AMT)
    data_bundle['amt'] = read_wide_table(path_amt)
    
    # Industry (申万) - 增加清洗逻辑
    ind_path = os.path.join(dest_folder, FILE_IND)
    if os.path.exists(ind_path):
        print(f"    - Reading {FILE_IND}...")
        try:
            df_ind = pd.read_excel(ind_path)
            # 代码列去重
            df_ind.drop_duplicates(subset=[df_ind.columns[0]], inplace=True)
            df_ind.set_index(df_ind.columns[0], inplace=True)
            
            # 【关键修复】确保市值列(第6列, index=5)是数字
            mv_col_name = df_ind.columns[5] 
            df_ind[mv_col_name] = clean_numeric_series(df_ind[mv_col_name])
            
            data_bundle['ind'] = df_ind
        except Exception as e: 
            print(f"    [Error] Industry table: {e}")
    
    # 3. 对齐日期
    if data_bundle['pct'] is not None and data_bundle['amt'] is not None:
        common = data_bundle['pct'].columns.intersection(data_bundle['amt'].columns)
        if len(common) > 250: common = common[-250:] # 最近250天
        
        data_bundle['dates'] = common
        data_bundle['pct'] = data_bundle['pct'][common]
        data_bundle['amt'] = data_bundle['amt'][common]
        print(f"  - Common Dates: {len(common)} days")
        
    return data_bundle

# ================= 业务模块 =================

def process_weekly_data():
    print("=== [Stage 2] Weekly Hotspots ===")
    database = {}
    target_folder = os.path.join(DATA_DIR, '001_热点股票排名')
    if os.path.exists(target_folder):
        group_data = []
        for file_path in glob.glob(os.path.join(target_folder, "*.*")):
            if os.path.basename(file_path).startswith('~$'): continue
            if not (file_path.endswith('.csv') or file_path.endswith('.xlsx')): continue
            df = read_file_smart(file_path)
            if df is not None:
                df = df.where(pd.notnull(df), "")
                group_data.append({
                    "fileName": os.path.basename(file_path),
                    "date": extract_date_from_filename(os.path.basename(file_path)),
                    "records": df.to_dict(orient='records')
                })
        database['001_热点股票排名'] = group_data
    
    tpl = os.path.join(WEB_TEMPLATE_DIR, 'template.html')
    out = os.path.join(OUTPUT_DIR, 'market_data.html')
    if os.path.exists(tpl):
        with open(tpl, 'r', encoding='utf-8') as f: html = f.read()
        with open(out, 'w', encoding='utf-8') as f: 
            f.write(html.replace('{{DATA_INJECTION}}', json.dumps(database, ensure_ascii=False)))

def process_sentiment_data(bundle):
    print("=== [Stage 3] Market Sentiment ===")
    df_pct = bundle['pct']
    df_amt = bundle['amt']
    dates = bundle['dates']
    if df_pct is None: return

    dest_s = os.path.join(DATA_DIR, '002_市场情绪跟踪')
    if not os.path.exists(dest_s): os.makedirs(dest_s)
    safe_copy(SOURCE_SENTIMENT_PATH, os.path.join(dest_s, '涨跌停数据.xlsx'))
    
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]
    
    # 1. 涨跌停
    limit_data = {}
    try:
        df_l = pd.read_excel(os.path.join(dest_s, '涨跌停数据.xlsx'))
        df_l['日期'] = pd.to_datetime(df_l['日期'])
        df_l.sort_values('日期', inplace=True)
        limit_data = {
            "dates": df_l['日期'].dt.strftime('%Y-%m-%d').tolist(),
            "volume": df_l['成交额'].tolist(),
            "up_counts": df_l['涨停家数'].tolist(),
            "down_counts": df_l['跌停家数'].tolist()
        }
    except: pass

    # 2. 涨跌分布
    c_up = (df_pct > 0).sum().tolist()
    c_down = (df_pct < 0).sum().tolist()
    c_total = df_pct.notnull().sum().tolist()
    c_flat = [t - u - d for t,u,d in zip(c_total, c_up, c_down)]
    r_up = [(u/t*100 if t else 0) for u,t in zip(c_up, c_total)]
    r_down = [(d/t*100 if t else 0) for d,t in zip(c_down, c_total)]

    # 3. 前日大涨大跌
    res_move = {"dates": date_strs[1:], "up_next": [], "down_next": [], "amt": []}
    for i in range(1, len(dates)):
        curr, prev = dates[i], dates[i-1]
        mask_up = df_pct[prev] > 8
        mask_down = df_pct[prev] < -8
        
        res_move['up_next'].append(df_pct.loc[mask_up, curr].median() if mask_up.any() else 0)
        res_move['down_next'].append(df_pct.loc[mask_down, curr].median() if mask_down.any() else 0)
        res_move['amt'].append(df_amt.loc[mask_up|mask_down, curr].sum())
    
    res_move = {k: [0 if pd.isna(x) else x for x in v] for k,v in res_move.items()}

    # 4. 集中度与背离
    res_conc = {"dates": date_strs, "top100_amt": [], "top100_ratio": [], "bot3000_ratio": []}
    res_perf = {"dates": date_strs, "top100_pct": [], "bot3000_pct": [], "diff": []}
    
    for d in dates:
        s_amt = df_amt[d].dropna().sort_values(ascending=False)
        total = s_amt.sum()
        if len(s_amt) > 100:
            top100 = s_amt.iloc[:100]
            bot3000 = s_amt.iloc[-3000:] if len(s_amt)>3000 else s_amt.iloc[-min(len(s_amt),3000):]
            
            res_conc['top100_amt'].append(top100.sum())
            res_conc['top100_ratio'].append(top100.sum()/total*100)
            res_conc['bot3000_ratio'].append(bot3000.sum()/total*100)
            
            m_top = df_pct.loc[top100.index, d].median()
            m_bot = df_pct.loc[bot3000.index, d].median()
            res_perf['top100_pct'].append(m_top)
            res_perf['bot3000_pct'].append(m_bot)
            res_perf['diff'].append(m_top - m_bot)
        else:
            for k in ['top100_amt','top100_ratio','bot3000_ratio']: res_conc[k].append(0)
            for k in ['top100_pct','bot3000_pct','diff']: res_perf[k].append(0)
            
    res_perf = {k: [0 if pd.isna(x) else x for x in v] for k,v in res_perf.items()}

    full_data = {
        "menu": [
            {"id": "limit_up_down", "name": "涨跌停与成交额"},
            {"id": "market_breadth", "name": "两市涨跌分布"},
            {"id": "big_movers", "name": "前日大涨大跌个股"},
            {"id": "turnover_conc", "name": "成交额集中度"},
            {"id": "turnover_perf", "name": "量价背离监控"}
        ],
        "charts": {
            "limit_up_down": limit_data,
            "market_breadth": {"dates": date_strs, "up": c_up, "down": c_down, "flat": c_flat, "r_up": r_up, "r_down": r_down},
            "big_movers": res_move,
            "turnover_conc": res_conc,
            "turnover_perf": res_perf
        }
    }
    
    tpl = os.path.join(WEB_TEMPLATE_DIR, 'market_sentiment.html')
    out = os.path.join(OUTPUT_DIR, 'market_sentiment.html')
    if os.path.exists(tpl):
        with open(tpl, 'r', encoding='utf-8') as f: html = f.read()
        with open(out, 'w', encoding='utf-8') as f: 
            f.write(html.replace('{{CHART_DATA}}', json.dumps(full_data, ensure_ascii=False)))

def process_industry_analysis(bundle):
    print("=== [Stage 4] Industry Analysis ===")
    
    df_ind = bundle['ind']
    df_pct = bundle['pct']
    df_amt = bundle['amt']
    dates = bundle['dates']
    
    if df_ind is None or df_pct is None: return

    # 找交集代码
    valid_codes = df_ind.index.intersection(df_pct.index).intersection(df_amt.index)
    print(f"  - Valid Stocks: {len(valid_codes)}")
    
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]
    stocks_data = {}
    
    subset_pct = df_pct.loc[valid_codes]
    subset_amt = df_amt.loc[valid_codes]
    
    print("  - Packaging Data...")
    for code in valid_codes:
        row = df_ind.loc[code]
        # 假设df_ind已经set_index，所以iloc[0]是名称
        name = str(row.iloc[0])
        sw1 = str(row.iloc[1]) if pd.notnull(row.iloc[1]) else ""
        sw2 = str(row.iloc[2]) if pd.notnull(row.iloc[2]) else ""
        sw3 = str(row.iloc[3]) if pd.notnull(row.iloc[3]) else ""
        
        # 【修复】强制转float，避免字符串导致前端显示0
        mv = float(row.iloc[5])
        
        # 序列数据
        p_vals = [round(x, 2) if pd.notnull(x) else 0 for x in subset_pct.loc[code].values]
        a_vals = [round(x, 2) if pd.notnull(x) else 0 for x in subset_amt.loc[code].values]
        
        stocks_data[str(code)] = {
            "n": name,
            "s1": sw1, "s2": sw2, "s3": sw3,
            "mv": mv,
            "p": p_vals,
            "a": a_vals
        }
        
    dataset = {
        "dates": date_strs,
        "stocks": stocks_data
    }
    
    tpl = os.path.join(WEB_TEMPLATE_DIR, 'industry_monitor.html')
    out = os.path.join(OUTPUT_DIR, 'industry_monitor.html')
    if os.path.exists(tpl):
        with open(tpl, 'r', encoding='utf-8') as f: html = f.read()
        final_html = html.replace('{{INDUSTRY_DATA}}', json.dumps(dataset, ensure_ascii=False))
        with open(out, 'w', encoding='utf-8') as f: f.write(final_html)

# ================= 模块5：指数监控 =================
def load_and_clean_index_excel(file_path):
    try:
        if not os.path.exists(file_path): return None, None, None
        df_price = pd.read_excel(file_path, sheet_name='Price-提取') 
        df_pe = pd.read_excel(file_path, sheet_name='市盈率-不剔除负值-提取')
        df_amount = pd.read_excel(file_path, sheet_name='Amount-提取')
        
        cleaned = []
        for df in [df_price, df_pe, df_amount]:
            if '指数' in df.columns:
                df = df.set_index('指数')
                cols_to_drop = [c for c in df.columns if c in ['序号', '代码', 'Index', 'Code']]
                df = df.drop(columns=cols_to_drop)
                df = df.T
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notnull()]
            df.columns = df.columns.astype(str)
            cleaned.append(df)
        return cleaned[0], cleaned[1], cleaned[2]
    except: return None, None, None

def process_index_data():
    print("=== [Stage 5] Index Monitor ===")
    
    dest_folder = os.path.join(DATA_DIR, '003_指数量价跟踪')
    if not os.path.exists(dest_folder): os.makedirs(dest_folder)
    
    path_recent = os.path.join(dest_folder, '指数跟踪第五版-提取表.xlsx')
    path_early = os.path.join(dest_folder, '指数跟踪第五版 - 前期表.xlsx')
    
    safe_copy(SOURCE_INDEX_RECENT, path_recent)
    
    p1, pe1, v1 = load_and_clean_index_excel(path_recent)
    p2, pe2, v2 = load_and_clean_index_excel(path_early)
    
    if p1 is None: return
    
    if p2 is not None:
        df_price = pd.concat([p2, p1])
        df_pe = pd.concat([pe2, pe1])
        df_amount = pd.concat([v2, v1])
    else:
        df_price, df_pe, df_amount = p1, pe1, v1
        
    for df in [df_price, df_pe, df_amount]:
        df = df[~df.index.duplicated(keep='last')]
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)

    user_defined_groups = {
        "核心宽基指数": ['上证指数', '沪深300', '中证500', '创业板指', '中证1000'],
        "扩展的宽基指数": ['东财全A', '上证指数','深证成指', '上证50', '沪深300', '中证500', '创业板指', '中证1000', '中证2000', '科创50'],
        "全部宽基指数": ['东财全A', '东财全A（非金融石化）', '上证指数', '深证成指','上证50', '沪深300', '中证500', '创业板指', '中证1000', '中证2000', '国证2000','创业50', '创业200','科创50', '科创100', '科创200','北证50', '华证微盘'],
        "主要行业指数": ['信息', '医药', '消费','可选','材料','工业','通信','金融', '能源', '公用'],
        "全部行业指数1": ['信息', '医药', '消费','可选','材料','工业','通信','金融', '能源', '公用', '中证环保', '深证红利', '中证红利', '养老产业', '食品饮料', '大农业', '生物科技', '银行', '保险', '证券公司', '地产', '基建工程', '环境治理', '建筑材料', '家用电器', '军工', '电子', '计算机指', '传媒', '5G通信', '交运', '机械设备', '新能车', '有色金属', '煤炭', '钢铁', '石油石化', '基础化工', '美容护理', '酒店、餐馆与休闲', '白酒', '动漫游戏', '300医药', '500医药', '半导体', 'CRO', '光伏', '锂电池', '疫苗', '中药', '航空机场', 'ST板块指数', '中华博彩', '电力电网', '网络安全', '机器人', '黄金', '工业金属', '创新药', '机床', '汽车零部件', '光通信','低空经济','商业航天','智能驾驶','人工智能','创业板人工智能','软件','量子科技','海洋装备','金融科技','数字货币','零售','贸易','稀土永磁','大飞机','恒生港股通新经济', '恒生科技', '恒生医疗保健'],
        "全部行业指数2": ['信息', '医药', '消费','可选','材料','工业','通信','金融', '能源', '公用', '中证环保', '深证红利', '中证红利', '养老产业', '食品饮料', '大农业', '生物科技', '银行', '保险', '证券公司', '地产', '基建工程', '环境治理', '建筑材料', '家用电器', '军工', '电子', '计算机指', '传媒', '5G通信', '交运', '机械设备', '新能车', '有色金属', '煤炭', '钢铁', '石油石化', '基础化工', '美容护理', '酒店、餐馆与休闲', '白酒', '动漫游戏', '300医药', '500医药', '半导体', 'CRO', '光伏', '锂电池', '疫苗', '中药', '航空机场', 'ST板块指数', '中华博彩', '电力电网', '网络安全', '机器人', '黄金', '工业金属', '创新药', '机床', '汽车零部件', '光通信','低空经济','商业航天','智能驾驶','人工智能','创业板人工智能','软件','量子科技','海洋装备','金融科技','数字货币','零售','贸易','稀土永磁','大飞机', '恒生港股通新经济', '恒生科技', '恒生医疗保健','昨日涨停']
    }
    
    final_groups = {}
    actual_cols = [str(c) for c in df_price.columns]
    matched_cols = set()
    
    for g_name, targets in user_defined_groups.items():
        final_groups[g_name] = []
        for alias in targets:
            for real in actual_cols:
                if alias in real:
                    final_groups[g_name].append({'alias': alias, 'key': real})
                    matched_cols.add(real)
                    break
    
    common = df_price.index.intersection(df_pe.index).intersection(df_amount.index)
    output = {
        "dates": common.strftime('%Y-%m-%d').tolist(),
        "groups": final_groups,
        "indices": {}
    }
    
    for col in matched_cols:
        if col in df_price.columns and col in df_pe.columns and col in df_amount.columns:
            output["indices"][col] = {
                "p": df_price.loc[common, col].tolist(),
                "pe": df_pe.loc[common, col].tolist(),
                "v": df_amount.loc[common, col].tolist()
            }
            
    tpl = os.path.join(WEB_TEMPLATE_DIR, 'index_monitor.html')
    out = os.path.join(OUTPUT_DIR, 'index_monitor.html')
    if os.path.exists(tpl):
        with open(tpl, 'r', encoding='utf-8') as f: html = f.read()
        with open(out, 'w', encoding='utf-8') as f: 
            f.write(html.replace('{{INDEX_DATA}}', json.dumps(output, ensure_ascii=False)))

# ================= 主程序入口 =================
if __name__ == "__main__":
    print("==================================================")
    print("   金融市场数据中台 - 自动构建程序 (Unified)")
    print("==================================================")
    
    # 1. 统一加载全A公共数据
    bundle = load_all_common_data()
    
    # 2. 执行各模块
    process_weekly_data()
    process_sentiment_data(bundle)
    process_industry_analysis(bundle)
    
    # 3. 指数模块 (独立IO)
    process_index_data()
    
    print("\n=== 全部任务执行完成 ===")