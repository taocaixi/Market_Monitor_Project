import os
import json
import numpy as np
import pandas as pd


def process_stock_tracker(bundle, web_template_dir, output_dir):
    print("=== [Stage 6] A Stock Tracker ===")

    df_pct = bundle['pct']
    df_amt = bundle['amt']
    df_price = bundle['price']
    df_ind = bundle['ind']
    dates = bundle['dates']

    if df_pct is None or df_amt is None or df_price is None or df_ind is None or len(dates) < 2:
        return

    valid_codes = df_ind.index.intersection(df_pct.index).intersection(df_amt.index).intersection(df_price.index)
    if len(valid_codes) == 0:
        return

    idx_last = len(dates) - 1

    def safe_ratio(a, b):
        if b == 0 or pd.isna(a) or pd.isna(b):
            return 0
        return float(a / b)

    def pct_between(arr, start_idx, end_idx):
        if start_idx < 0:
            start_idx = 0
        if end_idx <= start_idx:
            return 0
        base = arr[start_idx]
        curr = arr[end_idx]
        if base == 0 or pd.isna(base) or pd.isna(curr):
            return 0
        return float((curr / base - 1) * 100)

    def window_drawdown(price_arr, lookback=60):
        start = max(0, len(price_arr) - lookback)
        w = np.array(price_arr[start:], dtype=float)
        if len(w) == 0:
            return 0, 0
        peak = np.maximum.accumulate(w)
        draw = np.where(peak != 0, (w / peak - 1) * 100, 0)
        trough = np.minimum.accumulate(w)
        rise = np.where(trough != 0, (w / trough - 1) * 100, 0)
        return float(draw[-1]), float(np.nanmax(rise))

    ma_windows = [5, 10, 20, 60, 120, 250, 850]
    rows = []

    for code in valid_codes:
        pct_series = np.array(df_pct.loc[code].values, dtype=float)
        amt_series = np.array(df_amt.loc[code].values, dtype=float)
        price_series = np.array(df_price.loc[code].values, dtype=float)

        row = df_ind.loc[code]
        name = str(row.iloc[0]) if len(row) > 0 else ''
        sw1 = str(row.iloc[1]) if len(row) > 1 and pd.notnull(row.iloc[1]) else ''
        sw2 = str(row.iloc[2]) if len(row) > 2 and pd.notnull(row.iloc[2]) else ''
        sw3 = str(row.iloc[3]) if len(row) > 3 and pd.notnull(row.iloc[3]) else ''
        mv = float(row.iloc[5]) if len(row) > 5 and pd.notnull(row.iloc[5]) else 0
        ff_mv = float(row.iloc[6]) if len(row) > 6 and pd.notnull(row.iloc[6]) else 0
        pe_ttm = float(row.iloc[7]) if len(row) > 7 and pd.notnull(row.iloc[7]) else 0
        pb_mrq = float(row.iloc[8]) if len(row) > 8 and pd.notnull(row.iloc[8]) else 0

        one = {
            '证券代码': str(code),
            '证券名称': name,
            '上市日期': str(row.iloc[9]) if len(row) > 9 and pd.notnull(row.iloc[9]) else '',
            '上市板块': str(row.iloc[10]) if len(row) > 10 and pd.notnull(row.iloc[10]) else '',
            '申万一级行业': sw1,
            '申万二级行业': sw2,
            '申万三级行业': sw3,
            '市值(亿)': mv,
            '自由流通市值(亿)': ff_mv,
            '当日收盘价': float(price_series[idx_last]),
            '当日涨跌幅(%)': float(pct_series[idx_last]),
            '当日成交额(亿)': float(amt_series[idx_last]),
            'PE(TTM)': pe_ttm,
            'PB(MRQ)': pb_mrq,
            '当月涨跌幅(%)': pct_between(price_series, max(0, idx_last - 21), idx_last),
            '开盘涨跌幅(%)': None,
            '日内最高涨幅(%)': None,
            '日内最低跌幅(%)': None,
            '日内收盘最大回撤(%)': None,
            '日内收盘位置(%)': None,
            '振幅(%)': None,
            '本周涨跌幅(%)': pct_between(price_series, max(0, idx_last - 5), idx_last),
            '本季度涨跌幅(%)': pct_between(price_series, max(0, idx_last - 63), idx_last),
            '本年涨跌幅(%)': pct_between(price_series, max(0, idx_last - 250), idx_last),
            '近5日涨跌幅(%)': pct_between(price_series, max(0, idx_last - 5), idx_last),
            '近10日涨跌幅(%)': pct_between(price_series, max(0, idx_last - 10), idx_last),
            '近20日涨跌幅(%)': pct_between(price_series, max(0, idx_last - 20), idx_last),
            '近60日涨跌幅(%)': pct_between(price_series, max(0, idx_last - 60), idx_last),
            '近120日涨跌幅(%)': pct_between(price_series, max(0, idx_last - 120), idx_last),
            '近250日涨跌幅(%)': pct_between(price_series, max(0, idx_last - 250), idx_last),
            '近850日涨跌幅(%)': pct_between(price_series, max(0, idx_last - 850), idx_last),
            '年初至今涨跌幅(%)': pct_between(price_series, max(0, idx_last - 250), idx_last),
            'T-5日成交额(亿)': float(amt_series[max(0, idx_last - 5)]),
            'T-4日成交额(亿)': float(amt_series[max(0, idx_last - 4)]),
            'T-3日成交额(亿)': float(amt_series[max(0, idx_last - 3)]),
            'T-2日成交额(亿)': float(amt_series[max(0, idx_last - 2)]),
            'T-1日成交额(亿)': float(amt_series[max(0, idx_last - 1)]),
            '今日成交额变化率(%)': safe_ratio(amt_series[idx_last] - amt_series[max(0, idx_last - 1)], amt_series[max(0, idx_last - 1)]) * 100,
            '今日成交额/前5日均值': safe_ratio(amt_series[idx_last], np.mean(amt_series[max(0, idx_last - 5):idx_last]) if idx_last > 0 else 0),
            '今日成交额/前20日均值': safe_ratio(amt_series[idx_last], np.mean(amt_series[max(0, idx_last - 20):idx_last]) if idx_last > 0 else 0),
            '换手率(%)': safe_ratio(amt_series[idx_last], mv) * 100,
            '自由流通换手率(%)': safe_ratio(amt_series[idx_last], ff_mv) * 100,
            '近60日当前回撤(%)': window_drawdown(price_series, 60)[0],
            '近60日最大回升(%)': window_drawdown(price_series, 60)[1],
        }

        for w in ma_windows:
            start = max(0, idx_last - w + 1)
            ma = float(np.mean(price_series[start:idx_last + 1])) if idx_last >= start else 0
            one[f'{w}日均线乖离率(%)'] = safe_ratio(price_series[idx_last] - ma, ma) * 100
            prev_start = max(0, idx_last - w)
            prev_end = max(0, idx_last)
            prev_ma = float(np.mean(price_series[prev_start:prev_end])) if prev_end > prev_start else ma
            one[f'{w}日均线上行斜率(%)'] = safe_ratio(ma - prev_ma, prev_ma) * 100

        one['相对沪深300超额收益(%)'] = 0
        one['相对中证1000超额收益(%)'] = 0

        rows.append(one)

    # 近20日涨跌幅横截面排名
    sorted_rank = sorted(rows, key=lambda x: x['近20日涨跌幅(%)'], reverse=True)
    for i, item in enumerate(sorted_rank, start=1):
        item['近20日涨跌幅_rank'] = i

    columns = [
        {'key': '证券代码', 'label': '证券代码', 'module': 'basic'}, {'key': '证券名称', 'label': '证券名称', 'module': 'basic'},
        {'key': '上市日期', 'label': '上市日期', 'module': 'basic'}, {'key': '上市板块', 'label': '上市板块', 'module': 'basic'},
        {'key': '申万一级行业', 'label': '申万一级行业', 'module': 'basic'}, {'key': '申万二级行业', 'label': '申万二级行业', 'module': 'basic'}, {'key': '申万三级行业', 'label': '申万三级行业', 'module': 'basic'},
        {'key': '市值(亿)', 'label': '市值(亿)', 'module': 'basic'}, {'key': '自由流通市值(亿)', 'label': '自由流通市值(亿)', 'module': 'basic'}, {'key': '当日收盘价', 'label': '当日收盘价', 'module': 'basic'},
        {'key': '当日涨跌幅(%)', 'label': '当日涨跌幅(%)', 'module': 'basic'}, {'key': '当日成交额(亿)', 'label': '当日成交额(亿)', 'module': 'basic'}, {'key': 'PE(TTM)', 'label': 'PE(TTM)', 'module': 'basic'}, {'key': 'PB(MRQ)', 'label': 'PB(MRQ)', 'module': 'basic'},
        {'key': '当月涨跌幅(%)', 'label': '当月涨跌幅(%)', 'module': 'monthly'}, {'key': '开盘涨跌幅(%)', 'label': '开盘涨跌幅(%)', 'module': 'monthly'}, {'key': '日内最高涨幅(%)', 'label': '日内最高涨幅(%)', 'module': 'monthly'},
        {'key': '日内最低跌幅(%)', 'label': '日内最低跌幅(%)', 'module': 'monthly'}, {'key': '日内收盘最大回撤(%)', 'label': '日内收盘最大回撤(%)', 'module': 'monthly'}, {'key': '日内收盘位置(%)', 'label': '日内收盘位置(%)', 'module': 'monthly'}, {'key': '振幅(%)', 'label': '振幅(%)', 'module': 'monthly'},
        {'key': '本周涨跌幅(%)', 'label': '本周涨跌幅(%)', 'module': 'pct'}, {'key': '本季度涨跌幅(%)', 'label': '本季度涨跌幅(%)', 'module': 'pct'}, {'key': '本年涨跌幅(%)', 'label': '本年涨跌幅(%)', 'module': 'pct'},
        {'key': '近5日涨跌幅(%)', 'label': '近5日涨跌幅(%)', 'module': 'pct'}, {'key': '近10日涨跌幅(%)', 'label': '近10日涨跌幅(%)', 'module': 'pct'}, {'key': '近20日涨跌幅(%)', 'label': '近20日涨跌幅(%)', 'module': 'pct'},
        {'key': '近60日涨跌幅(%)', 'label': '近60日涨跌幅(%)', 'module': 'pct'}, {'key': '近120日涨跌幅(%)', 'label': '近120日涨跌幅(%)', 'module': 'pct'}, {'key': '近250日涨跌幅(%)', 'label': '近250日涨跌幅(%)', 'module': 'pct'}, {'key': '近850日涨跌幅(%)', 'label': '近850日涨跌幅(%)', 'module': 'pct'},
        {'key': '年初至今涨跌幅(%)', 'label': '年初至今涨跌幅(%)', 'module': 'pct'}, {'key': '近20日涨跌幅_rank', 'label': '近20日涨跌幅_rank', 'module': 'pct'},
        {'key': 'T-5日成交额(亿)', 'label': 'T-5日成交额(亿)', 'module': 'amount'}, {'key': 'T-4日成交额(亿)', 'label': 'T-4日成交额(亿)', 'module': 'amount'}, {'key': 'T-3日成交额(亿)', 'label': 'T-3日成交额(亿)', 'module': 'amount'},
        {'key': 'T-2日成交额(亿)', 'label': 'T-2日成交额(亿)', 'module': 'amount'}, {'key': 'T-1日成交额(亿)', 'label': 'T-1日成交额(亿)', 'module': 'amount'}, {'key': '今日成交额变化率(%)', 'label': '今日成交额变化率(%)', 'module': 'amount'},
        {'key': '今日成交额/前5日均值', 'label': '今日成交额/前5日均值', 'module': 'amount'}, {'key': '今日成交额/前20日均值', 'label': '今日成交额/前20日均值', 'module': 'amount'}, {'key': '换手率(%)', 'label': '换手率(%)', 'module': 'amount'}, {'key': '自由流通换手率(%)', 'label': '自由流通换手率(%)', 'module': 'amount'},
        {'key': '近60日当前回撤(%)', 'label': '近60日当前回撤(%)', 'module': 'drawdown'}, {'key': '近60日最大回升(%)', 'label': '近60日最大回升(%)', 'module': 'drawdown'},
    ]

    for w in ma_windows:
        columns.append({'key': f'{w}日均线乖离率(%)', 'label': f'{w}日均线乖离率(%)', 'module': 'ma'})
    for w in ma_windows:
        columns.append({'key': f'{w}日均线上行斜率(%)', 'label': f'{w}日均线上行斜率(%)', 'module': 'ma'})

    columns.extend([
        {'key': '相对沪深300超额收益(%)', 'label': '相对沪深300超额收益(%)', 'module': 'excess'},
        {'key': '相对中证1000超额收益(%)', 'label': '相对中证1000超额收益(%)', 'module': 'excess'},
    ])

    output = {
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'columns': columns,
        'rows': rows
    }

    tpl = os.path.join(web_template_dir, 'a_stock_tracker.html')
    out = os.path.join(output_dir, 'a_stock_tracker.html')
    if os.path.exists(tpl):
        with open(tpl, 'r', encoding='utf-8') as f:
            html = f.read()
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html.replace('{{STOCK_TRACKER_DATA}}', json.dumps(output, ensure_ascii=False)))

