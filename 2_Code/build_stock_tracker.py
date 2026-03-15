import os
import json
import numpy as np
import pandas as pd


def _load_index_price_df(source_recent, source_early):
    def _load_one(path):
        if not path or (not os.path.exists(path)):
            return None
        try:
            df_price = pd.read_excel(path, sheet_name='Price-提取')
            if '指数' in df_price.columns:
                df_price = df_price.set_index('指数')
                cols_to_drop = [c for c in df_price.columns if c in ['序号', '代码', 'Index', 'Code']]
                df_price = df_price.drop(columns=cols_to_drop, errors='ignore')
                df_price = df_price.T
            df_price.index = pd.to_datetime(df_price.index, errors='coerce')
            df_price = df_price[df_price.index.notnull()]
            df_price.columns = [str(c) for c in df_price.columns]
            return df_price
        except Exception:
            return None

    recent_df = _load_one(source_recent)
    early_df = _load_one(source_early)
    if recent_df is None and early_df is None:
        return None

    if recent_df is None:
        out = early_df.copy()
    elif early_df is None:
        out = recent_df.copy()
    else:
        out = pd.concat([early_df, recent_df])

    out = out[~out.index.duplicated(keep='last')]
    out.sort_index(inplace=True)
    out = out.apply(pd.to_numeric, errors='coerce').ffill()
    return out


def _find_col(df, aliases):
    if df is None:
        return None
    cols = [str(c) for c in df.columns]
    for a in aliases:
        for c in cols:
            if a in c:
                return c
    return None


def _nearest_prior_ts(ts_index, target_ts):
    valid = ts_index[ts_index <= target_ts]
    if len(valid) == 0:
        return ts_index[0]
    return valid[-1]


def _get_row_val(row, keywords, fallback_idx=None):
    for col in row.index:
        name = str(col)
        if any(k in name for k in keywords):
            val = row[col]
            if pd.notnull(val):
                return val
    if fallback_idx is not None and len(row) > fallback_idx and pd.notnull(row.iloc[fallback_idx]):
        return row.iloc[fallback_idx]
    return ''



def _format_date_value(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    ts = pd.to_datetime(v, errors='coerce')
    if pd.isna(ts):
        return str(v)
    return ts.strftime('%Y-%m-%d')


def process_stock_tracker(bundle, web_template_dir, output_dir, source_index_recent=None, source_index_early=None):
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
    target_date = pd.to_datetime(dates[idx_last])

    index_df = _load_index_price_df(source_index_recent, source_index_early)
    hs_col = _find_col(index_df, ['沪深300']) if index_df is not None else None
    zz_col = _find_col(index_df, ['中证1000']) if index_df is not None else None

    hs_pct = 0
    zz_pct = 0
    if index_df is not None and hs_col and zz_col and len(index_df.index) > 1:
        curr_ts = _nearest_prior_ts(index_df.index, target_date)
        curr_loc = index_df.index.get_loc(curr_ts)
        if curr_loc > 0:
            prev_ts = index_df.index[curr_loc - 1]
            hs_prev, hs_curr = float(index_df.loc[prev_ts, hs_col]), float(index_df.loc[curr_ts, hs_col])
            zz_prev, zz_curr = float(index_df.loc[prev_ts, zz_col]), float(index_df.loc[curr_ts, zz_col])
            if hs_prev != 0:
                hs_pct = (hs_curr / hs_prev - 1) * 100
            if zz_prev != 0:
                zz_pct = (zz_curr / zz_prev - 1) * 100

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

    def calc_window_metrics(price_arr, lookback=60):
        start = max(0, len(price_arr) - lookback)
        w = pd.Series(np.array(price_arr[start:], dtype=float))
        if len(w) == 0:
            return {
                '最大回撤(%)': 0,
                '当前较最高回撤(%)': 0,
                '最大回升(%)': 0,
                '当前较最低回升(%)': 0,
            }

        cum_max = w.cummax().replace(0, np.nan)
        drawdowns = (cum_max - w) / cum_max
        max_drawdown = float(drawdowns.max() * 100) if drawdowns.notna().any() else 0

        cum_min = w.cummin().replace(0, np.nan)
        recoveries = (w - cum_min) / cum_min
        max_recovery = float(recoveries.max() * 100) if recoveries.notna().any() else 0

        max_price = float(w.max()) if len(w) else 0
        min_price = float(w.min()) if len(w) else 0
        current_price = float(w.iloc[-1]) if len(w) else 0

        current_drawdown = ((max_price - current_price) / max_price * 100) if max_price else 0
        current_recovery = ((current_price - min_price) / min_price * 100) if min_price else 0

        return {
            '最大回撤(%)': max_drawdown,
            '当前较最高回撤(%)': float(current_drawdown),
            '最大回升(%)': max_recovery,
            '当前较最低回升(%)': float(current_recovery),
        }

    ma_windows = [5, 10, 20, 60, 120, 250, 850]
    rows = []

    for code in valid_codes:
        pct_series = np.array(df_pct.loc[code].values, dtype=float)
        amt_series = np.array(df_amt.loc[code].values, dtype=float)
        price_series = np.array(df_price.loc[code].values, dtype=float)

        row = df_ind.loc[code]
        name = str(_get_row_val(row, ['证券名称', '名称'], 0))
        sw1 = str(_get_row_val(row, ['申万一级', '一级行业'], 1))
        sw2 = str(_get_row_val(row, ['申万二级', '二级行业'], 2))
        sw3 = str(_get_row_val(row, ['申万三级', '三级行业'], 3))

        mv = float(_get_row_val(row, ['总市值', '市值'], 5) or 0)
        ff_mv = float(_get_row_val(row, ['自由流通市值', '流通市值'], 6) or 0)
        pe_ttm = float(_get_row_val(row, ['PE', '市盈率'], 7) or 0)
        pb_mrq = float(_get_row_val(row, ['PB', '市净率'], 8) or 0)
        listed_date = _format_date_value(_get_row_val(row, ['上市日期'], 9))
        listed_board = str(_get_row_val(row, ['上市板块', '板块'], 10))

        draw_metrics = calc_window_metrics(price_series, 60)

        one = {
            '证券代码': str(code),
            '证券名称': name,
            '上市日期': listed_date,
            '上市板块': listed_board,
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
            '近60日最大回撤(%)': draw_metrics['最大回撤(%)'],
            '近60日当前回撤(%)': draw_metrics['当前较最高回撤(%)'],
            '近60日最大回升(%)': draw_metrics['最大回升(%)'],
            '近60日当前回升(%)': draw_metrics['当前较最低回升(%)'],
            '当日涨跌幅相对沪深300(%)': float(pct_series[idx_last]) - hs_pct,
            '当日涨跌幅相对中证1000(%)': float(pct_series[idx_last]) - zz_pct,
        }

        for w in ma_windows:
            start = max(0, idx_last - w + 1)
            ma = float(np.mean(price_series[start:idx_last + 1])) if idx_last >= start else 0
            one[f'{w}日均线乖离率(%)'] = safe_ratio(price_series[idx_last] - ma, ma) * 100
            prev_start = max(0, idx_last - w)
            prev_end = max(0, idx_last)
            prev_ma = float(np.mean(price_series[prev_start:prev_end])) if prev_end > prev_start else ma
            one[f'{w}日均线上行斜率(%)'] = safe_ratio(ma - prev_ma, prev_ma) * 100

        rows.append(one)

    # 行业中位数超额（当日涨跌幅）
    g1_med = {}
    g2_med = {}
    g3_med = {}
    for r in rows:
        g1_med.setdefault(r['申万一级行业'], []).append(r['当日涨跌幅(%)'])
        g2_med.setdefault(r['申万二级行业'], []).append(r['当日涨跌幅(%)'])
        g3_med.setdefault(r['申万三级行业'], []).append(r['当日涨跌幅(%)'])
    g1_med = {k: float(np.median(v)) for k, v in g1_med.items() if k != ''}
    g2_med = {k: float(np.median(v)) for k, v in g2_med.items() if k != ''}
    g3_med = {k: float(np.median(v)) for k, v in g3_med.items() if k != ''}

    for r in rows:
        p = r['当日涨跌幅(%)']
        r['当日涨跌幅相对申万一级中位数(%)'] = p - g1_med.get(r['申万一级行业'], 0)
        r['当日涨跌幅相对申万二级中位数(%)'] = p - g2_med.get(r['申万二级行业'], 0)
        r['当日涨跌幅相对申万三级中位数(%)'] = p - g3_med.get(r['申万三级行业'], 0)

    # 横截面排名（整数）
    sorted_rank_20 = sorted(rows, key=lambda x: x['近20日涨跌幅(%)'], reverse=True)
    for i, item in enumerate(sorted_rank_20, start=1):
        item['近20日涨跌幅_rank'] = int(i)

    sorted_rank_ytd = sorted(rows, key=lambda x: x['年初至今涨跌幅(%)'], reverse=True)
    for i, item in enumerate(sorted_rank_ytd, start=1):
        item['年初至今涨跌幅_rank'] = int(i)

    columns = [
        {'key': '证券代码', 'label': '证券代码', 'module': 'basic'}, {'key': '证券名称', 'label': '证券名称', 'module': 'basic'},
        {'key': '上市日期', 'label': '上市日期', 'module': 'basic'}, {'key': '上市板块', 'label': '上市板块', 'module': 'basic'},
        {'key': '申万一级行业', 'label': '申万一级行业', 'module': 'basic'}, {'key': '申万二级行业', 'label': '申万二级行业', 'module': 'basic'}, {'key': '申万三级行业', 'label': '申万三级行业', 'module': 'basic'},
        {'key': '市值(亿)', 'label': '市值(亿)', 'module': 'basic'}, {'key': '自由流通市值(亿)', 'label': '自由流通市值(亿)', 'module': 'basic'}, {'key': '当日收盘价', 'label': '当日收盘价', 'module': 'basic'},
        {'key': '当日涨跌幅(%)', 'label': '当日涨跌幅(%)', 'module': 'basic'}, {'key': '当日成交额(亿)', 'label': '当日成交额(亿)', 'module': 'basic'}, {'key': 'PE(TTM)', 'label': 'PE(TTM)', 'module': 'basic'}, {'key': 'PB(MRQ)', 'label': 'PB(MRQ)', 'module': 'basic'},
        {'key': '当月涨跌幅(%)', 'label': '当月涨跌幅(%)', 'module': 'daily'}, {'key': '开盘涨跌幅(%)', 'label': '开盘涨跌幅(%)', 'module': 'daily'}, {'key': '日内最高涨幅(%)', 'label': '日内最高涨幅(%)', 'module': 'daily'},
        {'key': '日内最低跌幅(%)', 'label': '日内最低跌幅(%)', 'module': 'daily'}, {'key': '日内收盘最大回撤(%)', 'label': '日内收盘最大回撤(%)', 'module': 'daily'}, {'key': '日内收盘位置(%)', 'label': '日内收盘位置(%)', 'module': 'daily'}, {'key': '振幅(%)', 'label': '振幅(%)', 'module': 'daily'},
        {'key': '本周涨跌幅(%)', 'label': '本周涨跌幅(%)', 'module': 'pct'}, {'key': '本季度涨跌幅(%)', 'label': '本季度涨跌幅(%)', 'module': 'pct'}, {'key': '本年涨跌幅(%)', 'label': '本年涨跌幅(%)', 'module': 'pct'},
        {'key': '近5日涨跌幅(%)', 'label': '近5日涨跌幅(%)', 'module': 'pct'}, {'key': '近10日涨跌幅(%)', 'label': '近10日涨跌幅(%)', 'module': 'pct'}, {'key': '近20日涨跌幅(%)', 'label': '近20日涨跌幅(%)', 'module': 'pct'},
        {'key': '近60日涨跌幅(%)', 'label': '近60日涨跌幅(%)', 'module': 'pct'}, {'key': '近120日涨跌幅(%)', 'label': '近120日涨跌幅(%)', 'module': 'pct'}, {'key': '近250日涨跌幅(%)', 'label': '近250日涨跌幅(%)', 'module': 'pct'}, {'key': '近850日涨跌幅(%)', 'label': '近850日涨跌幅(%)', 'module': 'pct'},
        {'key': '年初至今涨跌幅(%)', 'label': '年初至今涨跌幅(%)', 'module': 'pct'}, {'key': '年初至今涨跌幅_rank', 'label': '年初至今涨跌幅_rank', 'module': 'pct'}, {'key': '近20日涨跌幅_rank', 'label': '近20日涨跌幅_rank', 'module': 'pct'},
        {'key': 'T-5日成交额(亿)', 'label': 'T-5日成交额(亿)', 'module': 'amount'}, {'key': 'T-4日成交额(亿)', 'label': 'T-4日成交额(亿)', 'module': 'amount'}, {'key': 'T-3日成交额(亿)', 'label': 'T-3日成交额(亿)', 'module': 'amount'},
        {'key': 'T-2日成交额(亿)', 'label': 'T-2日成交额(亿)', 'module': 'amount'}, {'key': 'T-1日成交额(亿)', 'label': 'T-1日成交额(亿)', 'module': 'amount'}, {'key': '今日成交额变化率(%)', 'label': '今日成交额变化率(%)', 'module': 'amount'},
        {'key': '今日成交额/前5日均值', 'label': '今日成交额/前5日均值', 'module': 'amount'}, {'key': '今日成交额/前20日均值', 'label': '今日成交额/前20日均值', 'module': 'amount'}, {'key': '换手率(%)', 'label': '换手率(%)', 'module': 'amount'}, {'key': '自由流通换手率(%)', 'label': '自由流通换手率(%)', 'module': 'amount'},
        {'key': '近60日最大回撤(%)', 'label': '近60日最大回撤(%)', 'module': 'drawdown'}, {'key': '近60日当前回撤(%)', 'label': '近60日当前回撤(%)', 'module': 'drawdown'}, {'key': '近60日最大回升(%)', 'label': '近60日最大回升(%)', 'module': 'drawdown'}, {'key': '近60日当前回升(%)', 'label': '近60日当前回升(%)', 'module': 'drawdown'},
    ]

    for w in ma_windows:
        columns.append({'key': f'{w}日均线乖离率(%)', 'label': f'{w}日均线乖离率(%)', 'module': 'ma'})
    for w in ma_windows:
        columns.append({'key': f'{w}日均线上行斜率(%)', 'label': f'{w}日均线上行斜率(%)', 'module': 'ma'})

    columns.extend([
        {'key': '当日涨跌幅相对沪深300(%)', 'label': '当日涨跌幅相对沪深300(%)', 'module': 'excess'},
        {'key': '当日涨跌幅相对中证1000(%)', 'label': '当日涨跌幅相对中证1000(%)', 'module': 'excess'},
        {'key': '当日涨跌幅相对申万一级中位数(%)', 'label': '当日涨跌幅相对申万一级中位数(%)', 'module': 'excess'},
        {'key': '当日涨跌幅相对申万二级中位数(%)', 'label': '当日涨跌幅相对申万二级中位数(%)', 'module': 'excess'},
        {'key': '当日涨跌幅相对申万三级中位数(%)', 'label': '当日涨跌幅相对申万三级中位数(%)', 'module': 'excess'},
    ])

    output = {
        'dates': [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates],
        'columns': columns,
        'rows': rows,
        'meta': {
            'start_date': pd.to_datetime(dates[0]).strftime('%Y-%m-%d'),
            'end_date': pd.to_datetime(dates[-1]).strftime('%Y-%m-%d'),
            'trade_days': int(len(dates)),
        }
    }

    tpl = os.path.join(web_template_dir, 'a_stock_tracker.html')
    out = os.path.join(output_dir, 'a_stock_tracker.html')
    if os.path.exists(tpl):
        with open(tpl, 'r', encoding='utf-8') as f:
            html = f.read()
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html.replace('{{STOCK_TRACKER_DATA}}', json.dumps(output, ensure_ascii=False)))
