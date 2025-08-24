# dash_app_incidents_wardagg_centroid_lite.py
# -*- coding: utf-8 -*-
"""
宿主分布（幹周しきい＋点）＋事例（区市町村集計）
- デザイン更新（全体背景を明るい緑・左右は透明化・余白拡大・絵文字なし）
  * 画面全体の背景色: GLOBAL_BG（明るい緑）
  * 左右のペイン背景は透明（白カードのみ表示）
  * 外側にフル幅ラッパーを追加（minHeight=100vh）
  * maxWidth=1400px、地図の高さ=56vh、余白は広め
  * 公園情報はカード6つ（全体順位表示つき）
  * 「高リスクのみ表示」(risk-only) あり
"""

from __future__ import annotations

import os
import io
from pathlib import Path
import math
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context

# =============================
# 基本設定
# =============================
BASE = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
TREE_CSV = os.path.join(BASE, 'tokyo_gairoju.csv')               # 必須
INC_CSV  = os.path.join(BASE, 'incidents_tokyo.csv')             # 任意
CENT_CSV_CANDIDATES = [
    os.path.join(BASE, 'tokyo_municipality_centroids.csv'),
    '/mnt/data/tokyo_municipality_centroids.csv',
]
CENT_CSV = next((p for p in CENT_CSV_CANDIDATES if os.path.exists(p)), CENT_CSV_CANDIDATES[0])

# Plotly 既定テンプレート
px.defaults.template = 'plotly_white'

# ====== ブランド配色 ======
PRIMARY_GREEN   = '#2E7D32'  # 深緑（高リスク）
SECONDARY_GREEN = '#7CB342'  # 黄緑（その他）
ACCENT          = '#8E24AA'  # マゼンタ（強調）

RISK_ORDER = ['高リスク', 'その他']
RISK_COLORS = {'高リスク': PRIMARY_GREEN, 'その他': SECONDARY_GREEN}

# ====== 背景・余白 ======
# 全体背景（明るい緑）
GLOBAL_BG   = 'rgba(129,199,132,0.20)'   # #81C784 を薄めたトーン（必要に応じて0.18～0.24で調整）
PANE_BG     = 'transparent'              # 左右は透明（白カードのみ目立たせる）
MAIN_BG     = 'transparent'
CARD_BORDER = '#cfe8dc'

OUTER_GAP = '28px'
CARD_PAD  = '18px'
BLOCK_GAP = '18px'

# 左ペインのブロック共通スタイル（白カード）
SECTION_BOX = {
    'backgroundColor': '#fff',
    'border': f'1px solid {CARD_BORDER}',
    'borderRadius': '14px',
    'padding': CARD_PAD,
    'boxShadow': '0 1px 2px rgba(0,0,0,0.06)'
}

# =============================
# 宿主データ読み込み・整形
# =============================
df = pd.read_csv(TREE_CSV, encoding='cp932')
colmap = {}
for c0 in df.columns:
    c = str(c0).replace('\u3000','').replace('（','(').replace('）',')').replace(' ','')
    if c in ['樹高(m)','樹高m']:
        colmap[c0] = '樹高(m)'
    elif c in ['幹周(cm)','幹周cm','幹周( cm)']:
        colmap[c0] = '幹周(cm)'
    elif c in ['枝張(m)','枝張m']:
        colmap[c0] = '枝張(m)'
    elif c in ['緯度']:
        colmap[c0] = '緯度'
    elif c in ['経度']:
        colmap[c0] = '経度'
    elif c in ['行政区']:
        colmap[c0] = '行政区'
    elif c in ['樹種']:
        colmap[c0] = '樹種'
if colmap:
    df = df.rename(columns=colmap)

required = ['樹種','行政区','緯度','経度','樹高(m)','幹周(cm)']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"CSVに必要な列がありません: {missing}")

for c in ['樹高(m)','幹周(cm)']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

pine_hosts = ['アカマツ', 'クロマツ', 'リュウキュウマツ', 'ヒメコマツ', 'ヤクタネゴヨウ', 'カラマツ']
oak_hosts  = ['ミズナラ', 'フモトミズナラ', 'コナラ', 'クヌギ', 'アベマキ', 'カシワ', 'イチイガシ',
              'アカガシ', 'アラカシ', 'ウラジロガシ', 'シラカシ', 'ウバメガシ', 'クリ', 'スダジイ', 'ツブラジイ', 'マテバシイ']

pine_df = df[df['樹種'].isin(pine_hosts)].copy()
oak_df  = df[df['樹種'].isin(oak_hosts)].copy()
pine_df['病害'] = '松枯れ対象樹'
oak_df['病害']  = 'ナラ枯れ対象樹'
base_df = pd.concat([pine_df, oak_df], ignore_index=True)
base_df = base_df.dropna(subset=['緯度','経度'])

if base_df['幹周(cm)'].notna().any():
    girth_min = max(0, int(math.floor(base_df['幹周(cm)'].min())))
    girth_max = int(math.ceil(base_df['幹周(cm)'].max()))
else:
    girth_min, girth_max = 0, 200

species_options = sorted(base_df['樹種'].dropna().unique().tolist())
ward_options    = sorted(base_df['行政区'].dropna().unique().tolist())

# =============================
# 事例 & センチロイド
# =============================
inc = None
if os.path.exists(INC_CSV):
    inc = pd.read_csv(INC_CSV, encoding='utf-8')
    cmap = {}
    for c0 in inc.columns:
        low = str(c0).strip().lower().replace('　','').replace('（','(').replace('）',')')
        if low in ['disease','病害','事例区分']:
            cmap[c0] = 'disease'
        elif low in ['date','日時','年代','年']:
            cmap[c0] = 'date'
        elif low in ['year','年度']:
            cmap[c0] = 'year'
        elif low in ['ward_city','行政区','市区町村','区市町村']:
            cmap[c0] = 'ward_city'
        elif low in ['lat','緯度']:
            cmap[c0] = 'lat'
        elif low in ['lon','lng','経度']:
            cmap[c0] = 'lon'
        elif low in ['source_title','title']:
            cmap[c0] = 'source_title'
        elif low in ['source_url','url']:
            cmap[c0] = 'source_url'
    if cmap:
        inc = inc.rename(columns=cmap)
    if 'year' not in inc.columns and 'date' in inc.columns:
        inc['year'] = pd.to_datetime(inc['date'], errors='coerce').dt.year

cent = pd.read_csv(CENT_CSV, encoding='utf-8') if os.path.exists(CENT_CSV) else None
focus_ward_options = sorted(cent['ward_city'].dropna().unique().tolist()) if (isinstance(cent, pd.DataFrame) and 'ward_city' in cent.columns) else ward_options

# === 公園CSV（東京都・公園統計）読み込み ===
PARKS_CSV = Path(__file__).with_name("000027629.csv")
try:
    parks_df = pd.read_csv(PARKS_CSV, encoding="utf-8")
    for c in parks_df.columns:
        if c != "行政区分":
            parks_df[c] = pd.to_numeric(parks_df[c], errors="coerce")
except Exception:
    parks_df = pd.DataFrame()

def _is_total_row(s: pd.Series) -> pd.Series:
    return s.astype(str).str.contains("計", na=False)

def make_breakdown(area_name: str):
    if parks_df.empty or "行政区分" not in parks_df.columns:
        return None, {}
    row = parks_df.loc[parks_df["行政区分"] == area_name]
    if row.empty:
        return None, {}

    row = row.squeeze()
    pop = float(row["人口(B)(人)"]) if pd.notna(row["人口(B)(人)"]) else 0.0

    def get(col):
        return float(row[col]) if (col in parks_df.columns and pd.notna(row[col])) else 0.0

    breakdown = [
        ("都立公園",         get("都市公園都立公園数"),         get("都市公園都立公園面積平米")),
        ("区市町村立公園",   get("都市公園区市町村立公園数"),   get("都市公園区市町村立公園面積平米")),
        ("国営公園",         get("都市公園国営公園数"),         get("都市公園国営公園面積平米")),
        ("都市公園 計",      get("都市公園の計数"),             get("都市公園の計面積平米")),
        ("（参考）公立公園合計", get("国都区市町村立公園公立公園合計((八)ｲ+口)数"),
                                 get("国都区市町村立公園公立公園合計((八)ｲ+口)面積平米")),
        ("（参考）総合計",   get("総合計((ﾎ)八+二)数"),         get("総合計((ﾎ)八+二)面積平米")),
    ]
    table = [{
        "カテゴリ": k,
        "公園数": int(n) if not pd.isna(n) else 0,
        "面積(㎡)": round(a, 2),
        "1人当たり(㎡/人)": round(a / pop, 2) if pop > 0 else None
    } for (k, n, a) in breakdown]

    kpis = {
        "都市公園 計(㎡/人)":   round(get("都市公園一人当たり面積(ｲ/B)(平米)"), 2),
        "都市公園 面積割合(%)": round(get("都市公園面積の割合(ｲ/A)(%)"), 2),
        "公立公園合計(㎡/人)": round(get("国都区市町村立公園一人当たり面積(八/B)(平米)"), 2),
        "公立公園 面積割合(%)": round(get("国都区市町村立公園面積の割合(八/A)(%)"), 2),
        "総合 一人当たり(㎡/人)": round(get("一人当たり面積(ﾎ/B)(平米)"), 2),
        "総合 面積割合(%)":       round(get("面積の割合(ﾎ/A)(%)"), 2),
    }
    return table, kpis

# =============================
# Dash UI
# =============================
app = Dash(__name__)
app.title = '宿主分布 & リスク（幹周）＋事例（東京都全域集計）- Lite'

# ---- 左ペイン（白カードで機能ブロック化） ----
controls = html.Div([
    html.Div(id='summary', style={'margin':'0 0 12px'}),

    html.Div([
        html.H4('フィルター', style={'margin':'0 0 10px','fontWeight':'700'}),
        html.Label('病害区分（宿主）'),
        dcc.Dropdown(
            id='disease-filter',
            options=[
                {'label':'両方','value':'both'},
                {'label':'松枯れ対象樹のみ','value':'pine'},
                {'label':'ナラ枯れ対象樹のみ','value':'oak'}
            ], value='both', clearable=False
        ),
        html.Label('樹種（複数選択可）', style={'marginTop':'10px'}),
        dcc.Dropdown(id='species-filter', options=[{'label':s,'value':s} for s in species_options],
                     value=[], multi=True, placeholder='未選択=全て'),
        html.Label('自治体（区市町村・複数可）', style={'marginTop':'10px'}),
        dcc.Dropdown(id='ward-filter', options=[{'label':w,'value':w} for w in ward_options],
                     value=[], multi=True, placeholder='未選択=全て'),
        # 幹周しきい値 100cm（UIは非表示）
        html.Div([
            dcc.Slider(
                id='girth-threshold',
                min=100, max=100, step=None, value=100,
                marks={100: '100'},
                tooltip={'always_visible': True},
                disabled=True
            )
        ], style={'display':'none'})
    ], style=SECTION_BOX),

    html.Div([
        html.H4('表示対象', style={'margin':'0 0 10px','fontWeight':'700'}),
        dcc.Checklist(
            id='risk-only',
            options=[{'label':'ナラ枯れリスク（ブナ科 & 幹周≧しきい）だけ表示','value':'only'}],
            value=[]
        ),
    ], style=SECTION_BOX),

    html.Div([
        html.H4('事例レイヤー', style={'margin':'0 0 10px','fontWeight':'700'}),
        dcc.Checklist(
            id='inc-ward-agg',
            options=[{'label':'区市町村別の件数を地図に重ねる（バブル）','value':'on'}],
            value=['on']
        ),
    ], style=SECTION_BOX),

    html.Div([
        html.H4('アクション', style={'margin':'0 0 10px','fontWeight':'700'}),
        html.Div([
            html.Button('リセット', id='reset-btn',
                        style={'padding':'12px 16px','borderRadius':'9999px','border':f'1px solid {CARD_BORDER}',
                               'background':'#ffffff','boxShadow':'0 2px 10px rgba(0,0,0,0.06)','cursor':'pointer','marginRight':'10px'}),
            html.Button('CSVダウンロード', id='download-btn',
                        style={'padding':'12px 16px','borderRadius':'9999px','border':f'1px solid {CARD_BORDER}',
                               'background':'#ffffff','boxShadow':'0 2px 10px rgba(0,0,0,0.06)','cursor':'pointer'})
        ])
    ], style=SECTION_BOX),

    html.Details([
        html.Summary('ヘルプ / かんたん操作ガイド'),
        html.Ul([
            html.Li('① 「病害区分」「樹種」「自治体」で対象を選びます（未選択=全域）'),
            html.Li('② 幹周しきい値は 100cm 固定です（高リスク=ブナ科かつ幹周≧100cm）'),
            html.Li('③ 事例レイヤーをオンにすると件数感が把握できます'),
            html.Li('④ 右上のチップで「いま効いている条件」を確認できます（リセットで一括解除）'),
            html.Li('⑤ 「CSVダウンロード」で現在の条件のデータを保存できます'),
        ], style={'margin':'6px 0 0 14px'})
    ], open=False, style={**SECTION_BOX, 'marginTop':'-2px'}),
    dcc.Download(id='download'),
], style={
    'width':'420px','flex':'0 0 420px',
    'padding':CARD_PAD,
    'borderRight':f'1px solid {CARD_BORDER}',
    'backgroundColor': PANE_BG, 'borderRadius':'12px',
    'display':'flex','flexDirection':'column','rowGap':BLOCK_GAP
})

# ---- 右メイン ----
main = html.Div([
    html.H2('宿主分布 & リスク（幹周のみ判定）＋事例（東京都全域集計バブル）',
            style={'margin':'0','fontWeight':'700'}),
    html.P('※ 高リスク=ブナ科かつ幹周しきい値以上（背景知見：大径木でカシノナガキクイムシが繁殖しやすく、水分ストレスが枯死を助長。本判定は幹周のみ）。',
           style={'margin':'0 0 6px'}),

    dcc.Graph(
        id='map-graph',
        style={'height':'56vh','backgroundColor':'#fff','border':f'1px solid {CARD_BORDER}',
               'borderRadius':'14px','padding':'10px','boxShadow':'0 6px 16px rgba(0,0,0,0.06)'}
    ),

    html.Div([
        html.Span('● 高リスク', style={'marginRight':'12px','color': PRIMARY_GREEN}),
        html.Span('● その他', style={'color': SECONDARY_GREEN})
    ], style={'fontSize':'12px','userSelect':'none'}),

    html.Div([
        dcc.Graph(id='girth-hist',  style={'height':'36vh','backgroundColor':'#fff','border':f'1px solid {CARD_BORDER}','borderRadius':'12px','padding':'8px'}),
        dcc.Graph(id='height-hist', style={'height':'36vh','backgroundColor':'#fff','border':f'1px solid {CARD_BORDER}','borderRadius':'12px','padding':'8px'})
    ], style={'display':'grid','gridTemplateColumns':'1fr 1fr','gap':BLOCK_GAP}),

    html.Hr(style={'borderColor':CARD_BORDER}),
    html.H4("公園情報（東京都・最新年度：カード6つ）", style={'fontWeight':'700','margin':'0'}),
    html.Div("※ 表示対象の自治体は左の『自治体（区市町村・複数可）』の先頭を使用します",
             style={"fontSize":"0.9rem","color":"#444","margin":"0 0 10px"}),

    html.Div(
        id="park-summary",
        style={"display":"grid","gridTemplateColumns":"repeat(3, minmax(0, 1fr))","gap":BLOCK_GAP}
    ),
], style={
    'flex':'1 1 auto','padding':CARD_PAD,'backgroundColor': MAIN_BG, 'borderRadius':'12px',
    'display':'flex','flexDirection':'column','rowGap':BLOCK_GAP
})

# ---- レイアウト（フル幅ラッパー + 中央コンテナ） ----
app.layout = html.Div([
    html.Div([controls, main],
             style={'display':'flex','gap':OUTER_GAP,
                    'maxWidth':'1400px','margin':'0 auto','padding':OUTER_GAP})
], style={
    'backgroundColor': GLOBAL_BG,   # ページ全体の背景色
    'minHeight': '100vh',           # 画面全体をカバー
    'fontFamily':'"Noto Sans JP", sans-serif'
})

# =============================
# 共通関数
# =============================
def filter_hosts(disease: str, species_sel: list[str], ward_sel: list[str], girth_thr: int, risk_only_values: list[str]):
    if disease == 'pine':
        df_use = base_df[base_df['病害']=='松枯れ対象樹'].copy()
    elif disease == 'oak':
        df_use = base_df[base_df['病害']=='ナラ枯れ対象樹'].copy()
    else:
        df_use = base_df.copy()

    if species_sel:
        df_use = df_use[df_use['樹種'].isin(species_sel)]
    if ward_sel:
        df_use = df_use[df_use['行政区'].isin(ward_sel)]

    is_oak_host = df_use['樹種'].isin(oak_hosts)
    is_large = df_use['幹周(cm)'] >= (girth_thr if girth_thr is not None else 0)
    df_use['リスク'] = ['高リスク' if (oh and lg) else 'その他' for oh, lg in zip(is_oak_host, is_large)]

    risk_only = 'only' in (risk_only_values or [])
    df_plot = df_use[df_use['リスク']=='高リスク'].copy() if risk_only else df_use.copy()
    return df_use, df_plot

def filter_incidents(inc_year_range: list[int]):
    if inc is None or inc.empty:
        return pd.DataFrame(), pd.DataFrame()
    inc_use = inc.copy()
    if 'year' in inc_use.columns and isinstance(inc_year_range, (list, tuple)):
        y0, y1 = inc_year_range[0], inc_year_range[1]
        inc_use = inc_use[(inc_use['year'] >= y0) & (inc_use['year'] <= y1)]
    ward_agg = pd.DataFrame()
    if 'ward_city' in inc_use.columns:
        ward_agg = inc_use.groupby('ward_city', dropna=True).size().reset_index(name='count').sort_values('count', ascending=False)
    return inc_use, ward_agg

def get_center_for_ward(name, df_context: pd.DataFrame):
    if not name:
        return None
    if isinstance(cent, pd.DataFrame) and {'ward_city','lat_c','lon_c'}.issubset(cent.columns):
        m = cent.loc[cent['ward_city'] == name, ['lat_c', 'lon_c']].dropna().head(1)
        if not m.empty:
            try:
                return float(m['lat_c'].iat[0]), float(m['lon_c'].iat[0])
            except Exception:
                pass
    if (isinstance(df_context, pd.DataFrame) and not df_context.empty and
        {'行政区','緯度','経度'}.issubset(df_context.columns)):
        sub = df_context.loc[df_context['行政区'] == name, ['緯度','経度']].dropna()
        if not sub.empty:
            return float(sub['緯度'].mean()), float(sub['経度'].mean())
    return None

# =============================
# メイン更新
# =============================
@app.callback(
    Output('map-graph','figure'),
    Output('girth-hist','figure'),
    Output('height-hist','figure'),
    Output('summary','children'),
    Input('disease-filter','value'),
    Input('species-filter','value'),
    Input('ward-filter','value'),
    Input('girth-threshold','value'),
    Input('risk-only','value'),
    Input('inc-ward-agg','value'),
    Input('reset-btn','n_clicks'),
)
def update(disease, species_sel, ward_sel, girth_thr, risk_only_values, inc_ward_agg, n_reset):

    # リセット
    triggered = [t['prop_id'] for t in (callback_context.triggered or [])]
    if triggered and triggered[0].startswith('reset-btn'):
        species_sel = []
        ward_sel = []
        girth_thr = 100
        risk_only_values = []
        inc_ward_agg = ['on']
        if inc is not None and 'year' in inc.columns and inc['year'].notna().any():
            inc_year_range = [int(inc['year'].min()), int(inc['year'].max())]
        else:
            inc_year_range = [2019, 2025]

    # 宿主フィルタ
    df_use, df_plot = filter_hosts(disease, species_sel or [], ward_sel or [], girth_thr, risk_only_values or [])

    # 事例範囲（UIなし＝全期間）
    if 'inc_year_range' not in locals():
        if inc is not None and 'year' in inc.columns and inc['year'].notna().any():
            inc_year_range = [int(inc['year'].dropna().min()), int(inc['year'].dropna().max())]
        else:
            inc_year_range = [2010, 2025]

    # 事例フィルタ
    inc_use, ward_agg = filter_incidents(inc_year_range)

    default_center_lat, default_center_lon = 35.6895, 139.6917

    # 地図
    if df_plot.empty:
        map_fig = px.scatter_mapbox(lat=[default_center_lat], lon=[default_center_lon], zoom=9)
        map_fig.update_layout(mapbox_style='carto-positron',
                              margin=dict(l=0,r=0,t=0,b=0), legend_title_text='リスク')
    else:
        map_fig = px.scatter_mapbox(
            df_plot, lat='緯度', lon='経度', color='リスク',
            category_orders={'リスク': RISK_ORDER},
            color_discrete_map=RISK_COLORS,
            hover_name='樹種',
            hover_data={'行政区': True, '幹周(cm)': True, '樹高(m)': True, '緯度': False, '経度': False},
            zoom=10
        )
        map_fig.update_traces(marker=dict(size=8, opacity=0.72))
        map_fig.update_layout(mapbox_style='carto-positron',
                              margin=dict(l=0,r=0,t=0,b=0), legend_title_text='リスク')

    # 事例（区市町村集計）バブル重畳
    if (inc is not None and not inc.empty
        and isinstance(cent, pd.DataFrame) and not cent.empty
        and inc_ward_agg and 'on' in inc_ward_agg):
        if not ward_agg.empty:
            agg = ward_agg.merge(cent, on='ward_city', how='left').dropna(subset=['lat_c','lon_c'])
            if not agg.empty:
                size = (agg['count']**0.5)*12 + 6
                map_fig.add_scattermapbox(
                    lat=agg['lat_c'], lon=agg['lon_c'],
                    mode='markers', name='事例（区市町村集計）',
                    marker={'size': size, 'opacity': 0.35, 'color': ACCENT},
                    text=agg['ward_city'],
                    hovertemplate='区市町村: %{text}<br>件数: %{customdata[0]}',
                    customdata=agg[['count']]
                )
                if ward_sel:
                    sel_name = ward_sel[0]
                    sel = agg[agg['ward_city'] == sel_name]
                    if not sel.empty:
                        base_size = (sel['count']**0.5)*12
                        halo_size = (base_size + 22).tolist()
                        core_size = (base_size + 8).tolist()
                        map_fig.add_scattermapbox(
                            lat=sel['lat_c'], lon=sel['lon_c'],
                            mode='markers', name='選択中',
                            marker={'size': halo_size, 'opacity': 0.35, 'color': ACCENT, 'allowoverlap': True},
                            showlegend=False
                        )
                        map_fig.add_scattermapbox(
                            lat=sel['lat_c'], lon=sel['lon_c'],
                            mode='markers', name='',
                            marker={'size': core_size, 'opacity': 0.9, 'color': ACCENT, 'allowoverlap': True},
                            showlegend=False
                        )

    # ヒスト
    girth_df = df_use.dropna(subset=['幹周(cm)'])
    if not girth_df.empty:
        girth_fig = px.histogram(girth_df, x='幹周(cm)', color='リスク', nbins=40, title='幹周(cm) 分布',
                                 category_orders={'リスク': RISK_ORDER}, color_discrete_map=RISK_COLORS)
        girth_fig.add_vline(x=girth_thr, line_dash='dash', line_color=ACCENT, annotation_text=f"しきい {girth_thr}cm")
        girth_fig.update_layout(margin=dict(l=8,r=8,t=36,b=8), barmode='overlay')
        girth_fig.update_traces(opacity=0.75)
    else:
        girth_fig = px.histogram(x=[0], nbins=10, title='幹周(cm) 分布（該当データなし）')
        girth_fig.add_annotation(text='該当データなし — 条件を見直してください', x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        girth_fig.update_layout(margin=dict(l=8,r=8,t=36,b=8))

    height_df = df_use.dropna(subset=['樹高(m)'])
    if not height_df.empty:
        height_fig = px.histogram(height_df, x='樹高(m)', color='リスク', nbins=40, title='樹高(m) 分布',
                                  category_orders={'リスク': RISK_ORDER}, color_discrete_map=RISK_COLORS)
        height_fig.update_layout(margin=dict(l=8,r=8,t=36,b=8), barmode='overlay')
        height_fig.update_traces(opacity=0.75)
    else:
        height_fig = px.histogram(x=[0], nbins=10, title='樹高(m) 分布（該当データなし）')
        height_fig.add_annotation(text='該当データなし — 条件を見直してください', x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        height_fig.update_layout(margin=dict(l=8,r=8,t=36,b=8))

    # KPI
    total = int(len(df_use))
    risk_n = int((df_use['リスク']=='高リスク').sum())
    inc_count = 0
    period_text = '-'
    if inc is not None and 'year' in inc.columns and isinstance(inc_year_range, (list, tuple)):
        inc_count = int(len(inc_use))
        period_text = f"{inc_year_range[0]}–{inc_year_range[1]}"
    risk_pct = (risk_n / total * 100.0) if total else 0.0

    kpi_card_style = {'border':f'1px solid {CARD_BORDER}','borderRadius':'12px',
                      'padding':CARD_PAD,'boxShadow':'0 1px 2px rgba(0,0,0,0.06)','backgroundColor':'#fff'}
    kpi_cards = html.Div([
        html.Div([html.Div('表示本数', style={'fontSize':'12px','color':'#333'}),
                  html.Div(f"{total:,}", style={'fontSize':'20px','fontWeight':'700'})], style=kpi_card_style),
        html.Div([html.Div('高リスク本数 / 割合', style={'fontSize':'12px','color':'#333'}),
                  html.Div(f"{risk_n:,}  ／  {risk_pct:.1f}%", style={'fontSize':'20px','fontWeight':'700'})], style=kpi_card_style),
        html.Div([html.Div('事例件数（期間）', style={'fontSize':'12px','color':'#333'}),
                  html.Div(f"{inc_count:,}  （{period_text}）", style={'fontSize':'20px','fontWeight':'700'})], style=kpi_card_style),
        html.Div([html.Div('自治体選択数', style={'fontSize':'12px','color':'#333'}),
                  html.Div(f"{len(ward_sel or [])}", style={'fontSize':'20px','fontWeight':'700'})], style=kpi_card_style),
    ], style={'display':'grid','gridTemplateColumns':'repeat(2, minmax(0, 1fr))','gap':BLOCK_GAP,
              'position':'sticky','top':'0','zIndex':1,'marginBottom':'8px'})

    def _label_disease(d):
        return {'both':'両方','pine':'松枯れ対象樹','oak':'ナラ枯れ対象樹'}.get(d, '両方')
    chip_items = [
        html.Span(f"病害: {_label_disease(disease)}", style={'border':f'1px solid {CARD_BORDER}','borderRadius':'20px','padding':'4px 10px','fontSize':'12px','backgroundColor':'#fff'}),
        html.Span(f"樹種: {len(species_sel or [])} 種" if (species_sel and len(species_sel)>0) else "樹種: 全て", style={'border':f'1px solid {CARD_BORDER}','borderRadius':'20px','padding':'4px 10px','fontSize':'12px','backgroundColor':'#fff'}),
        html.Span(f"自治体: {len(ward_sel or [])} 件" if (ward_sel and len(ward_sel)>0) else "自治体: 全域", style={'border':f'1px solid {CARD_BORDER}','borderRadius':'20px','padding':'4px 10px','fontSize':'12px','backgroundColor':'#fff'}),
    ]
    if inc is not None and 'year' in inc.columns and isinstance(inc_year_range, (list, tuple)):
        chip_items.append(html.Span(f"事例: {inc_year_range[0]}–{inc_year_range[1]}",
                                    style={'border':f'1px solid {CARD_BORDER}','borderRadius':'20px','padding':'4px 10px','fontSize':'12px','backgroundColor':'#fff'}))
    if 'only' in (risk_only_values or []):
        chip_items.append(html.Span("高リスクのみ", style={'border':f'1px solid {CARD_BORDER}','borderRadius':'20px','padding':'4px 10px','fontSize':'12px','backgroundColor':'#fff'}))
    summary_children = html.Div([
        kpi_cards,
        html.Div(chip_items, style={'display':'flex','flexWrap':'wrap','gap':'6px','marginTop':'6px'})
    ])
    return map_fig, girth_fig, height_fig, summary_children

# =============================
# ダウンロード
# =============================
@app.callback(
    Output('download','data'),
    Input('download-btn','n_clicks'),
    State('disease-filter','value'),
    State('species-filter','value'),
    State('ward-filter','value'),
    State('girth-threshold','value'),
    State('risk-only','value'),
    prevent_initial_call=True
)
def download_csv(n, disease, species_sel, ward_sel, girth_thr, risk_only_values):
    if inc is not None and 'year' in inc.columns and inc['year'].notna().any():
        inc_year_range = [int(inc['year'].dropna().min()), int(inc['year'].dropna().max())]
    else:
        inc_year_range = [2010, 2025]

    df_use, _ = filter_hosts(disease, species_sel or [], ward_sel or [], girth_thr, risk_only_values or [])
    _, ward_agg = filter_incidents(inc_year_range)

    buf = io.StringIO()
    buf.write('# hosts_filtered.csv\n')
    df_use.to_csv(buf, index=False)
    buf.write('\n\n# incidents_ward_agg.csv\n')
    (ward_agg if (ward_agg is not None and not ward_agg.empty) else pd.DataFrame({'ward_city':[], 'count':[]})).to_csv(buf, index=False)
    buf.seek(0)
    return dict(content=buf.getvalue(), filename='filtered_hosts_and_incidents.csv')

# =============================
# 公園情報（カード6つ・順位表示つき）
# =============================
@app.callback(
    Output("park-summary", "children"),
    Input("ward-filter", "value"),
    prevent_initial_call=False
)
def update_park_views(ward_filter_values):
    target = (ward_filter_values[0] if ward_filter_values else None)

    if parks_df.empty:
        return [html.Div("公園CSVが読み込まれていません", style={'padding':CARD_PAD})]

    if not target:
        return [html.Div("地域を選択してください", style={'padding':CARD_PAD})]

    table, kpis = make_breakdown(target)
    if table is None or not kpis:
        return [html.Div(f"{target} のデータが見つかりません", style={'padding':CARD_PAD})]

    rank_col_map = {
        "都市公園 計(㎡/人)": "都市公園一人当たり面積(ｲ/B)(平米)",
        "都市公園 面積割合(%)": "都市公園面積の割合(ｲ/A)(%)",
        "公立公園合計(㎡/人)": "国都区市町村立公園一人当たり面積(八/B)(平米)",
        "公立公園 面積割合(%)": "国都区市町村立公園面積の割合(八/A)(%)",
        "総合 一人当たり(㎡/人)": "一人当たり面積(ﾎ/B)(平米)",
        "総合 面積割合(%)":       "面積の割合(ﾎ/A)(%)",
    }
    try:
        area_idx = parks_df.index[parks_df["行政区分"] == target][0]
    except Exception:
        area_idx = None

    valid_mask = (~_is_total_row(parks_df["行政区分"])) if ("行政区分" in parks_df.columns) else None

    def rank_text_for(label: str) -> str:
        col = rank_col_map.get(label)
        if not col or area_idx is None or valid_mask is None or col not in parks_df.columns:
            return ""
        s = parks_df.loc[valid_mask, col].dropna()
        if s.empty or area_idx not in s.index:
            return ""
        ranks = s.rank(ascending=False, method='min')
        rk = int(ranks.loc[area_idx])
        return f"全体 {rk} 位 / {len(s)}"

    order = [
        "都市公園 計(㎡/人)", "都市公園 面積割合(%)",
        "公立公園合計(㎡/人)", "公立公園 面積割合(%)",
        "総合 一人当たり(㎡/人)", "総合 面積割合(%)"
    ]

    tiles = []
    for label in order:
        val = kpis.get(label)
        tiles.append(
            html.Div([
                html.Div(label, style={"fontSize": "0.9rem", "color": "#333"}),
                html.Div(f"{val:,.2f}" if isinstance(val, (int, float)) else "-", style={"fontSize": "1.5rem", "fontWeight": "700","lineHeight":"1.2"}),
                html.Div(rank_text_for(label), style={"fontSize":"0.8rem","color":"#666","marginTop":"2px"})
            ], style={
                "border": f"1px solid {CARD_BORDER}",
                "borderRadius": "12px",
                "padding": CARD_PAD,
                "boxShadow": "0 1px 2px rgba(0,0,0,0.06)",
                "backgroundColor": "#fff",
                "minHeight":"90px"
            })
        )
    return tiles

# =============================
# 起動
# =============================
if __name__ == '__main__':
    app.run(debug=True)
