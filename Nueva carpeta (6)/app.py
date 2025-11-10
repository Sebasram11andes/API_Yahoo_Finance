
# Dash tablero — Pred vs Real + tabla de variaciones
import os, joblib, numpy as np, pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go

BASE_DIR = r"C:\Users\Leonardo\Documents\Nueva carpeta (6)"
DATA_PATH = os.path.join(BASE_DIR, "dataset_modelado.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.joblib")

# Cargar datos y modelo
if not os.path.exists(DATA_PATH):
    raise RuntimeError(f"No existe dataset_modelado.csv en: {DATA_PATH}")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No existe model.joblib en: {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None)
df = df.dropna(subset=["Date"]).sort_values(["Ticker","Date"])
model = joblib.load(MODEL_PATH)

FEATURES = ["Ticker","Close","Volume","ret_1d","ma_5","ma_20","vol_ma5","rsi_14"]
tickers = sorted(df["Ticker"].unique().tolist())
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

app = Dash(__name__, title="Entrega 2 - Tablero de Predicción")
app.layout = html.Div(
    style={"backgroundColor":"#0f172a","color":"#e2e8f0","fontFamily":"Inter, Segoe UI, Arial","padding":"16px"},
    children=[
        html.H2("Predicción de Close t+1 (Regresión lineal)", style={"marginBottom":"8px"}),
        html.Div([
            html.Div([
                html.Label("Seleccionar fecha"),
                dcc.DatePickerSingle(
                    id="date_picker", display_format="YYYY-MM-DD",
                    min_date_allowed=min_date, max_date_allowed=max_date,
                    date=max_date
                ),
                html.Br(),
                html.Label("Ticker"),
                dcc.Dropdown(
                    id="ticker_dd",
                    options=[{"label":t, "value":t} for t in tickers],
                    value=tickers[0], clearable=False, style={"width":"220px"}
                ),
                html.Div("Horizonte mostrado: 1 día (t+1)", style={"fontSize":"12px","opacity":0.75,"marginTop":"8px"})
            ], style={"width":"260px","paddingRight":"16px"}),

            html.Div([ dcc.Graph(id="main_graph", style={"height":"420px"}) ], style={"flex":"1"})
        ], style={"display":"flex","alignItems":"flex-start"}),

        html.Hr(style={"opacity":0.2}),
        html.H4("Variaciones por Ticker en la fecha seleccionada (t → t+1)"),
        dash_table.DataTable(
            id="table_vars",
            style_header={"backgroundColor":"#1f2937","color":"#f9fafb","fontWeight":"bold"},
            style_cell={"backgroundColor":"#0f172a","color":"#e5e7eb","border":"1px solid #1f2937","padding":"6px"},
            page_size=10
        )
    ]
)

def compute_predictions_at_date(date_sel):
    m = []
    for t in tickers:
        sub = df[df["Ticker"]==t].copy()
        if not pd.api.types.is_datetime64_any_dtype(sub["Date"]):
            sub["Date"] = pd.to_datetime(sub["Date"], utc=True, errors="coerce").dt.tz_localize(None)
            sub = sub.dropna(subset=["Date"])
        sub = sub[sub["Date"]<=pd.Timestamp(date_sel)]
        if sub.empty: 
            continue
        last = sub.iloc[-1]
        row = {c:last[c] for c in FEATURES if c in sub.columns}
        row_df = pd.DataFrame([row], columns=FEATURES)
        try:
            pred = float(model.predict(row_df)[0])
            close_now = float(last["Close"])
            var_rel = pred / close_now if close_now!=0 else np.nan
            var_pct = (pred - close_now)/close_now if close_now!=0 else np.nan
            var_log = np.log(pred / close_now) if close_now>0 and pred>0 else np.nan
            m.append({
                "Fecha": pd.to_datetime(last["Date"]).date(),
                "Ticker": t,
                "Close_actual": round(close_now, 4),
                "Pred_t1": round(pred, 4),
                "Variación Relativa": round(var_rel, 3),
                "Variación Porcentual": f"{var_pct*100:.2f}%",
                "Variación Logarítmica": round(var_log, 4) if pd.notna(var_log) else None
            })
        except Exception:
            continue
    return pd.DataFrame(m)

@app.callback(
    Output("main_graph","figure"),
    Output("table_vars","data"),
    Output("table_vars","columns"),
    Input("date_picker","date"),
    Input("ticker_dd","value")
)
def update_view(date_sel, ticker_sel):
    if date_sel is None:
        date_sel = max_date
    date_sel = pd.to_datetime(date_sel).date()

    # Subconjunto del ticker
    sub = df[df["Ticker"] == ticker_sel].copy()
    # Garantizar tipo datetime por seguridad
    if not pd.api.types.is_datetime64_any_dtype(sub["Date"]):
        sub["Date"] = pd.to_datetime(sub["Date"], utc=True, errors="coerce").dt.tz_localize(None)
        sub = sub.dropna(subset=["Date"])
    if sub.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ No se encontraron datos para el ticker {ticker_sel}.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color="orange", size=16)
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
        return fig, [], []

    # Si la fecha seleccionada no existe, usar la más cercana anterior
    valid_dates = sub["Date"].dt.date[sub["Date"].dt.date <= date_sel]
    if valid_dates.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ No hay datos históricos previos a {date_sel} para {ticker_sel}.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color="orange", size=16)
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
        return fig, [], []

    date_sel = valid_dates.max()
    sub = sub[sub["Date"].dt.date <= date_sel]

    # Crear figura con serie histórica
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["Date"], y=sub["Close"], mode="lines",
        name="Precio ajustado"
    ))

    # Intentar generar predicción
    try:
        row = pd.DataFrame([sub.iloc[-1][FEATURES]], columns=FEATURES)
        if row.isnull().any().any():
            raise ValueError("Faltan valores en las columnas necesarias para la predicción.")

        pred = float(model.predict(row)[0])
        fig.add_trace(go.Scatter(
            x=[sub["Date"].iloc[-1] + pd.Timedelta(days=1)],
            y=[pred], mode="markers+lines",
            name="Predicción (t+1)", line=dict(dash="dash")
        ))

        # Etiqueta de valor predicho
        fig.add_annotation(
            text=f"Predicción Close t+1 = {pred:.2f}",
            x=sub["Date"].iloc[-1] + pd.Timedelta(days=1),
            y=pred,
            showarrow=True, arrowhead=1, font=dict(color="#38bdf8", size=14),
            arrowcolor="#38bdf8"
        )

    except Exception as e:
        fig.add_annotation(
            text=f"⚠️ No se pudo generar predicción para {ticker_sel} ({date_sel}):<br>{str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color="orange", size=15)
        )

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=10, t=40, b=40),
        xaxis_title="", yaxis_title="Close"
    )

    # Tabla de variaciones
    table_df = compute_predictions_at_date(date_sel)
    cols = [{"name": c, "id": c} for c in table_df.columns] if not table_df.empty else []
    data = table_df.to_dict("records") if not table_df.empty else []
    return fig, data, cols

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
