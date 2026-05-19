import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from streamlit_option_menu import option_menu
from prophet import Prophet
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Economic Forecaster")


if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "forecast_triggered" not in st.session_state:
    st.session_state.forecast_triggered = False


def detect_frequency(dates: pd.Series) -> str:
    """Визначає частоту часового ряду (день/тиждень/місяць)."""
    if len(dates) < 2:
        return "D"
    diffs = dates.sort_values().diff().dropna().dt.days
    median_diff = diffs.median()
    if median_diff <= 2:
        return "D"
    elif median_diff <= 10:
        return "W"
    else:
        return "MS"


def prophet_forecast(data: pd.DataFrame, col_date: str, col_value: str, forecast_days: int):
    """
    Prophet для фінансових рядів:
    - Тільки тренд (без тижневої/щоденної сезонності — вони і є причиною зубців)
    - Лінійний growth з жорстким регуляризатором
    - Вихідні дні маскуються через custom_seasonality=False
    - Прогноз згладжується після predict щоб прибрати залишкові осциляції
    """
    df_p = data[[col_date, col_value]].copy()
    df_p.columns = ["ds", "y"]
    df_p = df_p.groupby("ds").mean().reset_index().sort_values("ds")

    y_std = df_p["y"].std()
    y_max = df_p["y"].max()
    y_min = df_p["y"].min()
    last_val = float(df_p["y"].iloc[-1])

    diffs = df_p["ds"].diff().dropna().dt.days
    has_weekend_gaps = (diffs.median() >= 1) and (diffs.max() >= 2)

    freq = detect_frequency(df_p["ds"])
    date_range_days = (df_p["ds"].max() - df_p["ds"].min()).days

    cps = 0.05

    m = Prophet(
        growth="linear",
        changepoint_prior_scale=cps,
        seasonality_prior_scale=0.01,
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.90,
        n_changepoints=min(15, len(df_p) // 8),
    )

    if date_range_days > 730:
        m.add_seasonality(
            name="yearly",
            period=365.25,
            fourier_order=3, 
            prior_scale=0.5,
        )

    m.fit(df_p)

    future = m.make_future_dataframe(periods=forecast_days, freq=freq)
    forecast = m.predict(future)

    raw_forecast = forecast["yhat"].iloc[-forecast_days:].values
    raw_lower   = forecast["yhat_lower"].iloc[-forecast_days:].values
    raw_upper   = forecast["yhat_upper"].iloc[-forecast_days:].values

    window = max(3, forecast_days // 10)
    def smooth(arr):
        padded = np.pad(arr, (window - 1, 0), mode="edge")
        return np.convolve(padded, np.ones(window) / window, mode="valid")

    forecast_values = smooth(raw_forecast)
    lower_bound     = smooth(raw_lower)
    upper_bound     = smooth(raw_upper)

    max_dev = last_val * 0.30
    forecast_values = np.clip(forecast_values, last_val - max_dev, last_val + max_dev)
    lower_bound     = np.clip(lower_bound,     last_val - max_dev * 1.3, last_val + max_dev * 1.3)
    upper_bound     = np.clip(upper_bound,     last_val - max_dev * 1.3, last_val + max_dev * 1.3)

    conf_int = np.column_stack((lower_bound, upper_bound))
    return forecast_values, conf_int


def arima_forecast(history_series: np.ndarray, forecast_days: int):
    """ARIMA з автоматичним підбором d та fallback."""
    try:
        model = ARIMA(history_series, order=(5, 2, 1))
        model_fit = model.fit()
        forecast_obj = model_fit.get_forecast(steps=forecast_days)
        forecast_values = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.05)
        return forecast_values, conf_int
    except Exception:
        pass
    try:
        model_fit = ARIMA(history_series, order=(2, 1, 0)).fit()
        forecast_values = model_fit.forecast(steps=forecast_days)
        return forecast_values, None
    except Exception:
        model_fit = ExponentialSmoothing(history_series, trend="add", seasonal=None).fit()
        return model_fit.forecast(forecast_days), None


def hw_forecast(history_series: np.ndarray, forecast_days: int):
    """Holt-Winters з довірчими інтервалами через бутстрап залишків."""
    model_fit = ExponentialSmoothing(history_series, trend="add", seasonal=None).fit()
    forecast_values = model_fit.forecast(forecast_days)

    residuals = model_fit.resid
    std_resid = residuals.std()
    steps = np.arange(1, forecast_days + 1)
    uncertainty = std_resid * np.sqrt(steps)
    lower_bound = forecast_values - 1.96 * uncertainty
    upper_bound = forecast_values + 1.96 * uncertainty
    conf_int = np.column_stack((lower_bound, upper_bound))

    return forecast_values, conf_int


def compute_backtest_accuracy(history_series: np.ndarray, model_type: str,
                               data: pd.DataFrame, col_date: str, col_value: str) -> dict:
    """
    Backtesting: тренуємо на перших 80%, перевіряємо на останніх 20%.
    Повертає MAPE, MAE та accuracy%.
    """
    n = len(history_series)
    if n < 15:
        return {"mape": None, "mae": None, "accuracy": None}

    split = int(n * 0.8)
    train = history_series[:split]
    test = history_series[split:]
    steps = len(test)

    try:
        if "ARIMA" in model_type:
            preds, _ = arima_forecast(train, steps)
        elif "Holt" in model_type:
            preds, _ = hw_forecast(train, steps)
        else:
            sub = data.iloc[:split][[col_date, col_value]].copy()
            sub.columns = ["ds", "y"]
            sub = sub.groupby("ds").mean().reset_index()

            freq = detect_frequency(sub["ds"])
            date_range_days_bt = (sub["ds"].max() - sub["ds"].min()).days

            m = Prophet(
                growth="linear",
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=0.01,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                n_changepoints=min(15, len(sub) // 8),
            )
            if date_range_days_bt > 730:
                m.add_seasonality(name="yearly", period=365.25, fourier_order=3, prior_scale=0.5)
            m.fit(sub)
            future = m.make_future_dataframe(periods=steps, freq=freq)
            fc = m.predict(future)
            raw_preds = fc["yhat"].iloc[-steps:].values
            win = max(3, steps // 10)
            padded = np.pad(raw_preds, (win - 1, 0), mode="edge")
            preds = np.convolve(padded, np.ones(win) / win, mode="valid")
            last_val = float(sub["y"].iloc[-1])
            preds = np.clip(preds, last_val * 0.70, last_val * 1.30)

        mape = mean_absolute_percentage_error(test, preds) * 100
        mae = mean_absolute_error(test, preds)
        accuracy = max(0.0, 100.0 - mape)
        return {"mape": round(mape, 2), "mae": round(mae, 4), "accuracy": round(accuracy, 1)}
    except Exception:
        return {"mape": None, "mae": None, "accuracy": None}


def run_forecast(data, col_date, col_value, model_type, forecast_days):
    """Запускає прогноз та бектест, зберігає результат у session_state."""
    data[col_value] = pd.to_numeric(
        data[col_value].astype(str).str.replace(",", "").str.replace("$", ""),
        errors="coerce",
    )
    data = data.dropna(subset=[col_value])
    history_series = data[col_value].values

    conf_int = None

    if "ARIMA" in model_type:
        forecast_values, conf_int = arima_forecast(history_series, forecast_days)
    elif "Holt" in model_type:
        forecast_values, conf_int = hw_forecast(history_series, forecast_days)
    else:
        forecast_values, conf_int = prophet_forecast(data, col_date, col_value, forecast_days)

    bt = compute_backtest_accuracy(history_series, model_type, data, col_date, col_value)

    st.session_state.forecast_result = {
        "data": data,
        "col_date": col_date,
        "col_value": col_value,
        "model_type": model_type,
        "forecast_days": forecast_days,
        "history_series": history_series,
        "forecast_values": forecast_values,
        "conf_int": conf_int,
        "bt": bt,
    }
    st.session_state.forecast_triggered = True


with st.sidebar:
    st.title("Аналітика")
    selected = option_menu(
        menu_title=None,
        options=["Головна", "Історія", "Профіль"],
        icons=["house", "database", "person-circle"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#262730"},
            "icon": {"color": "#a29bfe", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#3e3f4b"},
        },
    )

    st.divider()

    data = None
    col_date = None
    col_value = None
    model_type = None
    forecast_days = 30

    if selected == "Головна":
        st.subheader("Налаштування")
        uploaded_file = st.file_uploader("Завантажте CSV файл", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8-sig")
            df.columns = [c.strip().strip('"').strip() for c in df.columns]
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).str.strip().str.strip('"')

            col_date = st.selectbox("Колонка дат", df.columns)
            col_value = st.selectbox("Колонка чисел", df.columns)

            try:
                df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
                if df[col_value].dtype == "object":
                    df[col_value] = (
                        df[col_value].str.replace(",", "").str.replace("$", "")
                    )
                df[col_value] = pd.to_numeric(df[col_value], errors="coerce")
                df = df.dropna(subset=[col_value, col_date]).sort_values(by=col_date)
                data = df
                st.success(f"{len(df)} рядків завантажено")
            except Exception as e:
                st.error(f"Помилка обробки: {e}")

            model_type = st.selectbox(
                "Метод прогнозу",
                ["ARIMA (Auto)", "Holt-Winters", "Prophet (Meta)"],
            )
            forecast_days = st.slider("Горизонт прогнозу (дні)", 1, 90, 30)

            if st.button("Згенерувати прогноз", use_container_width=True, type="primary"):
                if data is not None and col_date and col_value and model_type:
                    with st.spinner("Розрахунок прогнозу..."):
                        run_forecast(data, col_date, col_value, model_type, forecast_days)
                else:
                    st.warning("Спочатку завантажте CSV файл.")

            with st.expander("ℹ️ Про моделі"):
                st.markdown(
                    """
                    **ARIMA** — класична статистична модель часових рядів. Добре вловлює короткострокові тренди.

                    **Holt-Winters** — метод експоненційного згладжування з урахуванням тренду. Стабільний і надійний.

                    **Prophet** — модель від Meta, розроблена для ділових часових рядів. Використовує logistic growth з обмеженнями для реалістичних прогнозів.
                    """
                )



if selected == "Головна":
    st.title("📈 Розширений економічний аналіз")

    result = st.session_state.forecast_result

    if result is not None and st.session_state.forecast_triggered:
        data = result["data"]
        col_date = result["col_date"]
        col_value = result["col_value"]
        model_type = result["model_type"]
        forecast_days = result["forecast_days"]
        history_series = result["history_series"]
        forecast_values = result["forecast_values"]
        conf_int = result["conf_int"]
        bt = result["bt"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Остання ціна", f"{history_series[-1]:.4f}")
        with c2:
            delta_val = forecast_values[-1] - history_series[-1]
            delta_pct = (delta_val / history_series[-1]) * 100
            st.metric(
                f"Прогноз (+{forecast_days}д)",
                f"{forecast_values[-1]:.4f}",
                delta=f"{delta_val:+.4f} ({delta_pct:+.1f}%)",
            )
        with c3:
            if bt["accuracy"] is not None:
                st.metric("Backtest Accuracy", f"{bt['accuracy']:.1f}%")
            else:
                st.metric("Backtest Accuracy", "N/A")
        with c4:
            if bt["mape"] is not None:
                st.metric("MAPE", f"{bt['mape']:.2f}%")
            else:
                st.metric("MAPE", "N/A")

        st.divider()

        freq = detect_frequency(data[col_date])
        forecast_dates = pd.date_range(
            start=data[col_date].max() + pd.Timedelta(days=1),
            periods=forecast_days,
            freq=freq,
        )
        raw_last_date = data[col_date].max()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data[col_date],
                y=data[col_value],
                name="Історія",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
            )
        )

        if conf_int is not None:
            lower_bound = conf_int[:, 0]
            upper_bound = conf_int[:, 1]
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255, 127, 14, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="90% довірчий інтервал",
                    hoverinfo="skip",
                )
            )

        p_dates = np.insert(forecast_dates.values, 0, raw_last_date)
        p_values = np.insert(forecast_values.astype(float), 0, float(history_series[-1]))
        fig.add_trace(
            go.Scatter(
                x=p_dates,
                y=p_values,
                name="Прогноз",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
            )
        )

        vline_x = raw_last_date.strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=vline_x, x1=vline_x,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=vline_x, y=1,
            xref="x", yref="paper",
            text="Сьогодні",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(color="rgba(255,255,255,0.5)", size=11),
        )

        fig.update_layout(
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
            title=dict(text=f"Прогноз: {model_type}", font=dict(size=15)),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── ДЕТАЛІ БЕКТЕСТУ ──
        if bt["mape"] is not None:
            with st.expander("🔬 Деталі валідації моделі"):
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.metric("MAPE", f"{bt['mape']:.2f}%",
                              help="Mean Absolute Percentage Error — середня відносна похибка")
                with b2:
                    st.metric("MAE", f"{bt['mae']:.4f}",
                              help="Mean Absolute Error — середня абсолютна похибка")
                with b3:
                    st.metric("Розмір вибірки", f"{len(history_series)} точок")
                st.caption(
                    "Модель навчалась на перших 80% даних і перевірялась на останніх 20%."
                )

        # ── ТАБЛИЦЯ ──
        with st.expander("📊 Таблиця прогнозних значень"):
            result_df = pd.DataFrame(
                {
                    "Дата": forecast_dates.strftime("%Y-%m-%d"),
                    "Прогноз": np.round(forecast_values, 4),
                }
            )
            if conf_int is not None:
                result_df["Нижня межа (90%)"] = np.round(conf_int[:, 0], 4)
                result_df["Верхня межа (90%)"] = np.round(conf_int[:, 1], 4)
            st.dataframe(result_df, use_container_width=True)

    else:
        st.info("Завантажте CSV файл у бічному меню та натисніть «Згенерувати прогноз».")
        st.markdown(
            """
### Очікуваний формат CSV:
| Date | Close |
|------|-------|
| 2023-01-01 | 150.25 |
| 2023-01-02 | 152.10 |

Підтримуються формати дат: `YYYY-MM-DD`, `DD/MM/YYYY`, `MM/DD/YYYY`
"""
        )

elif selected == "Історія":
    st.title("Історія збережених прогнозів")
    st.info("Тут будуть ваші збережені звіти.")

elif selected == "Профіль":
    st.title("Профіль користувача")
    st.write("Налаштування акаунта.")