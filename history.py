import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from database import get_user_forecasts, get_forecast_detail, delete_forecast



def show():
    if "history_view" not in st.session_state:
        st.session_state.history_view = "list"
    if "history_selected_id" not in st.session_state:
        st.session_state.history_selected_id = None

    if st.session_state.history_view == "detail" and st.session_state.history_selected_id:
        _show_detail(st.session_state.history_selected_id)
    else:
        _show_list()


def _show_list():
    st.title("Історія прогнозів")

    user_id   = st.session_state.user["id"]
    forecasts = get_user_forecasts(user_id)

    if not forecasts:
        st.info("У вас ще немає збережених прогнозів. Перейдіть на Головну, згенеруйте прогноз і натисніть «Зберегти».")
        return

    st.markdown(f"Знайдено **{len(forecasts)}** прогнозів. Натисніть на назву щоб переглянути деталі.")
    st.divider()

    for fc in forecasts:
        _forecast_card(fc)


def _forecast_card(fc: dict):
    delta = fc.get("delta_pct")
    if delta is not None:
        delta_str  = f"{delta:+.2f}%"
        delta_color = "#2ecc71" if delta >= 0 else "#e74c3c"
    else:
        delta_str  = "—"
        delta_color = "#888"

    accuracy = fc.get("accuracy")
    accuracy_str = f"{accuracy:.1f}%" if accuracy is not None else "—"

    created = fc["created_at"][:10]

    with st.container(border=True):
        col_name, col_meta, col_btns = st.columns([3, 4, 2])

        with col_name:
            st.markdown(f"### {fc['name']}")
            st.caption(f"Збережено: {created}")

        with col_meta:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Модель",    fc["model_type"])
            with m2:
                st.metric("Горизонт",  f"{fc['forecast_days']} дн.")
            with m3:
                st.metric("Точність",  accuracy_str)

            st.markdown(
                f"Прогнозна зміна: <span style='color:{delta_color}; font-weight:600'>{delta_str}</span>",
                unsafe_allow_html=True,
            )

        with col_btns:
            if st.button("Переглянути", key=f"view_{fc['id']}", use_container_width=True):
                st.session_state.history_view       = "detail"
                st.session_state.history_selected_id = fc["id"]
                st.rerun()

            if st.button("Видалити", key=f"del_{fc['id']}", use_container_width=True):
                st.session_state[f"confirm_delete_{fc['id']}"] = True
                st.rerun()

        if st.session_state.get(f"confirm_delete_{fc['id']}"):
            st.warning(f"Видалити прогноз **{fc['name']}**? Це незворотно.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Так, видалити", key=f"yes_{fc['id']}", type="primary", use_container_width=True):
                    delete_forecast(fc["id"], st.session_state.user["id"])
                    st.session_state.pop(f"confirm_delete_{fc['id']}", None)
                    st.rerun()
            with c2:
                if st.button("Скасувати", key=f"no_{fc['id']}", use_container_width=True):
                    st.session_state.pop(f"confirm_delete_{fc['id']}", None)
                    st.rerun()


def _show_detail(forecast_id: int):
    user_id = st.session_state.user["id"]
    fc = get_forecast_detail(forecast_id, user_id)

    if st.button("← Назад до списку", key="back_to_list"):
        st.session_state.history_view        = "list"
        st.session_state.history_selected_id = None
        st.rerun()

    if fc is None:
        st.error("Прогноз не знайдено або не належить вашому акаунту.")
        return

    st.title(f" {fc['name']}")
    st.caption(f"Збережено: {fc['created_at'][:16].replace('T', ' ')}  |  Модель: {fc['model_type']}  |  Горизонт: {fc['forecast_days']} днів")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Остання реальна ціна", f"{fc['last_value']:.4f}")
    with c2:
        delta_val = fc["forecast_end"] - fc["last_value"]
        delta_pct = fc.get("delta_pct") or 0
        st.metric(
            "Прогноз (кінець горизонту)",
            f"{fc['forecast_end']:.4f}",
            delta=f"{delta_val:+.4f} ({delta_pct:+.2f}%)",
        )
    with c3:
        st.metric("Backtest Accuracy", f"{fc['accuracy']:.1f}%" if fc.get("accuracy") else "N/A")
    with c4:
        st.metric("MAPE", f"{fc['mape']:.2f}%" if fc.get("mape") else "N/A")

    st.divider()

    history_points  = fc["history_points"]   
    forecast_points = fc["forecast_points"] 

    if not forecast_points:
        st.warning("Дані графіку недоступні.")
        return

    hist_dates  = [p["date"]  for p in history_points]
    hist_values = [p["value"] for p in history_points]

    fc_dates  = [p["date"]  for p in forecast_points]
    fc_values = [p["value"] for p in forecast_points]
    has_ci    = "lower" in forecast_points[0]

    fig = go.Figure()

    if hist_dates:
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_values,
            name="Історія (останні 60 точок)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        ))

    if has_ci:
        lower = [p["lower"] for p in forecast_points]
        upper = [p["upper"] for p in forecast_points]
        fig.add_trace(go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% довірчий інтервал",
            hoverinfo="skip",
        ))

    if hist_dates:
        p_dates  = [hist_dates[-1]]  + fc_dates
        p_values = [hist_values[-1]] + fc_values
    else:
        p_dates  = fc_dates
        p_values = fc_values

    fig.add_trace(go.Scatter(
        x=p_dates, y=p_values,
        name="Прогноз",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
    ))

    if hist_dates:
        vline_x = hist_dates[-1]
        fig.add_shape(type="line", x0=vline_x, x1=vline_x, y0=0, y1=1,
                      xref="x", yref="paper",
                      line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"))
        fig.add_annotation(x=vline_x, y=1, xref="x", yref="paper",
                           text="Кінець даних", showarrow=False,
                           xanchor="left", yanchor="top",
                           font=dict(color="rgba(255,255,255,0.5)", size=11))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)"),
        title=dict(text=f"Прогноз: {fc['model_type']}", font=dict(size=15)),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Таблиця прогнозних значень"):
        df = pd.DataFrame({
            "Дата":    fc_dates,
            "Прогноз": [round(v, 4) for v in fc_values],
        })
        if has_ci:
            df["Нижня межа (90%)"] = [round(p["lower"], 4) for p in forecast_points]
            df["Верхня межа (90%)"] = [round(p["upper"], 4) for p in forecast_points]
        st.dataframe(df, use_container_width=True)