import streamlit as st
from database import get_user_stats


def show():
    st.title("Профіль")

    user = st.session_state.get("user")
    if not user:
        return

    c_info, c_gap = st.columns([2, 3])
    with c_info:
        with st.container(border=True):
            st.markdown(f"**Ім'я користувача:** {user['username']}")
            st.markdown(f"**Email:** {user['email']}")
            st.markdown(f"**Зареєстровано:** {user['created_at'][:10]}")

    st.divider()

    st.subheader("Статистика")
    stats = get_user_stats(user["id"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Всього прогнозів", stats.get("total_forecasts") or 0)
    with c2:
        st.metric("Середня точність",
                  f"{stats['avg_accuracy']:.1f}%" if stats.get("avg_accuracy") else "—")
    with c3:
        st.metric("Улюблена модель", stats.get("favorite_model") or "—")

    st.divider()

    col_btn, col_gap = st.columns([1, 3])
    with col_btn:
        if st.button("Вийти з акаунту", use_container_width=True, type="primary"):
            for key in ["user", "forecast_result", "forecast_triggered",
                        "forecast_saved", "uploaded_filename"]:
                st.session_state[key] = None
            st.session_state.forecast_triggered = False
            st.session_state.forecast_saved = False
            st.rerun()