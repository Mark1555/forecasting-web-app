import streamlit as st
from database import get_user_stats


def show():
    st.title("Профіль")

    user = st.session_state.get("user")
    if not user:
        return

    st.markdown(f"**Ім'я:** {user['username']}")
    st.markdown(f"**Email:** {user['email']}")
    st.markdown(f"**Зареєстровано:** {user['created_at'][:10]}")

    st.divider()

    stats = get_user_stats(user["id"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Всього прогнозів", stats.get("total_forecasts") or 0)
    with c2:
        st.metric("Середня точність", f"{stats['avg_accuracy']:.1f}%" if stats.get("avg_accuracy") else "—")
    with c3:
        st.metric("Улюблена модель", stats.get("favorite_model") or "—")