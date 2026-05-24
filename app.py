"""
app.py — Точка входу Economic Forecaster
=========================================
Логіка:
  1. Якщо користувач НЕ авторизований → показуємо auth.show()
     (сайдбар прихований)
  2. Якщо авторизований → показуємо сайдбар з навігацією
     і відповідну сторінку: home / history / profile
"""

import streamlit as st
from streamlit_option_menu import option_menu
from database import init_db

# ── Ініціалізація БД при першому запуску ────────────────────────────────────
init_db()

# ── Конфігурація сторінки ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Economic Forecaster",
    page_icon="📈",
    layout="wide",
)

# ── Ініціалізація session state ─────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None
if "forecast_triggered" not in st.session_state:
    st.session_state.forecast_triggered = False
if "forecast_saved" not in st.session_state:
    st.session_state.forecast_saved = False


# ════════════════════════════════════════════════════════════════════════════
# НЕ АВТОРИЗОВАНИЙ — показуємо сторінку входу/реєстрації
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.user is None:

    # Повністю ховаємо сайдбар на сторінці авторизації
    st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none !important; }
            [data-testid="stSidebarCollapseButton"] { display: none !important; }
        </style>
    """, unsafe_allow_html=True)

    import auth
    auth.show()


# ════════════════════════════════════════════════════════════════════════════
# АВТОРИЗОВАНИЙ — сайдбар + навігація
# ════════════════════════════════════════════════════════════════════════════

else:
    current_user = st.session_state.user

    with st.sidebar:
        st.title("Аналітика")
        st.write(f"Вітаємо, **{current_user['username']}**!")
        st.divider()

        selected = option_menu(
            menu_title=None,
            options=["Головна", "Історія", "Профіль"],
            icons=["house", "clock-history", "person-circle"],
            default_index=0,
            styles={
                "container":        {"padding": "5px", "background-color": "#262730"},
                "icon":             {"color": "#a29bfe", "font-size": "17px"},
                "nav-link":         {"font-size": "16px", "text-align": "left", "margin": "2px 0"},
                "nav-link-selected":{"background-color": "#3e3f4b"},
            },
        )

        st.divider()

        if st.button("Вийти з акаунту", use_container_width=True):
            # Очищаємо всю сесію
            for key in ["user", "forecast_result", "forecast_triggered",
                        "forecast_saved", "uploaded_filename"]:
                st.session_state[key] = None
            st.session_state.forecast_triggered = False
            st.session_state.forecast_saved = False
            st.rerun()

    # ── Маршрутизація ────────────────────────────────────────────────────────

    if selected == "Головна":
        import home
        with st.sidebar:
            home.render_sidebar()
        home.show()

    elif selected == "Історія":
        import history
        history.show()

    elif selected == "Профіль":
        import profile
        profile.show()