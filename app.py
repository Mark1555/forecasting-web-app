import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", page_title="Economic Forecaster")

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {display: none !important;}
            [data-testid="stSidebarCollapseButton"] {display: none !important;}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    from pages import auth
    auth.show()

else:
    current_user = st.session_state.user 
    
    with st.sidebar:
        st.title("Аналітика")
        st.write(f"Вітаємо, **{current_user['username']}**!")
       
        selected = option_menu(
            menu_title=None,
            options=["Головна", "Історія", "Профіль"],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#262730"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#3e3f4b"},
            }
        )
        
        st.divider()
        
        if st.button("Вийти з акаунту", use_container_width=True):
            st.session_state.user = None
            st.session_state.forecast_result = None
            st.session_state.forecast_triggered = False
            st.rerun()

    if selected == "Головна":
        from pages import home
        home.render(user_id=current_user["id"])

    elif selected == "Історія":
        st.title("Історія збережених прогнозів")
        st.info("Ця сторінка зараз перебуває в розробці. Тут будуть ваші збережені звіти з бази даних.")

    elif selected == "Профіль":
        st.title("Профіль користувача")
        st.markdown(f"""
        ### Інформація про акаунт:
        * **Ім'я користувача:** {current_user['username']}
        * **Електронна пошта:** {current_user['email']}
        * **ID користувача в базі:** {current_user['id']}
        """)
        st.info("Додаткові налаштування акаунта з'являться згодом.")