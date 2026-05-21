"""
pages/auth.py — Сторінка авторизації (вхід / реєстрація)
"""

import streamlit as st
from database import register_user, login_user


def show():
    """Відображає форму входу або реєстрації. Керує st.session_state.user."""

    st.markdown(
        """
        <style>
            /* Змушуємо кнопки вкладок розтягуватися на всю ширину від лівого до правого краю */
            button[data-baseweb="tab"] {
                flex: 1 !important;
                height: 50px !important;
            }
            /* Збільшуємо розмір шрифту та центруємо текст всередині кнопок вкладок */
            button[data-baseweb="tab"] p {
                font-size: 18px !important;
                font-weight: 600 !important;
                text-align: center !important;
                width: 100% !important;
            }
            /* Центруємо всі заголовки тексту (h2, h4, h5) на сторінці */
            h2, h4, h5 {
                text-align: center !important;
                width: 100% !important;
            }
            /* Центруємо назви полів введення (labels) над інпутами */
            [data-testid="stWidgetLabel"] p {
                text-align: center !important;
                width: 100% !important;
            }
            /* Центруємо системні сповіщення (помилки / успіх) */
            [data-testid="stNotification"] {
                text-align: center !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    col_l, col_c, col_r = st.columns([1, 1.6, 1])

    with col_c:
        st.markdown("## Forecaster")
        st.divider()

        tab_login, tab_register = st.tabs(["Вхід", "Реєстрація"])

        with tab_login:
            st.markdown("#### Увійти в акаунт")

            login_input = st.text_input(
                "Ім'я користувача або email",
                key="login_input",
                placeholder="username або email@example.com",
            )
            password_input = st.text_input(
                "Пароль",
                type="password",
                key="login_password",
                placeholder="",
            )

            if st.button("Увійти", use_container_width=True, type="primary", key="btn_login"):
                if not login_input or not password_input:
                    st.error("Будь ласка, заповніть усі поля.")
                else:
                    user = login_user(login_input, password_input)
                    if user:
                        st.session_state.user = user
                        st.success("Успішний вхід! Завантаження...")
                        st.rerun()
                    else:
                        st.error("Невірне ім'я користувача/email або пароль.")

        with tab_register:
            st.markdown("#### Створити акаунт")

            reg_username = st.text_input(
                "Ім'я користувача",
                key="reg_username",
                placeholder="мінімум 3 символи",
            )
            reg_email = st.text_input(
                "Email",
                key="reg_email",
                placeholder="email@example.com",
            )
            reg_password = st.text_input(
                "Пароль",
                type="password",
                key="reg_password",
                placeholder="мінімум 6 символів",
            )
            reg_password2 = st.text_input(
                "Повторіть пароль",
                type="password",
                key="reg_password2",
                placeholder="",
            )

            if st.button("Зареєструватись", use_container_width=True, type="primary", key="btn_register"):
                if not reg_username or not reg_email or not reg_password or not reg_password2:
                    st.error("Заповніть всі поля.")
                elif reg_password != reg_password2:
                    st.error("Паролі не співпадають.")
                else:
                    result = register_user(reg_username, reg_email, reg_password)
                    if result["ok"]:
                        st.success("Акаунт створено! Тепер увійдіть через вкладку «Вхід».")
                    else:
                        st.error(result["message"])