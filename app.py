"""
Streamlit-приложение:
Оценка экономической эффективности реализации мероприятий
по снижению уровней загрязнения атмосферного воздуха
(МР 5.1.0158-19)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Оценка экономической эффективности",
    page_icon="🌬️",
    layout="wide",
)

# ── Стили ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 1.4rem; font-weight: 700; color: #1a4f7a;
    border-bottom: 3px solid #1a4f7a; padding-bottom: 8px; margin-bottom: 16px;
}
.section-title {
    font-size: 1.05rem; font-weight: 600; color: #2e6da4;
    background: #eaf3fb; padding: 6px 12px; border-radius: 6px; margin: 12px 0 8px 0;
}
.metric-good  { color: #1e8449; font-weight: 700; }
.metric-bad   { color: #c0392b; font-weight: 700; }
.metric-warn  { color: #d35400; font-weight: 700; }
.info-box {
    background: #eaf3fb; border-left: 4px solid #2e6da4;
    padding: 10px 14px; border-radius: 4px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

YEARS = list(range(2018, 2025))

# ── Вспомогательные функции ────────────────────────────────────────────────────
def safe_div(a, b, threshold=1e-9):
    """Деление с защитой от нуля, None и слишком малых знаменателей."""
    try:
        if b is None or a is None:
            return None
        if abs(b) < threshold:
            return None
        return a / b
    except Exception:
        return None

def safe_avg(values):
    """Среднее, игнорируя None."""
    vals = [v for v in values if v is not None]
    return np.mean(vals) if vals else None

def calc_year(row, prev_row):
    """
    Рассчитывает все производные показатели для одного года.
    row, prev_row — словари с ключами:
        ΔGp, ΔRp, ΔHIp, ΔCRp, ΔYp, Cp,
        ΔGk, ΔRk, ΔHIk, ΔCRk, ΔYk, Ck
    Возвращает расширенный словарь.
    """
    r = dict(row)          # копия
    p = prev_row           # предыдущий год (может быть None для 2018)

    # ── Проверка: все фактические данные пустые?
    fact_keys = ["ΔGk", "ΔRk", "ΔHIk", "ΔCRk", "ΔYk", "Ck"]
    has_fact = any(r.get(k) is not None for k in fact_keys)

    # ── Блок результативности (col 15-19) ──────────────────────────────────────
    if has_fact:
        r["res_G"]  = safe_div(r.get("ΔGk"),  r.get("ΔGp"))
        r["res_R"]  = safe_div(r.get("ΔRk"),  r.get("ΔRp"))
        r["res_HI"] = safe_div(r.get("ΔHIk"), r.get("ΔHIp"))
        r["res_CR"] = safe_div(r.get("ΔCRk"), r.get("ΔCRp"))
        r["res_Y"]  = safe_div(r.get("ΔYk"),  r.get("ΔYp"))
        r["res_int"] = safe_avg([r["res_G"], r["res_R"], r["res_HI"], r["res_CR"], r["res_Y"]])
    else:
        for k in ["res_G","res_R","res_HI","res_CR","res_Y","res_int"]:
            r[k] = None

    # ── Приростные значения затрат ──────────────────────────────────────────────
    if has_fact:
        r["ΔCk"] = (r.get("Ck") or 0) - (p.get("Ck") or 0) if p else r.get("Ck")
        r["ΔCp"] = (r.get("Cp") or 0) - (p.get("Cp") or 0) if p else r.get("Cp")

        # Чистый экономический эффект
        r["Ek"] = safe_div_sub(r.get("ΔYk"), r.get("Ck"))   # ΔYk - Ck
        r["Ep"] = safe_div_sub(r.get("ΔYp"), r.get("Cp"))   # ΔYp - Cp

        r["ΔEk"] = (r["Ek"] or 0) - (p.get("Ek") or 0) if p else r["Ek"]
        r["ΔEp"] = (r["Ep"] or 0) - (p.get("Ep") or 0) if p else r["Ep"]

        # Критерий 21: эффективность ЧЭЭ
        if r["Ek"] is not None and r["Ep"] is not None:
            if r["Ek"] < 0 or r["Ep"] < 0:
                r["crit21"] = 0.0
            else:
                r["crit21"] = safe_div(r["Ek"], r["Ep"])
        else:
            r["crit21"] = None

        # Предельные затраты (col AB, AC)
        r["MCk"] = safe_div(r.get("ΔCk"), r.get("ΔEk"))
        r["MCp"] = safe_div(r.get("ΔCp"), r.get("ΔEp"))
        # Критерий 22: MCk/MCp (факт/план, по МР 5.1.0158-19 п.5.3)
        r["crit22"] = safe_div(r.get("MCk"), r.get("MCp"))

        # Средние затраты на единицу снижения риска — ACDk (факт)
        r["ACDk_G"]  = safe_div(r.get("Ck"), r.get("ΔGk"))
        r["ACDk_R"]  = safe_div(r.get("Ck"), r.get("ΔRk"))
        r["ACDk_HI"] = safe_div(r.get("Ck"), r.get("ΔHIk"))
        r["ACDk_CR"] = safe_div(r.get("Ck"), r.get("ΔCRk"))

        # ACDp (план)
        r["ACDp_G"]  = safe_div(r.get("Cp"), r.get("ΔGp"))
        r["ACDp_R"]  = safe_div(r.get("Cp"), r.get("ΔRp"))
        r["ACDp_HI"] = safe_div(r.get("Cp"), r.get("ΔHIp"))
        r["ACDp_CR"] = safe_div(r.get("Cp"), r.get("ΔCRp"))

        # Критерии 23-26: ACDk/ACDp (факт/план, по МР 5.1.0158-19 п.5.4)
        r["crit23"] = safe_div(r.get("ACDk_G"),  r.get("ACDp_G"))
        r["crit24"] = safe_div(r.get("ACDk_R"),  r.get("ACDp_R"))
        r["crit25"] = safe_div(r.get("ACDk_HI"), r.get("ACDp_HI"))
        r["crit26"] = safe_div(r.get("ACDk_CR"), r.get("ACDp_CR"))

        # Предельные затраты на единицу снижения риска — MCDk, MCDp
        dG_k  = (r.get("ΔGk") or 0)  - (p.get("ΔGk") or 0)  if p else r.get("ΔGk")
        dR_k  = (r.get("ΔRk") or 0)  - (p.get("ΔRk") or 0)  if p else r.get("ΔRk")
        dHI_k = (r.get("ΔHIk") or 0) - (p.get("ΔHIk") or 0) if p else r.get("ΔHIk")
        dCR_k = (r.get("ΔCRk") or 0) - (p.get("ΔCRk") or 0) if p else r.get("ΔCRk")

        dG_p  = (r.get("ΔGp") or 0)  - (p.get("ΔGp") or 0)  if p else r.get("ΔGp")
        dR_p  = (r.get("ΔRp") or 0)  - (p.get("ΔRp") or 0)  if p else r.get("ΔRp")
        dHI_p = (r.get("ΔHIp") or 0) - (p.get("ΔHIp") or 0) if p else r.get("ΔHIp")
        dCR_p = (r.get("ΔCRp") or 0) - (p.get("ΔCRp") or 0) if p else r.get("ΔCRp")

        r["MCDk_G"]  = safe_div(r.get("ΔCk"), dG_k)
        r["MCDk_R"]  = safe_div(r.get("ΔCk"), dR_k)
        r["MCDk_HI"] = safe_div(r.get("ΔCk"), dHI_k)
        r["MCDk_CR"] = safe_div(r.get("ΔCk"), dCR_k)

        r["MCDp_G"]  = safe_div(r.get("ΔCp"), dG_p)
        r["MCDp_R"]  = safe_div(r.get("ΔCp"), dR_p)
        r["MCDp_HI"] = safe_div(r.get("ΔCp"), dHI_p)
        r["MCDp_CR"] = safe_div(r.get("ΔCp"), dCR_p)

        # Критерии 27-30: (MCDk - ACDk) / (MCDp - ACDp)
        # Порог: если |знаменатель| слишком мал — результат ненадёжен → None
        def mcd_acd_ratio(mcd_k, acd_k, mcd_p, acd_p):
            if any(v is None for v in [mcd_k, acd_k, mcd_p, acd_p]):
                return None
            numer = mcd_k - acd_k
            denom = mcd_p - acd_p
            if abs(denom) < 1e-6:
                return None
            if abs(numer) > 1e-9 and abs(denom) < abs(numer) * 0.01:
                return None
            result = numer / denom
            # Обрезаем аномальные значения (за пределами [-100, 100])
            if abs(result) > 100:
                return None
            return result

        r["crit27"] = mcd_acd_ratio(r.get("MCDk_G"),  r.get("ACDk_G"),  r.get("MCDp_G"),  r.get("ACDp_G"))
        r["crit28"] = mcd_acd_ratio(r.get("MCDk_R"),  r.get("ACDk_R"),  r.get("MCDp_R"),  r.get("ACDp_R"))
        r["crit29"] = mcd_acd_ratio(r.get("MCDk_HI"), r.get("ACDk_HI"), r.get("MCDp_HI"), r.get("ACDp_HI"))
        r["crit30"] = mcd_acd_ratio(r.get("MCDk_CR"), r.get("ACDk_CR"), r.get("MCDp_CR"), r.get("ACDp_CR"))

        # Интегральный показатель эффективности (31)
        r["eff_int"] = safe_avg([
            r.get("crit21"), r.get("crit22"),
            r.get("crit23"), r.get("crit24"), r.get("crit25"), r.get("crit26"),
            r.get("crit27"), r.get("crit28"), r.get("crit29"), r.get("crit30"),
        ])
    else:
        for k in ["ΔCk","ΔCp","Ek","Ep","ΔEk","ΔEp","crit21","MCk","MCp","crit22",
                  "ACDk_G","ACDk_R","ACDk_HI","ACDk_CR",
                  "ACDp_G","ACDp_R","ACDp_HI","ACDp_CR",
                  "crit23","crit24","crit25","crit26",
                  "MCDk_G","MCDk_R","MCDk_HI","MCDk_CR",
                  "MCDp_G","MCDp_R","MCDp_HI","MCDp_CR",
                  "crit27","crit28","crit29","crit30","eff_int"]:
            r[k] = None

    return r

def safe_div_sub(a, b):
    """a - b с защитой от None."""
    if a is None or b is None:
        return None
    return a - b

def fmt(val, decimals=3):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"

def color_value(val):
    if val is None:
        return "—"
    color = "#1e8449" if val >= 1.0 else "#c0392b"
    return f'<span style="color:{color};font-weight:700">{val:.3f}</span>'

# ── Заголовок ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌬️ Оценка экономической эффективности реализации мероприятий по снижению загрязнения атмосферного воздуха</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Реализация Методических рекомендаций <b>МР 5.1.0158-19</b>. Базовый год — 2017. Период оценки: 2018–2024.</div>', unsafe_allow_html=True)

# ── Ввод данных ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Ввод исходных данных</div>', unsafe_allow_html=True)

tab_plan, tab_fact = st.tabs(["📌 Целевые (плановые) показатели", "📊 Фактические показатели"])

INPUT_FIELDS_PLAN = [
    ("ΔGp",  "ΔGp — снижение смертности (случаев)"),
    ("ΔRp",  "ΔRp — снижение заболеваемости (случаев)"),
    ("ΔHIp", "ΔHIp — уменьшение индекса опасности (б/р)"),
    ("ΔCRp", "ΔCRp — снижение канц. риска (сл./1 000 000)"),
    ("ΔYp",  "ΔYp — предотвращённый экон. ущерб (тыс. руб.)"),
    ("Cp",   "Cp — плановые затраты (тыс. руб.)"),
]
INPUT_FIELDS_FACT = [
    ("ΔGk",  "ΔGk — снижение смертности (случаев)"),
    ("ΔRk",  "ΔRk — снижение заболеваемости (случаев)"),
    ("ΔHIk", "ΔHIk — уменьшение индекса опасности (б/р)"),
    ("ΔCRk", "ΔCRk — снижение канц. риска (сл./1 000 000)"),
    ("ΔYk",  "ΔYk — предотвращённый экон. ущерб (тыс. руб.)"),
    ("Ck",   "Ck — фактические затраты (тыс. руб.)"),
]

if "input_data" not in st.session_state:
    st.session_state.input_data = {yr: {} for yr in YEARS}

with tab_plan:
    st.caption("Значения нарастающим итогом относительно базового 2017 года")
    cols = st.columns(len(YEARS))
    for i, yr in enumerate(YEARS):
        with cols[i]:
            st.markdown(f"**{yr}**")
            for key, label in INPUT_FIELDS_PLAN:
                val = st.number_input(
                    label.split("—")[0].strip(),
                    value=st.session_state.input_data[yr].get(key),
                    key=f"plan_{yr}_{key}",
                    help=label,
                    step=0.001,
                    format="%.3f",
                )
                st.session_state.input_data[yr][key] = val if val != 0.0 else None

with tab_fact:
    st.caption("Фактические значения нарастающим итогом относительно 2017 года")
    cols = st.columns(len(YEARS))
    for i, yr in enumerate(YEARS):
        with cols[i]:
            st.markdown(f"**{yr}**")
            for key, label in INPUT_FIELDS_FACT:
                val = st.number_input(
                    label.split("—")[0].strip(),
                    value=st.session_state.input_data[yr].get(key),
                    key=f"fact_{yr}_{key}",
                    help=label,
                    step=0.001,
                    format="%.3f",
                )
                st.session_state.input_data[yr][key] = val if val != 0.0 else None

# ── Расчёт ────────────────────────────────────────────────────────────────────
if st.button("⚙️ Рассчитать", type="primary", use_container_width=True):
    results = {}
    prev = None
    for yr in YEARS:
        row_data = st.session_state.input_data[yr].copy()
        results[yr] = calc_year(row_data, prev)
        prev = results[yr]
    st.session_state.results = results

if "results" not in st.session_state:
    st.info("Введите данные и нажмите **Рассчитать**.")
    st.stop()

results = st.session_state.results

# ── Сводная таблица результативности ──────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📈 Результаты расчёта</div>', unsafe_allow_html=True)

tab_res, tab_eff, tab_charts = st.tabs(["Результативность", "Эффективность", "📊 Графики"])

# ── Таблица результативности
with tab_res:
    st.markdown("**Критериальные показатели результативности** (значение ≥ 1,0 — высокая результативность)")
    rows_res = []
    for yr in YEARS:
        r = results[yr]
        rows_res.append({
            "Год": yr,
            "ΔGk/ΔGp (смертность)":     fmt(r.get("res_G")),
            "ΔRk/ΔRp (заболеваемость)": fmt(r.get("res_R")),
            "ΔHIk/ΔHIp (инд. опасн.)":  fmt(r.get("res_HI")),
            "ΔCRk/ΔCRp (канц. риск)":   fmt(r.get("res_CR")),
            "ΔYk/ΔYp (экон. ущерб)":    fmt(r.get("res_Y")),
            "Интегральный (20)":         fmt(r.get("res_int")),
        })
    df_res = pd.DataFrame(rows_res).set_index("Год")
    st.dataframe(df_res, use_container_width=True)

    # Интегральный показатель по годам
    int_vals = {yr: results[yr].get("res_int") for yr in YEARS}
    valid_int = {yr: v for yr, v in int_vals.items() if v is not None}
    if valid_int:
        st.markdown("**Интегральный показатель результативности по годам:**")
        for yr, v in valid_int.items():
            icon = "✅" if v >= 1.0 else "⚠️"
            bar_pct = min(int(v * 50), 100)
            bar_color = "#1e8449" if v >= 1.0 else "#c0392b"
            bar_html = f'<div style="background:#ddd;border-radius:4px;height:14px;width:200px;display:inline-block;vertical-align:middle"><div style="background:{bar_color};width:{bar_pct}%;height:100%;border-radius:4px"></div></div>'
            st.markdown(f"{icon} **{yr}**: {v:.3f} &nbsp; {bar_html}", unsafe_allow_html=True)

# ── Таблица эффективности
with tab_eff:
    st.markdown("**Критериальные показатели эффективности** (значение ≥ 1,0 — высокая эффективность)")

    rows_eff = []
    for yr in YEARS:
        r = results[yr]
        rows_eff.append({
            "Год": yr,
            "Ek (ЧЭЭ факт, тыс.р.)":     fmt(r.get("Ek"), 1),
            "Ep (ЧЭЭ план, тыс.р.)":      fmt(r.get("Ep"), 1),
            "Крит.21 Ek/Ep":              fmt(r.get("crit21")),
            "Крит.22 MCp/MCk":            fmt(r.get("crit22")),
            "Крит.23 ACD ∆G":             fmt(r.get("crit23")),
            "Крит.24 ACD ∆R":             fmt(r.get("crit24")),
            "Крит.25 ACD ∆HI":            fmt(r.get("crit25")),
            "Крит.26 ACD ∆CR":            fmt(r.get("crit26")),
            "Крит.27 MCD ∆G":             fmt(r.get("crit27")),
            "Крит.28 MCD ∆R":             fmt(r.get("crit28")),
            "Крит.29 MCD ∆HI":            fmt(r.get("crit29")),
            "Крит.30 MCD ∆CR":            fmt(r.get("crit30")),
            "Интегральный (31)":          fmt(r.get("eff_int")),
        })
    df_eff = pd.DataFrame(rows_eff).set_index("Год")
    st.dataframe(df_eff, use_container_width=True)

    # Краткая интерпретация
    for yr in YEARS:
        r = results[yr]
        ei = r.get("eff_int")
        ri = r.get("res_int")
        if ei is None and ri is None:
            continue
        ei_str = f"эффективность: {'✅ высокая' if ei and ei >= 1 else ('⚠️ низкая' if ei is not None else '—')}"
        ri_str = f"результативность: {'✅ высокая' if ri and ri >= 1 else ('⚠️ низкая' if ri is not None else '—')}"
        st.markdown(f"**{yr}** — {ri_str}; {ei_str}")

# ── Графики ────────────────────────────────────────────────────────────────────
with tab_charts:
    yrs_with_data = [yr for yr in YEARS if results[yr].get("res_int") is not None]
    if not yrs_with_data:
        st.info("Нет данных для построения графиков.")
    else:
        # 1. Интегральные показатели результативности и эффективности
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=yrs_with_data,
            y=[results[yr].get("res_int") for yr in yrs_with_data],
            name="Результативность (инт.)",
            marker_color="#2e6da4",
            opacity=0.85,
        ))
        fig1.add_trace(go.Bar(
            x=yrs_with_data,
            y=[results[yr].get("eff_int") for yr in yrs_with_data],
            name="Эффективность (инт.)",
            marker_color="#e67e22",
            opacity=0.85,
        ))
        fig1.add_hline(y=1.0, line_dash="dash", line_color="red",
                       annotation_text="Порог 1,0", annotation_position="bottom right")
        fig1.update_layout(
            title="Интегральные показатели результативности и эффективности по годам",
            xaxis_title="Год", yaxis_title="Значение",
            barmode="group", template="plotly_white", legend_title="Показатель",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Критерии результативности по показателям
        fig2 = go.Figure()
        metrics_res = [
            ("res_G",  "Смертность",    "#1a6b3a"),
            ("res_R",  "Заболеваемость","#2980b9"),
            ("res_HI", "Инд. опасности","#8e44ad"),
            ("res_CR", "Канц. риск",    "#c0392b"),
            ("res_Y",  "Экон. ущерб",   "#d35400"),
        ]
        for key, label, color in metrics_res:
            fig2.add_trace(go.Scatter(
                x=yrs_with_data,
                y=[results[yr].get(key) for yr in yrs_with_data],
                name=label, mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7),
            ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray",
                       annotation_text="Порог 1,0")
        fig2.update_layout(
            title="Критерии результативности по отдельным показателям",
            xaxis_title="Год", yaxis_title="Отношение факт/план",
            template="plotly_white", legend_title="Показатель",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Чистый экономический эффект план vs факт
        yrs_eff = [yr for yr in YEARS if results[yr].get("Ek") is not None]
        if yrs_eff:
            fig3 = make_subplots(rows=1, cols=2,
                                 subplot_titles=("Чистый экон. эффект (тыс. руб.)",
                                                  "Соотношение затрат план/факт"))

            fig3.add_trace(go.Bar(
                x=yrs_eff,
                y=[results[yr].get("Ek") for yr in yrs_eff],
                name="Ek (факт)", marker_color="#2e86ab",
            ), row=1, col=1)
            fig3.add_trace(go.Bar(
                x=yrs_eff,
                y=[results[yr].get("Ep") for yr in yrs_eff],
                name="Ep (план)", marker_color="#a8d8ea", opacity=0.7,
            ), row=1, col=1)

            fig3.add_trace(go.Scatter(
                x=yrs_eff,
                y=[results[yr].get("Ck") for yr in yrs_eff],
                name="Затраты факт (Ck)", mode="lines+markers",
                line=dict(color="#e74c3c", width=2),
            ), row=1, col=2)
            fig3.add_trace(go.Scatter(
                x=yrs_eff,
                y=[results[yr].get("Cp") for yr in yrs_eff],
                name="Затраты план (Cp)", mode="lines+markers",
                line=dict(color="#3498db", width=2, dash="dot"),
            ), row=1, col=2)

            fig3.update_layout(
                title_text="Экономические показатели: план vs факт",
                template="plotly_white", barmode="group",
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 4. Радарная диаграмма критериев эффективности (последний год с данными)
        last_yr = yrs_with_data[-1]
        r = results[last_yr]
        radar_keys = ["crit21","crit22","crit23","crit24","crit25","crit26","crit27","crit28","crit29","crit30"]
        radar_labels = ["21: Ek/Ep","22: MCp/MCk","23: ACD ΔG","24: ACD ΔR",
                        "25: ACD ΔHI","26: ACD ΔCR","27: MCD ΔG","28: MCD ΔR",
                        "29: MCD ΔHI","30: MCD ΔCR"]
        radar_vals = [r.get(k) for k in radar_keys]
        valid_radar = [(lbl, val) for lbl, val in zip(radar_labels, radar_vals) if val is not None]

        if len(valid_radar) >= 3:
            lbls, vals = zip(*valid_radar)
            lbls = list(lbls) + [lbls[0]]
            vals = list(vals) + [vals[0]]
            fig4 = go.Figure(go.Scatterpolar(
                r=vals, theta=lbls, fill="toself",
                line_color="#e67e22", fillcolor="rgba(230,126,34,0.2)",
                name=f"{last_yr}"
            ))
            fig4.add_trace(go.Scatterpolar(
                r=[1.0]*len(lbls), theta=lbls,
                mode="lines", line=dict(color="red", dash="dash"),
                name="Порог 1,0"
            ))
            fig4.update_layout(
                title=f"Радар критериев эффективности — {last_yr}",
                polar=dict(radialaxis=dict(visible=True)),
                template="plotly_white",
            )
            st.plotly_chart(fig4, use_container_width=True)

# ── Экспорт в CSV ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">💾 Экспорт результатов</div>', unsafe_allow_html=True)

all_rows = []
for yr in YEARS:
    r = results[yr]
    all_rows.append({
        "Год": yr,
        "ΔGp": fmt(r.get("ΔGp")), "ΔRp": fmt(r.get("ΔRp")),
        "ΔHIp": fmt(r.get("ΔHIp")), "ΔCRp": fmt(r.get("ΔCRp")),
        "ΔYp": fmt(r.get("ΔYp")), "Cp": fmt(r.get("Cp")),
        "ΔGk": fmt(r.get("ΔGk")), "ΔRk": fmt(r.get("ΔRk")),
        "ΔHIk": fmt(r.get("ΔHIk")), "ΔCRk": fmt(r.get("ΔCRk")),
        "ΔYk": fmt(r.get("ΔYk")), "Ck": fmt(r.get("Ck")),
        "Рез.(15) ΔGk/ΔGp":  fmt(r.get("res_G")),
        "Рез.(16) ΔRk/ΔRp":  fmt(r.get("res_R")),
        "Рез.(17) ΔHIk/ΔHIp":fmt(r.get("res_HI")),
        "Рез.(18) ΔCRk/ΔCRp":fmt(r.get("res_CR")),
        "Рез.(19) ΔYk/ΔYp":  fmt(r.get("res_Y")),
        "Инт.результат.(20)": fmt(r.get("res_int")),
        "Крит.(21) Ek/Ep":    fmt(r.get("crit21")),
        "Крит.(22) MCp/MCk":  fmt(r.get("crit22")),
        "Крит.(23) ACD ΔG":   fmt(r.get("crit23")),
        "Крит.(24) ACD ΔR":   fmt(r.get("crit24")),
        "Крит.(25) ACD ΔHI":  fmt(r.get("crit25")),
        "Крит.(26) ACD ΔCR":  fmt(r.get("crit26")),
        "Крит.(27) MCD ΔG":   fmt(r.get("crit27")),
        "Крит.(28) MCD ΔR":   fmt(r.get("crit28")),
        "Крит.(29) MCD ΔHI":  fmt(r.get("crit29")),
        "Крит.(30) MCD ΔCR":  fmt(r.get("crit30")),
        "Инт.эффект.(31)":    fmt(r.get("eff_int")),
    })

df_export = pd.DataFrame(all_rows)
csv_data = df_export.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    "⬇️ Скачать результаты (CSV)",
    data=csv_data,
    file_name="economic_efficiency_results.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption("МР 5.1.0158-19 · Федеральный проект «Чистый воздух»")
