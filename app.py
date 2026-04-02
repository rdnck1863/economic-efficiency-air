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
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Оценка экономической эффективности",
    page_icon="🌬️",
    layout="wide",
)

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
.info-box {
    background: #eaf3fb; border-left: 4px solid #2e6da4;
    padding: 10px 14px; border-radius: 4px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Вспомогательные функции ────────────────────────────────────────────────────
def safe_div(a, b, threshold=1e-12):
    try:
        if a is None or b is None: return None
        if abs(b) < threshold: return None
        return a / b
    except Exception:
        return None

def safe_avg(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None

def safe_sub(a, b):
    if a is None or b is None: return None
    return a - b

def fmt(val, decimals=3):
    if val is None: return "—"
    return f"{val:,.{decimals}f}".replace(",", "\u00a0")

def fmt_sci(val):
    if val is None: return "—"
    if val == 0: return "0"
    if abs(val) >= 0.001:
        return f"{val:.6f}".rstrip("0").rstrip(".")
    return f"{val:.3e}"

def fmt_big(val):
    if val is None: return "—"
    return f"{val:,.1f}".replace(",", "\u00a0")

def bar_html(val):
    if val is None: return ""
    pct = min(int(abs(val) * 50), 100) if val >= 0 else 0
    color = "#1e8449" if val >= 1.0 else "#c0392b"
    return (f'<div style="background:#ddd;border-radius:4px;height:12px;'
            f'width:180px;display:inline-block;vertical-align:middle">'
            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div></div>')

# ── Расчёт ────────────────────────────────────────────────────────────────────
def calc_year(row, prev_row):
    r = dict(row)
    p = prev_row

    fact_keys = ["ΔGk", "ΔRk", "ΔHIk", "ΔCRk", "ΔYk", "Ck"]
    has_fact = any(r.get(k) is not None for k in fact_keys)

    if has_fact:
        r["res_G"]   = safe_div(r.get("ΔGk"),  r.get("ΔGp"))
        r["res_R"]   = safe_div(r.get("ΔRk"),  r.get("ΔRp"))
        r["res_HI"]  = safe_div(r.get("ΔHIk"), r.get("ΔHIp"))
        r["res_CR"]  = safe_div(r.get("ΔCRk"), r.get("ΔCRp"))
        r["res_Y"]   = safe_div(r.get("ΔYk"),  r.get("ΔYp"))
        r["res_int"] = safe_avg([r["res_G"], r["res_R"], r["res_HI"], r["res_CR"], r["res_Y"]])
    else:
        for k in ["res_G","res_R","res_HI","res_CR","res_Y","res_int"]:
            r[k] = None

    if has_fact:
        r["ΔCk"] = safe_sub(r.get("Ck"), p.get("Ck") if p else None) if p else r.get("Ck")
        r["ΔCp"] = safe_sub(r.get("Cp"), p.get("Cp") if p else None) if p else r.get("Cp")
        r["Ek"]  = safe_sub(r.get("ΔYk"), r.get("Ck"))
        r["Ep"]  = safe_sub(r.get("ΔYp"), r.get("Cp"))
        r["ΔEk"] = safe_sub(r["Ek"], p.get("Ek") if p else None) if p else r["Ek"]
        r["ΔEp"] = safe_sub(r["Ep"], p.get("Ep") if p else None) if p else r["Ep"]

        if r["Ek"] is not None and r["Ep"] is not None:
            r["crit21"] = 0.0 if (r["Ek"] < 0 or r["Ep"] < 0) else safe_div(r["Ek"], r["Ep"])
        else:
            r["crit21"] = None

        r["MCk"]    = safe_div(r.get("ΔCk"), r.get("ΔEk"))
        r["MCp"]    = safe_div(r.get("ΔCp"), r.get("ΔEp"))
        r["crit22"] = safe_div(r.get("MCk"), r.get("MCp"))

        r["ACDk_G"]  = safe_div(r.get("Ck"), r.get("ΔGk"))
        r["ACDk_R"]  = safe_div(r.get("Ck"), r.get("ΔRk"))
        r["ACDk_HI"] = safe_div(r.get("Ck"), r.get("ΔHIk"))
        r["ACDk_CR"] = safe_div(r.get("Ck"), r.get("ΔCRk"))
        r["ACDp_G"]  = safe_div(r.get("Cp"), r.get("ΔGp"))
        r["ACDp_R"]  = safe_div(r.get("Cp"), r.get("ΔRp"))
        r["ACDp_HI"] = safe_div(r.get("Cp"), r.get("ΔHIp"))
        r["ACDp_CR"] = safe_div(r.get("Cp"), r.get("ΔCRp"))

        r["crit23"] = safe_div(r.get("ACDk_G"),  r.get("ACDp_G"))
        r["crit24"] = safe_div(r.get("ACDk_R"),  r.get("ACDp_R"))
        r["crit25"] = safe_div(r.get("ACDk_HI"), r.get("ACDp_HI"))
        r["crit26"] = safe_div(r.get("ACDk_CR"), r.get("ACDp_CR"))

        def _d(key_k, key_p):
            vk = safe_sub(r.get(key_k), p.get(key_k) if p else None) if p else r.get(key_k)
            vp = safe_sub(r.get(key_p), p.get(key_p) if p else None) if p else r.get(key_p)
            return vk, vp

        dG_k, dG_p   = _d("ΔGk","ΔGp")
        dR_k, dR_p   = _d("ΔRk","ΔRp")
        dHI_k, dHI_p = _d("ΔHIk","ΔHIp")
        dCR_k, dCR_p = _d("ΔCRk","ΔCRp")

        r["MCDk_G"]  = safe_div(r.get("ΔCk"), dG_k)
        r["MCDk_R"]  = safe_div(r.get("ΔCk"), dR_k)
        r["MCDk_HI"] = safe_div(r.get("ΔCk"), dHI_k)
        r["MCDk_CR"] = safe_div(r.get("ΔCk"), dCR_k)
        r["MCDp_G"]  = safe_div(r.get("ΔCp"), dG_p)
        r["MCDp_R"]  = safe_div(r.get("ΔCp"), dR_p)
        r["MCDp_HI"] = safe_div(r.get("ΔCp"), dHI_p)
        r["MCDp_CR"] = safe_div(r.get("ΔCp"), dCR_p)

        def mcd_ratio(mcd_k, acd_k, mcd_p, acd_p):
            if any(v is None for v in [mcd_k, acd_k, mcd_p, acd_p]): return None
            num = mcd_k - acd_k
            den = mcd_p - acd_p
            if abs(den) < 1e-6: return None
            if abs(num) > 1e-9 and abs(den) < abs(num) * 0.01: return None
            res = num / den
            return None if abs(res) > 100 else res

        r["crit27"] = mcd_ratio(r.get("MCDk_G"),  r.get("ACDk_G"),  r.get("MCDp_G"),  r.get("ACDp_G"))
        r["crit28"] = mcd_ratio(r.get("MCDk_R"),  r.get("ACDk_R"),  r.get("MCDp_R"),  r.get("ACDp_R"))
        r["crit29"] = mcd_ratio(r.get("MCDk_HI"), r.get("ACDk_HI"), r.get("MCDp_HI"), r.get("ACDp_HI"))
        r["crit30"] = mcd_ratio(r.get("MCDk_CR"), r.get("ACDk_CR"), r.get("MCDp_CR"), r.get("ACDp_CR"))

        r["eff_int"] = safe_avg([r.get(f"crit{i}") for i in range(21, 31)])
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

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🌬️ Оценка экономической эффективности реализации мероприятий по снижению загрязнения атмосферного воздуха</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Реализация Методических рекомендаций <b>МР 5.1.0158-19</b>.</div>', unsafe_allow_html=True)

# ── Настройка периода ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚙️ Настройка периода оценки</div>', unsafe_allow_html=True)

col_base, col_mgmt = st.columns([1, 3])
with col_base:
    base_year = st.number_input(
        "Базовый год", min_value=2000, max_value=2050, value=2017, step=1,
        help="Год, относительно которого рассчитываются нарастающие итоги"
    )

with col_mgmt:
    if "eval_years" not in st.session_state:
        st.session_state.eval_years = [2018, 2019, 2020]

    st.markdown("**Управление годами оценки:**")
    c1, c2, c3, c4 = st.columns([1.2, 0.8, 1.2, 0.8])
    with c1:
        new_yr = st.number_input(
            "Год для добавления", min_value=2000, max_value=2060,
            value=max(st.session_state.eval_years) + 1 if st.session_state.eval_years else base_year + 1,
            step=1, label_visibility="collapsed"
        )
    with c2:
        if st.button("➕ Добавить"):
            if new_yr not in st.session_state.eval_years:
                st.session_state.eval_years.append(int(new_yr))
                st.session_state.eval_years.sort()
                st.rerun()
            else:
                st.warning(f"{new_yr} уже есть")
    with c3:
        if st.session_state.eval_years:
            yr_del = st.selectbox("Удалить год", options=st.session_state.eval_years,
                                   label_visibility="collapsed")
    with c4:
        if st.session_state.eval_years:
            if st.button("➖ Удалить"):
                st.session_state.eval_years.remove(yr_del)
                if "input_data" in st.session_state and yr_del in st.session_state.input_data:
                    del st.session_state.input_data[yr_del]
                st.rerun()

    if st.session_state.eval_years:
        st.caption(
            f"Годы оценки: **{', '.join(map(str, st.session_state.eval_years))}** "
            f"(базовый: **{base_year}**)"
        )

YEARS = st.session_state.eval_years

if "input_data" not in st.session_state:
    st.session_state.input_data = {}
for yr in YEARS:
    if yr not in st.session_state.input_data:
        st.session_state.input_data[yr] = {}

# ── Поля ввода ────────────────────────────────────────────────────────────────
INPUT_PLAN = [
    ("ΔGp",  "ΔGp",  "Снижение смертности (случаев)",           False),
    ("ΔRp",  "ΔRp",  "Снижение заболеваемости (случаев)",       False),
    ("ΔHIp", "ΔHIp", "Уменьшение индекса опасности (б/р)",       False),
    ("ΔCRp", "ΔCRp", "Снижение канц. риска (сл./1 000 000)",     True),
    ("ΔYp",  "ΔYp",  "Предотвращённый ущерб (тыс. руб.)",       False),
    ("Cp",   "Cp",   "Плановые затраты (тыс. руб.)",            False),
]
INPUT_FACT = [
    ("ΔGk",  "ΔGk",  "Снижение смертности (случаев)",           False),
    ("ΔRk",  "ΔRk",  "Снижение заболеваемости (случаев)",       False),
    ("ΔHIk", "ΔHIk", "Уменьшение индекса опасности (б/р)",       False),
    ("ΔCRk", "ΔCRk", "Снижение канц. риска (сл./1 000 000)",     True),
    ("ΔYk",  "ΔYk",  "Предотвращённый ущерб (тыс. руб.)",       False),
    ("Ck",   "Ck",   "Фактические затраты (тыс. руб.)",         False),
]

def render_inputs(fields, prefix):
    if not YEARS:
        st.info("Сначала добавьте годы оценки выше.")
        return
    cols = st.columns(len(YEARS))
    for i, yr in enumerate(YEARS):
        with cols[i]:
            st.markdown(f"**{yr}**")
            for key, label, hint, is_sci in fields:
                current = st.session_state.input_data.get(yr, {}).get(key, None)
                col_inp, col_clr = st.columns([5, 1])
                with col_clr:
                    st.markdown("<div style='margin-top:22px'></div>", unsafe_allow_html=True)
                    if st.button("✕", key=f"clr_{prefix}_{yr}_{key}", help="Очистить"):
                        st.session_state.input_data.setdefault(yr, {})[key] = None
                        st.rerun()
                with col_inp:
                    fmt_str  = "%.2e" if is_sci else "%.3f"
                    step_val = 1e-7   if is_sci else 0.001
                    val = st.number_input(
                        label,
                        value=float(current) if current is not None else 0.0,
                        key=f"{prefix}_{yr}_{key}",
                        help=hint,
                        step=step_val,
                        format=fmt_str,
                    )
                    st.session_state.input_data.setdefault(yr, {})[key] = val if val != 0.0 else None

st.markdown('<div class="section-title">📋 Ввод исходных данных</div>', unsafe_allow_html=True)
st.caption(f"Значения нарастающим итогом относительно базового **{base_year}** года")

tab_plan, tab_fact = st.tabs(["📌 Целевые (плановые) показатели", "📊 Фактические показатели"])
with tab_plan:
    render_inputs(INPUT_PLAN, "plan")
with tab_fact:
    render_inputs(INPUT_FACT, "fact")

# ── Расчёт ────────────────────────────────────────────────────────────────────
if st.button("⚙️ Рассчитать", type="primary", use_container_width=True):
    results = {}
    prev = None
    for yr in YEARS:
        row_data = st.session_state.input_data.get(yr, {}).copy()
        results[yr] = calc_year(row_data, prev)
        prev = results[yr]
    st.session_state.results = results
    st.session_state.base_year_used = base_year

if "results" not in st.session_state:
    st.info("Введите данные и нажмите **Рассчитать**.")
    st.stop()

results = st.session_state.results
base_year_used = st.session_state.get("base_year_used", base_year)

# ── Результаты ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📈 Результаты расчёта</div>', unsafe_allow_html=True)

tab_res, tab_eff, tab_charts = st.tabs(["Результативность", "Эффективность", "📊 Графики"])

# ── Результативность ──────────────────────────────────────────────────────────
with tab_res:
    st.markdown("**Критериальные показатели результативности** (значение ≥ 1,0 — высокая)")
    rows_res = []
    for yr in YEARS:
        r = results[yr]
        rows_res.append({
            "Год": yr,
            "ΔGk/ΔGp (смертность)":      fmt(r.get("res_G")),
            "ΔRk/ΔRp (заболеваемость)":  fmt(r.get("res_R")),
            "ΔHIk/ΔHIp (инд. опасн.)":   fmt(r.get("res_HI")),
            "ΔCRk/ΔCRp (канц. риск)":    fmt(r.get("res_CR")),
            "ΔYk/ΔYp (экон. ущерб)":     fmt(r.get("res_Y")),
            "Интегральный (20)":          fmt(r.get("res_int")),
        })
    st.dataframe(pd.DataFrame(rows_res).set_index("Год"), use_container_width=True)

    st.markdown("**Интегральный показатель результативности по годам:**")
    has_any_res = False
    for yr in YEARS:
        ri = results[yr].get("res_int")
        if ri is None: continue
        has_any_res = True
        color  = "#1e8449" if ri >= 1.0 else "#c0392b"
        level  = "высокая результативность" if ri >= 1.0 else "низкая результативность"
        icon   = "✅" if ri >= 1.0 else "⚠️"
        st.markdown(
            f"{icon} **{yr}**: "
            f'<span style="color:{color};font-weight:700">{ri:.3f}</span>'
            f' — <span style="color:{color}">{level}</span> &nbsp; {bar_html(ri)}',
            unsafe_allow_html=True
        )
    if not has_any_res:
        st.info("Нет фактических данных для расчёта.")

# ── Эффективность ─────────────────────────────────────────────────────────────
with tab_eff:
    st.markdown("**Критериальные показатели эффективности** (значение ≥ 1,0 — высокая)")
    rows_eff = []
    for yr in YEARS:
        r = results[yr]
        rows_eff.append({
            "Год": yr,
            "Ek факт (тыс.р.)":  fmt_big(r.get("Ek")),
            "Ep план (тыс.р.)":  fmt_big(r.get("Ep")),
            "21: Ek/Ep":         fmt(r.get("crit21")),
            "22: MCk/MCp":       fmt(r.get("crit22")),
            "23: ACD ΔG":        fmt(r.get("crit23")),
            "24: ACD ΔR":        fmt(r.get("crit24")),
            "25: ACD ΔHI":       fmt(r.get("crit25")),
            "26: ACD ΔCR":       fmt_sci(r.get("crit26")),
            "27: MCD ΔG":        fmt(r.get("crit27")),
            "28: MCD ΔR":        fmt(r.get("crit28")),
            "29: MCD ΔHI":       fmt(r.get("crit29")),
            "30: MCD ΔCR":       fmt_sci(r.get("crit30")),
            "Интегральный (31)": fmt(r.get("eff_int")),
        })
    st.dataframe(pd.DataFrame(rows_eff).set_index("Год"), use_container_width=True)

    st.markdown("**Интегральный показатель эффективности по годам:**")
    has_any_eff = False
    for yr in YEARS:
        ei = results[yr].get("eff_int")
        if ei is None: continue
        has_any_eff = True
        color = "#1e8449" if ei >= 1.0 else "#c0392b"
        level = "высокая эффективность" if ei >= 1.0 else "низкая эффективность"
        icon  = "✅" if ei >= 1.0 else "⚠️"
        st.markdown(
            f"{icon} **{yr}**: "
            f'<span style="color:{color};font-weight:700">{ei:.3f}</span>'
            f' — <span style="color:{color}">{level}</span> &nbsp; {bar_html(ei)}',
            unsafe_allow_html=True
        )
    if not has_any_eff:
        st.info("Нет фактических данных для расчёта.")

    st.markdown("---")
    st.markdown("**Сводная оценка по годам:**")
    for yr in YEARS:
        ri = results[yr].get("res_int")
        ei = results[yr].get("eff_int")
        if ri is None and ei is None: continue
        def _v(val, kind):
            if val is None: return f"{kind} —"
            color = "#1e8449" if val >= 1.0 else "#c0392b"
            level = "высокая" if val >= 1.0 else "низкая"
            return f'{kind}: <span style="color:{color};font-weight:700">{level} ({val:.3f})</span>'
        st.markdown(
            f"**{yr}** — {_v(ri,'результативность')} &nbsp;|&nbsp; {_v(ei,'эффективность')}",
            unsafe_allow_html=True
        )

# ── Графики ───────────────────────────────────────────────────────────────────
with tab_charts:
    yrs_with_data = [yr for yr in YEARS if results[yr].get("res_int") is not None]
    if not yrs_with_data:
        st.info("Нет данных для построения графиков.")
    else:
        # ── График 1 ──────────────────────────────────────────────────────────
        st.markdown("#### График 1. Интегральные показатели результативности и эффективности")
        st.info(
            "📖 **Что показывает:** сравнение итоговых оценок по двум критериям для каждого года. "
            "Значение ≥ 1,0 (красная линия) означает, что фактические результаты соответствуют "
            "плану или превышают его. **Синий** — результативность (снижение риска), "
            "**оранжевый** — экономическая эффективность (окупаемость затрат)."
        )
        res_vals = [results[yr].get("res_int") for yr in yrs_with_data]
        eff_vals = [results[yr].get("eff_int") for yr in yrs_with_data]
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=yrs_with_data, y=res_vals, name="Результативность",
            marker_color="#2e6da4", opacity=0.85,
            text=[f"{v:.3f}" if v is not None else "" for v in res_vals],
            textposition="outside"))
        fig1.add_trace(go.Bar(x=yrs_with_data, y=eff_vals, name="Эффективность",
            marker_color="#e67e22", opacity=0.85,
            text=[f"{v:.3f}" if v is not None else "" for v in eff_vals],
            textposition="outside"))
        fig1.add_hline(y=1.0, line_dash="dash", line_color="red",
                       annotation_text="Порог 1,0", annotation_position="bottom right")
        fig1.update_layout(xaxis_title="Год", yaxis_title="Значение",
            barmode="group", template="plotly_white",
            xaxis=dict(type="category"), legend_title="Критерий")
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("**Выводы:**")
        for yr in yrs_with_data:
            ri, ei = results[yr].get("res_int"), results[yr].get("eff_int")
            parts = []
            if ri is not None:
                parts.append(f"результативность {'✅ высокая' if ri>=1 else '⚠️ низкая'} ({ri:.3f})")
            if ei is not None:
                parts.append(f"эффективность {'✅ высокая' if ei>=1 else '⚠️ низкая'} ({ei:.3f})")
            if parts:
                st.markdown(f"- **{yr}**: {'; '.join(parts)}")

        st.markdown("---")

        # ── График 2 ──────────────────────────────────────────────────────────
        st.markdown("#### График 2. Критерии результативности по отдельным показателям")
        st.info(
            "📖 **Что показывает:** динамику выполнения плана по каждому показателю снижения "
            "риска для здоровья. Линия выше 1,0 — фактическое снижение риска превысило "
            "запланированное. **Как читать:** отставание кривой от порога указывает на "
            "показатель, по которому план не выполнен."
        )
        fig2 = go.Figure()
        metrics_res = [
            ("res_G",  "Смертность (ΔG)",      "#1a6b3a"),
            ("res_R",  "Заболеваемость (ΔR)",   "#2980b9"),
            ("res_HI", "Инд. опасности (ΔHI)",  "#8e44ad"),
            ("res_CR", "Канц. риск (ΔCR)",      "#c0392b"),
            ("res_Y",  "Экон. ущерб (ΔY)",      "#d35400"),
        ]
        for key, label, color in metrics_res:
            vals = [results[yr].get(key) for yr in yrs_with_data]
            if any(v is not None for v in vals):
                fig2.add_trace(go.Scatter(x=yrs_with_data, y=vals,
                    name=label, mode="lines+markers",
                    line=dict(color=color, width=2), marker=dict(size=8)))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Порог 1,0")
        fig2.update_layout(xaxis_title="Год", yaxis_title="Факт / план",
            template="plotly_white", xaxis=dict(type="category"), legend_title="Показатель")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Выводы:**")
        for key, label, _ in metrics_res:
            pairs = [(yr, results[yr].get(key)) for yr in yrs_with_data if results[yr].get(key) is not None]
            if not pairs: continue
            above = [str(yr) for yr, v in pairs if v >= 1.0]
            below = [str(yr) for yr, v in pairs if v < 1.0]
            parts = []
            if above: parts.append(f"план выполнен в {', '.join(above)}")
            if below: parts.append(f"план не выполнен в {', '.join(below)}")
            st.markdown(f"- **{label}**: {'; '.join(parts)}")

        st.markdown("---")

        # ── График 3 ──────────────────────────────────────────────────────────
        yrs_eff = [yr for yr in YEARS if results[yr].get("Ek") is not None]
        if yrs_eff:
            st.markdown("#### График 3. Экономические показатели: план vs факт")
            st.info(
                "📖 **Левая часть:** чистый экономический эффект (ЧЭЭ) — разница между "
                "предотвращённым ущербом и затратами. Положительные значения — вложения "
                "окупились. **Правая часть:** сравнение фактических и плановых затрат. "
                "Если красная линия ниже синей — программа реализуется экономнее плана."
            )
            fig3 = make_subplots(rows=1, cols=2,
                subplot_titles=("Чистый экон. эффект (тыс. руб.)", "Затраты план vs факт (тыс. руб.)"))
            fig3.add_trace(go.Bar(x=yrs_eff,
                y=[results[yr].get("Ek") for yr in yrs_eff],
                name="Ek — факт", marker_color="#2e86ab"), row=1, col=1)
            fig3.add_trace(go.Bar(x=yrs_eff,
                y=[results[yr].get("Ep") for yr in yrs_eff],
                name="Ep — план", marker_color="#a8d8ea", opacity=0.7), row=1, col=1)
            fig3.add_trace(go.Scatter(x=yrs_eff,
                y=[results[yr].get("Ck") for yr in yrs_eff],
                name="Затраты факт (Ck)", mode="lines+markers",
                line=dict(color="#e74c3c", width=2)), row=1, col=2)
            fig3.add_trace(go.Scatter(x=yrs_eff,
                y=[results[yr].get("Cp") for yr in yrs_eff],
                name="Затраты план (Cp)", mode="lines+markers",
                line=dict(color="#3498db", width=2, dash="dot")), row=1, col=2)
            fig3.update_layout(template="plotly_white", barmode="group",
                xaxis=dict(type="category"), xaxis2=dict(type="category"))
            st.plotly_chart(fig3, use_container_width=True)

            st.markdown("**Выводы:**")
            for yr in yrs_eff:
                ek = results[yr].get("Ek")
                ep = results[yr].get("Ep")
                ck = results[yr].get("Ck")
                cp = results[yr].get("Cp")
                parts = []
                if ek is not None and ep is not None and ep != 0:
                    ratio = ek / ep
                    parts.append(f"ЧЭЭ факт/план = {ratio:.3f} "
                                  f"({'выше' if ratio>=1 else 'ниже'} плана)")
                if ck is not None and cp is not None:
                    diff = ck - cp
                    sign = "экономия" if diff < 0 else "перерасход"
                    parts.append(f"затраты: {sign} {abs(diff):,.0f} тыс. руб.".replace(",", "\u00a0"))
                if parts:
                    st.markdown(f"- **{yr}**: {'; '.join(parts)}")

            st.markdown("---")

        # ── График 4 ──────────────────────────────────────────────────────────
        last_yr = yrs_with_data[-1]
        r_last = results[last_yr]
        radar_keys   = [f"crit{i}" for i in range(21, 31)]
        radar_labels = ["21: Ek/Ep", "22: MCk/MCp",
                        "23: ACD ΔG", "24: ACD ΔR", "25: ACD ΔHI", "26: ACD ΔCR",
                        "27: MCD ΔG", "28: MCD ΔR", "29: MCD ΔHI", "30: MCD ΔCR"]
        valid = [(lbl, r_last.get(k)) for lbl, k in zip(radar_labels, radar_keys)
                 if r_last.get(k) is not None]

        if len(valid) >= 3:
            st.markdown(f"#### График 4. Радар критериев эффективности — {last_yr}")
            st.info(
                f"📖 **Что показывает:** многокритериальную оценку экономической эффективности "
                f"за {last_yr} год. Каждая ось — отдельный критерий (пп. 5.1–5.6 МР 5.1.0158-19). "
                "**Красный контур** — пороговое значение 1,0. "
                "**Как читать:** чем больше оранжевая область совпадает с красным контуром или "
                "выходит за него — тем выше экономическая эффективность. Оси, уходящие внутрь "
                "красного контура, — критерии, по которым эффективность не достигнута."
            )
            lbls = [v[0] for v in valid] + [valid[0][0]]
            vals = [v[1] for v in valid] + [valid[0][1]]
            fig4 = go.Figure(go.Scatterpolar(
                r=vals, theta=lbls, fill="toself",
                line_color="#e67e22", fillcolor="rgba(230,126,34,0.2)", name=str(last_yr)))
            fig4.add_trace(go.Scatterpolar(
                r=[1.0]*len(lbls), theta=lbls, mode="lines",
                line=dict(color="red", dash="dash"), name="Порог 1,0"))
            fig4.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                template="plotly_white")
            st.plotly_chart(fig4, use_container_width=True)

            above = [lbl for lbl, v in valid if v >= 1.0]
            below = [lbl for lbl, v in valid if v < 1.0]
            st.markdown("**Выводы:**")
            if above:
                st.markdown(f"- ✅ **Критерии ≥ 1,0 (выполнены):** {', '.join(above)}")
            if below:
                st.markdown(f"- ⚠️ **Критерии < 1,0 (не выполнены):** {', '.join(below)}")

# ── Экспорт ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">💾 Экспорт результатов</div>', unsafe_allow_html=True)
rows_exp = []
for yr in YEARS:
    r = results[yr]
    rows_exp.append({
        "Год": yr, "Базовый год": base_year_used,
        "ΔGp": fmt(r.get("ΔGp")), "ΔRp": fmt(r.get("ΔRp")),
        "ΔHIp": fmt(r.get("ΔHIp")), "ΔCRp": fmt_sci(r.get("ΔCRp")),
        "ΔYp": fmt_big(r.get("ΔYp")), "Cp": fmt_big(r.get("Cp")),
        "ΔGk": fmt(r.get("ΔGk")), "ΔRk": fmt(r.get("ΔRk")),
        "ΔHIk": fmt(r.get("ΔHIk")), "ΔCRk": fmt_sci(r.get("ΔCRk")),
        "ΔYk": fmt_big(r.get("ΔYk")), "Ck": fmt_big(r.get("Ck")),
        "Рез.(15)": fmt(r.get("res_G")), "Рез.(16)": fmt(r.get("res_R")),
        "Рез.(17)": fmt(r.get("res_HI")), "Рез.(18)": fmt(r.get("res_CR")),
        "Рез.(19)": fmt(r.get("res_Y")), "Инт.рез.(20)": fmt(r.get("res_int")),
        "Крит.21": fmt(r.get("crit21")), "Крит.22": fmt(r.get("crit22")),
        "Крит.23": fmt(r.get("crit23")), "Крит.24": fmt(r.get("crit24")),
        "Крит.25": fmt(r.get("crit25")), "Крит.26": fmt_sci(r.get("crit26")),
        "Крит.27": fmt(r.get("crit27")), "Крит.28": fmt(r.get("crit28")),
        "Крит.29": fmt(r.get("crit29")), "Крит.30": fmt_sci(r.get("crit30")),
        "Инт.эфф.(31)": fmt(r.get("eff_int")),
    })
csv_data = pd.DataFrame(rows_exp).to_csv(index=False, encoding="utf-8-sig")
st.download_button("⬇️ Скачать результаты (CSV)", data=csv_data,
                   file_name="economic_efficiency_results.csv",
                   mime="text/csv", use_container_width=True)
st.caption("МР 5.1.0158-19 · Федеральный проект «Чистый воздух»")
