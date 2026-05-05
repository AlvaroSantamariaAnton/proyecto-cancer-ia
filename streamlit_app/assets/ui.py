"""
streamlit_app/assets/ui.py
==========================
Utilidades visuales compartidas por todas las páginas del dashboard.

Incluye:
  - Carga del CSS y configuración base
  - Paleta de colores corporativa
  - Helpers para tarjetas KPI, indicadores de riesgo y secciones
  - Configuración de Plotly con tema claro consistente
"""
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go


# === Paleta de colores clínica ===
COLORS = {
    "primary":       "#0066CC",   # Azul clínico
    "primary_dark":  "#004C99",
    "primary_light": "#3385D6",
    "secondary":     "#006D77",   # Teal médico
    "background":    "#FFFFFF",
    "surface":       "#F5F8FB",
    "border":        "#E2E8F0",
    "text":          "#1A202C",
    "text_muted":    "#64748B",
    "success":       "#10B981",
    "warning":       "#F59E0B",
    "danger":        "#DC2626",
    "info":          "#3B82F6",
    # Para gráficos comparativos de modelos
    "model_lr":      "#0066CC",
    "model_rf":      "#10B981",
    "model_xgb":     "#F59E0B",
    "model_lgb":     "#8B5CF6",
    "model_mlp":     "#DC2626",
}

# Plotly: tema claro consistente
PLOTLY_THEME = {
    "layout": {
        "font": {"family": "Inter, sans-serif", "color": COLORS["text"], "size": 12},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "colorway": [COLORS["model_lr"], COLORS["model_rf"], COLORS["model_xgb"],
                     COLORS["model_lgb"], COLORS["model_mlp"]],
        "xaxis": {"gridcolor": "#E2E8F0", "zerolinecolor": "#CBD5E1",
                  "linecolor": "#CBD5E1"},
        "yaxis": {"gridcolor": "#E2E8F0", "zerolinecolor": "#CBD5E1",
                  "linecolor": "#CBD5E1"},
        "legend": {"bgcolor": "rgba(255,255,255,0.95)", "bordercolor": "#E2E8F0",
                   "borderwidth": 1},
        "margin": {"t": 50, "r": 20, "b": 50, "l": 60},
    }
}


def load_css() -> None:
    """Carga el CSS personalizado del dashboard."""
    css_path = Path(__file__).parent / "styles.css"
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def app_header(title: str, subtitle: str = "") -> None:
    """Header consistente al inicio de cada página."""
    st.markdown(
        f'''
        <div class="app-header">
            <h1>{title}</h1>
            {f'<div class="subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        ''',
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    """Título de sección con estilo clínico."""
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str = "", delta_kind: str = "neutral") -> None:
    """
    Tarjeta KPI con estilo de panel clínico.

    Parameters
    ----------
    label : etiqueta superior (ej. "PACIENTES EVALUADOS")
    value : valor principal grande (ej. "10.001")
    delta : texto secundario opcional (ej. "+2.3% vs val")
    delta_kind : 'neutral', 'positive', 'negative'
    """
    delta_class = f"kpi-delta {delta_kind}" if delta else ""
    delta_html = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    st.markdown(
        f'''
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        ''',
        unsafe_allow_html=True,
    )


def risk_indicator(level: str, probability: float) -> None:
    """
    Indicador visual de nivel de riesgo (Alto / Moderado / Bajo).

    level : 'Alto', 'Moderado' o 'Bajo'
    probability : probabilidad [0-1]
    """
    css_class = f"risk-{level.lower()}"
    st.markdown(
        f'''
        <div class="risk-indicator {css_class}">
            <div class="risk-label">Nivel de riesgo estimado</div>
            <div class="risk-value">{level} &middot; {probability*100:.1f}%</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def factor_chip(text: str, severity: str) -> str:
    """
    Devuelve el HTML de un chip para un factor de riesgo.

    severity : 'alta', 'media', 'baja', 'info', 'protector'
    """
    return f'<span class="factor-chip chip-{severity}">{text}</span>'


def medical_disclaimer() -> None:
    """Disclaimer estándar en software clínico."""
    st.markdown(
        '''
        <div class="medical-disclaimer">
            <strong>Aviso:</strong> Este sistema es una herramienta de apoyo a la decisión clínica.
            Sus resultados deben ser interpretados por un profesional sanitario cualificado.
            No sustituye al juicio clínico ni a las pruebas diagnósticas confirmatorias.
        </div>
        ''',
        unsafe_allow_html=True,
    )


def info_card(text: str) -> None:
    """Tarjeta informativa neutra."""
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Aplica el tema claro corporativo a un gráfico de Plotly."""
    fig.update_layout(**PLOTLY_THEME["layout"])
    return fig