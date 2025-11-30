import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import dash
from sklearn.neighbors import KernelDensity
from typing import Literal
from dash import dcc, html, Input, Output
from useful_functions import tukey_outlier, variation_coeff, yule_coeff, nice_range

pio.renderers.default = "notebook"


def _init_app(
    data: pd.DataFrame,
    title: str = "Analyse visuelle des variables quantitatives",
    height: str = "auto",
    width: str = "100%",
    margin: str = "auto",
    font_family: str = "Arial",
    gap: str = "20px",
) -> dash.Dash:
    """
    Initialise l'application Dash pour l'analyse des variables quantitatives avec un design amélioré,
    """

    app = dash.Dash(__name__)

    bg_color = "#f9f9f9"
    card_bg = "#ffffff"
    control_label_color = "#333333"

    app.layout = html.Div(
        style={
            "width": width,
            "margin": margin,
            "font-family": font_family,
            "height": height,
            "background-color": bg_color,
            "padding": "20px",
        },
        children=[
            html.H2(
                title,
                style={
                    "text-align": "center",
                    "margin-bottom": "30px",
                    "color": "#222222",
                },
            ),
            html.Div(
                style={
                    "display": "flex",
                    "flex-wrap": "wrap",
                    "gap": gap,
                    "background-color": card_bg,
                    "padding": "20px",
                    "border-radius": "15px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
                },
                children=[
                    html.Div(
                        [
                            html.Label(
                                "Variable :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.Dropdown(
                                id="var-dropdown",
                                options=[
                                    {"label": v, "value": v} for v in data.columns
                                ],
                                value=data.columns[0],
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Annotation :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.Dropdown(
                                id="vline-dropdown",
                                options=[
                                    {"label": v, "value": v}
                                    for v in ["Non", "Ligne", "Ligne+Texte"]
                                ],
                                value="Non",
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Nombre de bins :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.Input(
                                id="bins-input",
                                type="number",
                                min=1,
                                max=500,
                                step=1,
                                value=15,
                                style={
                                    "width": "100%",
                                    "padding": "5px",
                                    "border-radius": "5px",
                                    "border": "1px solid #ccc",
                                },
                            ),
                        ],
                        style={"flex": "1", "min-width": "150px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Précision des bornes :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.Input(
                                id="precision-input",
                                type="number",
                                min=0,
                                max=10,
                                step=1,
                                value=1,
                                style={
                                    "width": "100%",
                                    "padding": "5px",
                                    "border-radius": "5px",
                                    "border": "1px solid #ccc",
                                },
                            ),
                        ],
                        style={"flex": "1", "min-width": "150px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Afficher les outliers (Tukey) :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.RadioItems(
                                id="tukey-outlier-toggle",
                                options=[
                                    {"label": "Oui", "value": True},
                                    {"label": "Non", "value": False},
                                ],
                                value=True,
                                inline=True,
                                inputStyle={"margin-right": "5px"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Normalisation de l'histogramme :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.Dropdown(
                                id="histnorm-dropdown",
                                options=[
                                    {
                                        "label": "Effectif"
                                        if v is None
                                        else v.capitalize(),
                                        "value": "" if v is None else v,
                                    }
                                    for v in [None, "probability", "density"]
                                ],
                                value="probability",
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Kernel Density :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.Dropdown(
                                id="kernel-dropdown",
                                options=[
                                    {"label": v.capitalize(), "value": v}
                                    for v in [
                                        "gaussian",
                                        "tophat",
                                        "epanechnikov",
                                        "exponential",
                                        "linear",
                                        "cosine",
                                    ]
                                ],
                                value="gaussian",
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
                ],
            ),
            html.Br(),
            html.Div(
                [
                    dcc.Graph(
                        id="graph-hist-box",
                        style={
                            "border-radius": "15px",
                            "background-color": card_bg,
                            "box-shadow": "0 4px 8px rgba(0,0,0,0.05)",
                            "padding": "10px",
                        },
                    )
                ]
            ),
            html.Br(),
            html.Div(
                id="stats-output",
                style={
                    "background-color": card_bg,
                    "padding": "15px",
                    "border-radius": "10px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.05)",
                    "font-size": "14px",
                    "line-height": "1.6",
                },
            ),
        ],
    )

    return app


def adding_vline(fig: go.Figure, annotation: str, resume: pd.Series) -> None:
    """
    Cette fonction permet à l'utilisateur de choisir s'il veut annoter son graphique avec des statistiques descriptives
    """
    if annotation == "Non":
        add_vline = False
        add_vline_annotation = False
    elif annotation == "Ligne":
        add_vline = True
        add_vline_annotation = False
    elif annotation == "Ligne+Texte":
        add_vline = True
        add_vline_annotation = True

    if add_vline:
        fig.add_vline(x=resume["min"], line_dash="dash", line_width=1, opacity=0.3)
        fig.add_vline(x=resume["25%"], line_dash="dash", line_width=1, opacity=0.3)
        fig.add_vline(
            x=resume["50%"],
            line_dash="dash",
            line_width=1,
            line_color="green",
            opacity=0.3,
        )
        fig.add_vline(
            x=resume["mean"],
            line_dash="dash",
            line_width=1,
            line_color="red",
            opacity=0.3,
        )
        fig.add_vline(x=resume["75%"], line_dash="dash", line_width=1, opacity=0.3)
        fig.add_vline(x=resume["max"], line_dash="dash", line_width=1, opacity=0.3)

        if add_vline_annotation:
            fig.add_annotation(
                x=resume["min"],
                y=1,
                yshift=20,
                yref="paper",
                text=f"Min={round(resume['min'],4)}",
                showarrow=False,
            )
            fig.add_annotation(
                x=resume["25%"],
                y=1,
                yshift=20,
                yref="paper",
                text=f"Q1={round(resume['25%'],4)}",
                showarrow=False,
            )
            fig.add_annotation(
                x=resume["50%"],
                y=1,
                yshift=-20,
                xshift=-40,
                yref="paper",
                text=f"Med={round(resume['50%'],4)}",
                showarrow=False,
            )
            fig.add_annotation(
                x=resume["mean"],
                y=1,
                yshift=-20,
                xshift=40,
                yref="paper",
                text=f"Mean={round(resume['mean'],4)}",
                showarrow=False,
            )
            fig.add_annotation(
                x=resume["75%"],
                y=1,
                yshift=20,
                yref="paper",
                text=f"Q3={round(resume['75%'],4)}",
                showarrow=False,
            )
            fig.add_annotation(
                x=resume["max"],
                y=1,
                yshift=20,
                yref="paper",
                text=f"Max={round(resume['max'],4)}",
                showarrow=False,
            )


def adding_kde(
    fig: go.Figure,
    series: pd.DataFrame,
    kernel: Literal[
        "gaussian",
        "tophat",
        "epanechnikov",
        "exponential",
        "linear",
        "cosine",
    ],
    histnorm: Literal[None, "probability", "density"],
    min_val: float,
    max_val: float,
    x_kde: int,
    bin_size: float,
):
    """
    Permet l'affichage d'une courbe kde
    """
    kde = KernelDensity(kernel=kernel, bandwidth=0.1).fit(series.values.reshape(-1, 1))
    x_d = np.linspace(min_val, max_val, x_kde)
    log_dens = kde.score_samples(x_d.reshape(-1, 1))
    dens = np.exp(log_dens)

    if histnorm in [None, ""]:
        dens = dens * len(series) * bin_size
    elif histnorm == "probability":
        dens = dens * bin_size
    elif histnorm == "density":
        dens = dens
    else:
        raise ValueError(
            f"histnorm '{histnorm}' non supporté pour la normalisation de la KDE"
        )

    trace_kde = go.Scatter(
        x=x_d,
        y=dens,
        mode="lines",
        line=dict(color="red", width=2),
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_trace(trace_kde, row=1, col=1)


def quantitative_analysis(
    data: pd.DataFrame,
    title: str = "Analyse visuelle des quantitative",
    height: str = "400px",
    width: str = "100%",
    margin: str = "auto",
    font_family="Arial",
    port: int = 8057,
    gap: str = "40px",
    add_kde: bool = False,
    x_kde: int = 50,
):
    """Permet d'effectuer une représentation synthétique des variables quantitatives
    Veiller à intégrer un dataframe qui ne contient que des colonnes de type quantitatif
    """

    app = _init_app(
        data=data,
        title=title,
        height=height,
        width=width,
        margin=margin,
        font_family=font_family,
        gap=gap,
    )

    @app.callback(
        Output("graph-hist-box", "figure"),
        Input("var-dropdown", "value"),
        Input("vline-dropdown", "value"),
        Input("bins-input", "value"),
        Input("precision-input", "value"),
        Input("tukey-outlier-toggle", "value"),
        Input("histnorm-dropdown", "value"),
        Input("kernel-dropdown", "value"),
    )
    def update_graph(
        variable: str,
        annotation: str,
        nbins: int,
        precision: int,
        tukey: bool,
        histnorm: Literal[None, "probability", "density"],
        kde: Literal[
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "cosine",
        ],
    ):
        if tukey:
            series = data[variable]
        else:
            data_with_bool_tukey = tukey_outlier(data[variable], threshold=1.5)
            series = data_with_bool_tukey[data_with_bool_tukey["bool_col"]][variable]

        resume = series.describe()
        min_val, max_val = nice_range(series, precision=precision)
        bin_size = round((max_val - min_val) / nbins, 2)
        bin_edges = np.linspace(min_val, max_val, nbins + 1)
        interval_labels = [
            f"{bin_edges[i]:.2f} ; {bin_edges[i+1]:.2f}"
            for i in range(len(bin_edges) - 1)
        ]

        trace_hist = go.Histogram(
            x=series,
            xbins=dict(start=min_val, end=max_val, size=bin_size),
            nbinsx=nbins,
            histnorm=histnorm,
            marker=dict(color="rgba(0,0,180,0.4)", line=dict(color="white", width=1)),
            showlegend=False,
            texttemplate="%{y:.1%}" if histnorm == "probability" else "%{y:2.f}",
            customdata=interval_labels,
            hovertemplate=(
                "Intervalle : [%{customdata}]<br>" "%{y:.2%}<br>" "<extra></extra>"
            )
            if histnorm == "probability"
            else "Intervalle : [%{customdata}]<br>" "%{y:.2f}<br>" "<extra></extra>",
        )

        trace_box = go.Box(
            x=series,
            marker=dict(color="rgba(0,0,180,0.4)", line=dict(color="white", width=1)),
            name="",
            showlegend=False,
            width=(resume["max"] - resume["min"]),
        )

        fig = make_subplots(
            rows=2, cols=1, vertical_spacing=0.1, row_heights=[0.9, 0.1]
        )
        fig.add_trace(trace_hist, row=1, col=1)
        fig.add_trace(trace_box, row=2, col=1)

        if add_kde:
            adding_kde(
                fig=fig,
                kernel=kde,
                series=pd.DataFrame(series),
                histnorm=histnorm,
                min_val=min_val,
                max_val=max_val,
                x_kde=x_kde,
                bin_size=bin_size,
            )

        adding_vline(fig, annotation, resume)

        tickvals = np.arange(min_val, max_val + bin_size, bin_size)
        fig.update_xaxes(
            row=1,
            col=1,
            title="",
            range=[resume["min"], resume["max"]],
            tickfont_size=8,
            ticks="outside",
            tickcolor="black",
            tickwidth=1,
            ticklen=2,
            tickvals=tickvals,
        )

        fig.update_xaxes(range=[resume["min"], resume["max"]], row=2, col=1)
        fig.update_yaxes(
            row=1,
            col=1,
            title="Fréquence"
            if histnorm == "probability"
            else "Densité"
            if histnorm == "density"
            else "Effectif",
            showticklabels=False,
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.5)",
        )

        fig.update_layout(
            plot_bgcolor="white",
            hoverlabel=dict(bgcolor="white", align="left"),
            bargap=0.03,
        )

        return fig

    @app.callback(Output("stats-output", "children"), Input("var-dropdown", "value"))
    def update_stats(variable: str):
        series = data[variable]
        return [
            html.P(f"Coefficient de Yule = {yule_coeff(series)}"),
            html.P(f"Coefficient de variation = {variation_coeff(series)}"),
            html.P(f"Skewness = {series.skew():.3f}"),
            html.P(f"Kurtosis = {series.kurt():.3f}"),
        ]

    app.run(debug=True, port=port)


if __name__ == "__main__":
    df = pd.read_csv("./car_insurance.csv")
    df_numeric = df[["policy_tenure", "age_of_car"]]
    quantitative_analysis(df_numeric, add_kde=True)
