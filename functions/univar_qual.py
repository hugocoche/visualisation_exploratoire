import pandas as pd
from typing import Literal
import plotly.graph_objects as go
import plotly.io as pio
import dash
from dash import dcc, html, Input, Output

pio.renderers.default = "notebook"


def init_app(
    data: pd.DataFrame,
    title: str = "Analyse visuelle des variables qualitative",
    height: str = "auto",
    width: str = "100%",
    margin: str = "auto",
    font_family: str = "Arial",
    gap: str = "20px",
) -> dash.Dash:
    """
    Initialise l'application Dash pour l'analyse des variables qualitatives avec un design amélioré,
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
                                "Nombre de modalités :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.Input(
                                id="mods-input",
                                type="number",
                                min=1,
                                max=50,
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
                                "Orientation :",
                                style={
                                    "font-weight": "bold",
                                    "color": control_label_color,
                                },
                            ),
                            dcc.RadioItems(
                                id="orientation-toggle",
                                options=[
                                    {"label": "Horizontal", "value": "h"},
                                    {"label": "Vertical", "value": "v"},
                                ],
                                value="h",
                                inline=True,
                                inputStyle={"margin-right": "5px"},
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
                        id="graph-bar",
                        style={
                            "border-radius": "15px",
                            "background-color": card_bg,
                            "box-shadow": "0 4px 8px rgba(0,0,0,0.05)",
                            "padding": "10px",
                        },
                    )
                ]
            ),
        ],
    )

    return app


def qualitative_analysis(
    data: pd.DataFrame,
    title: str = "Analyse visuelle des variables qualitatives",
    height: str = "400px",
    width: str = "100%",
    margin: str = "auto",
    font_family="Arial",
    port: int = 8057,
    gap: str = "40px",
):
    """Permet d'effectuer une représentation synthétique des variables qualitatives
    Veiller à intégrer un dataframe qui ne contient que des colonnes de type booléen ou catégorielle
    """

    app = init_app(
        data=data,
        title=title,
        height=height,
        width=width,
        margin=margin,
        font_family=font_family,
        gap=gap,
    )

    @app.callback(
        Output("graph-bar", "figure"),
        Input("var-dropdown", "value"),
        Input("mods-input", "value"),
        Input("orientation-toggle", "value")
    )
    def update_graph(variable: str, nmods: int, orientation: Literal["h", "v"]):
        counts = data[variable].value_counts().sort_index()
        x = counts.index
        y = counts.to_numpy(dtype=float) / float(counts.sum())

        tab = pd.DataFrame({"index": x, "values": y}).sort_values(
            "values", ascending=False if orientation=="v" else True
        )
        tab = tab.head(nmods)

        if orientation == "h":
            x = tab["values"]
            y = tab["index"]
        else:
            x = tab["index"]
            y = tab["values"]

        bar_text = [f"{v:.1%}" for v in tab["values"]]
        trace_bar = go.Bar(
            x=x,
            y=y,
            orientation=orientation,
            marker=dict(color="rgba(0,0,180,0.4)", line=dict(color="white", width=1)),
            showlegend=False,
            text=bar_text,
            textposition="auto",
            hoverinfo="skip",
        )

        fig = go.Figure()
        fig.add_trace(trace_bar)

        if orientation == "v":
            fig.update_xaxes(
                title="",
                type="category",
                tickfont_size=10,
                ticks="outside",
                tickcolor="black",
                tickwidth=1,
                ticklen=2,
            )
            fig.update_yaxes(showticklabels=False, ticks="")
        else:
            fig.update_yaxes(
                title="",
                type="category",
                tickfont_size=10,
                ticks="outside",
                tickcolor="black",
                tickwidth=1,
                ticklen=2,
            )
            fig.update_xaxes(showticklabels=False, ticks="")


        fig.update_layout(
            plot_bgcolor="white",
            hoverlabel=dict(bgcolor="white", align="left"),
            bargap=0.03,
        )

        return fig

    app.run(debug=True, port=port)


if __name__ == "__main__":
    df = pd.read_csv("./car_insurance.csv")
    df_cat = df[["is_esc", "is_claim", "model"]]
    qualitative_analysis(df_cat)
