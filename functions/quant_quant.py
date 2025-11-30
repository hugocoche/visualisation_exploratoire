"""
quant_quant.py

Analyse exploratoire bivariée quantitative avec Dash et Plotly.

Fonctionnalités :
- Scatter plot avec choix des axes X et Y
- Option pour afficher une régression polynomiale (choix du degré)
- Gestion des outliers via la méthode de Tukey
- Statistiques descriptives : R², Pearson, Spearman, Kendall, coefficients et p-values
- Option pour afficher une KDE bidimensionnelle avec choix du kernel
- Limitation du nombre de points pour optimiser l'affichage
"""

import pandas as pd
import numpy as np
from typing import Literal
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from dash import dash_table
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KernelDensity
from useful_functions import tukey_outlier

MAX_POINTS = 5000


def _init_app(data: pd.DataFrame, title: str = "Analyse bivariée quantitative") -> Dash:
    """
    Initialise l'application Dash avec le style "carte".
    """
    app = Dash(__name__)
    bg_color = "#f9f9f9"
    card_bg = "#ffffff"
    label_color = "#333333"

    app.layout = html.Div(
        style={
            "width": "100%",
            "margin": "auto",
            "font-family": "Arial",
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
            # Contrôles
            html.Div(
                style={
                    "display": "flex",
                    "flex-wrap": "wrap",
                    "gap": "20px",
                    "background-color": card_bg,
                    "padding": "20px",
                    "border-radius": "15px",
                    "box-shadow": "0 4px 8px rgba(0,0,0,0.1)",
                },
                children=[
                    # Choix des axes
                    html.Div(
                        [
                            html.Label(
                                "Abscisse (X) :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Dropdown(
                                id="x-dropdown",
                                options=[
                                    {"label": c, "value": c} for c in data.columns
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
                                "Ordonnée (Y) :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Dropdown(
                                id="y-dropdown",
                                options=[
                                    {"label": c, "value": c} for c in data.columns
                                ],
                                value=data.columns[1]
                                if len(data.columns) > 1
                                else data.columns[0],
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
                    # Options de régression
                    html.Div(
                        [
                            html.Label(
                                "Afficher régression :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.RadioItems(
                                id="regression-toggle",
                                options=[
                                    {"label": "Oui", "value": True},
                                    {"label": "Non", "value": False},
                                ],
                                value=False,
                                inline=True,
                                inputStyle={"margin-right": "5px"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "150px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Degré polynomial :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Input(
                                id="poly-degree",
                                type="number",
                                min=1,
                                max=5,
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
                    # Outliers
                    html.Div(
                        [
                            html.Label(
                                "Exclure outliers (Tukey) :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.RadioItems(
                                id="tukey-toggle",
                                options=[
                                    {"label": "Oui", "value": True},
                                    {"label": "Non", "value": False},
                                ],
                                value=True,
                                inline=True,
                                inputStyle={"margin-right": "5px"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "150px"},
                    ),
                    # KDE
                    html.Div(
                        [
                            html.Label(
                                "Afficher KDE :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.RadioItems(
                                id="kde-toggle",
                                options=[
                                    {"label": "Oui", "value": True},
                                    {"label": "Non", "value": False},
                                ],
                                value=False,
                                inline=True,
                                inputStyle={"margin-right": "5px"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "150px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Kernel KDE :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Dropdown(
                                id="kernel-dropdown",
                                options=[
                                    {"label": k.capitalize(), "value": k}
                                    for k in [
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
            # Graphique
            html.Div(
                [
                    dcc.Graph(
                        id="scatter-graph",
                        style={
                            "border-radius": "15px",
                            "background-color": card_bg,
                            "box-shadow": "0 4px 8px rgba(0,0,0,0.05)",
                            "padding": "10px",
                            "height": "600px",
                        },
                    )
                ]
            ),
            html.Br(),
            # Tableau des stats
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


def add_kde(
    fig,
    x,
    y,
    kernel: Literal[
        "gaussian", "tophat", "linear", "epanechnikov", "exponential", "cosine"
    ] = "gaussian",
    n_points: int = 100,
):
    """
    Ajoute une KDE bidimensionnelle au scatter plot.
    """
    X_grid = np.linspace(x.min(), x.max(), n_points)
    Y_grid = np.linspace(y.min(), y.max(), n_points)
    XX, YY = np.meshgrid(X_grid, Y_grid)
    positions = np.vstack([XX.ravel(), YY.ravel()]).T

    kde = KernelDensity(bandwidth=0.5, kernel=kernel)
    kde.fit(np.vstack([x, y]).T)
    Z = np.exp(kde.score_samples(positions)).reshape(n_points, n_points)

    fig.add_contour(
        x=X_grid, y=Y_grid, z=Z, colorscale="Reds", opacity=0.5, showscale=False
    )


def quantitative_bivariate_analysis(
    data: pd.DataFrame,
    type: Literal["heatmap", "scatter"] = "scatter",
    port: int = 8058,
):
    """
    Lancement de l'analyse bivariée quantitative.
    """
    app = _init_app(data)

    # Callback principal pour mise à jour du graphique
    @app.callback(
        Output("scatter-graph", "figure"),
        Input("x-dropdown", "value"),
        Input("y-dropdown", "value"),
        Input("regression-toggle", "value"),
        Input("poly-degree", "value"),
        Input("tukey-toggle", "value"),
        Input("kde-toggle", "value"),
        Input("kernel-dropdown", "value"),
    )
    def update_graph(
        x_col, y_col, show_regression, degree, exclude_outliers, show_kde, kernel
    ):
        df = data[[x_col, y_col]].copy()
        if exclude_outliers:
            for col in [x_col, y_col]:
                df_out = tukey_outlier(df[col], threshold=1.5)
                df = df[df_out["bool_col"]]

        if len(df) > MAX_POINTS:
            df = df.sample(MAX_POINTS, random_state=42)

        # Scatter plot avec marginals
        if type == "scatter":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                marginal_x="histogram",
                marginal_y="histogram",
                opacity=0.2,
                color_discrete_sequence=["blue"],
                labels={x_col: x_col, y_col: y_col},
            )
        else:
            fig = px.density_heatmap(
                df,
                x=x_col,
                y=y_col,
                nbinsx=40,
                nbinsy=40,
                color_continuous_scale="Viridis",
                labels={x_col: x_col, y_col: y_col},
            )
        fig.update_xaxes(
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.1)",
            griddash="dash",
        )
        fig.update_yaxes(
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(0,0,0,0.1)",
            griddash="dash",
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white", hovermode="closest")

        # Régression polynomiale
        if show_regression and len(df) > 1:
            X_poly = PolynomialFeatures(degree).fit_transform(df[[x_col]])
            model = sm.OLS(df[y_col], X_poly).fit()

            x_lin = np.linspace(df[x_col].min(), df[x_col].max(), 200)
            X_lin_poly = PolynomialFeatures(degree).fit_transform(x_lin.reshape(-1, 1))
            y_lin_pred = model.predict(X_lin_poly)

            fig.add_traces(
                px.line(x=x_lin, y=y_lin_pred, color_discrete_sequence=["red"])
                .update_traces(showlegend=False)
                .data
            )

        # KDE bidimensionnelle
        if show_kde and len(df) > 1:
            add_kde(fig, df[x_col], df[y_col], kernel=kernel)

        return fig

    # Callback pour le tableau des statistiques
    @app.callback(
        Output("stats-output", "children"),
        Input("x-dropdown", "value"),
        Input("y-dropdown", "value"),
        Input("poly-degree", "value"),
        Input("tukey-toggle", "value"),
    )
    def update_stats(x_col, y_col, degree, exclude_outliers):
        df = data[[x_col, y_col]].copy()
        if exclude_outliers:
            for col in [x_col, y_col]:
                df_out = tukey_outlier(df[col], threshold=1.5)
                df = df[df_out["bool_col"]]

        x = df[[x_col]]
        y = df[y_col]

        # Modèle polynomial
        X_poly = PolynomialFeatures(degree).fit_transform(x)
        model = sm.OLS(y, X_poly).fit()

        # Construction du tableau des coefficients
        table_data = [
            {
                "Coefficient": "alpha (intercept)",
                "Valeur": f"{model.params[0]:.3f}",
                "p-value": f"{model.pvalues[0]:.4f}",
            }
        ]
        for i in range(1, degree + 1):
            table_data.append(
                {
                    "Coefficient": f"beta_{i}",
                    "Valeur": f"{model.params[i]:.3f}",
                    "p-value": f"{model.pvalues[i]:.4f}",
                }
            )

        # Statistiques corrélationnelles
        r2 = model.rsquared
        pearson_corr, pearson_p = pearsonr(x.values.ravel(), y)
        spearman_corr, spearman_p = spearmanr(x.values.ravel(), y)
        kendall_corr, kendall_p = kendalltau(x.values.ravel(), y)

        table_data += [
            {"Coefficient": "R²", "Valeur": f"{r2:.3f}", "p-value": "-"},
            {
                "Coefficient": "Pearson r",
                "Valeur": f"{pearson_corr:.3f}",
                "p-value": f"{pearson_p:.4f}",
            },
            {
                "Coefficient": "Spearman r",
                "Valeur": f"{spearman_corr:.3f}",
                "p-value": f"{spearman_p:.4f}",
            },
            {
                "Coefficient": "Kendall tau",
                "Valeur": f"{kendall_corr:.3f}",
                "p-value": f"{kendall_p:.4f}",
            },
        ]

        return dash_table.DataTable(
            columns=[
                {"name": "Coefficient", "id": "Coefficient"},
                {"name": "Valeur", "id": "Valeur"},
                {"name": "p-value", "id": "p-value"},
            ],
            data=table_data,  # type: ignore
            style_cell={"textAlign": "center", "padding": "5px"},
            style_header={"fontWeight": "bold"},
            style_table={"width": "60%", "margin": "auto"},
            style_data_conditional=[  # type: ignore
                {
                    "if": {"filter_query": "{p-value} < 0.05", "column_id": "p-value"},
                    "color": "red",
                    "fontWeight": "bold",
                }
            ],
        )

    app.run(debug=True, port=port)


if __name__ == "__main__":
    df = pd.read_csv("./car_insurance.csv")
    df_numeric = df[
        ["policy_tenure", "age_of_car", "age_of_policyholder", "population_density"]
    ]
    quantitative_bivariate_analysis(df_numeric, type="scatter")
