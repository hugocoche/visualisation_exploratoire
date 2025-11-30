import pandas as pd
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List
from useful_functions import tukey_outlier
from scipy.stats import ttest_ind, mannwhitneyu, levene, f_oneway, kruskal


MAX_POINTS = 5000


def _init_app_quant_qual(
    variables_numeriques: List[str],
    variables_categorielles: List[str],
    title: str = "Analyse quantitative-qualitative",
) -> Dash:
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
                    html.Div(
                        [
                            html.Label(
                                "Variable numérique :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Dropdown(
                                id="num-dropdown",
                                options=[
                                    {"label": c, "value": c}
                                    for c in variables_numeriques
                                ],
                                value=variables_numeriques[0],
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Variable catégorielle :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Dropdown(
                                id="cat-dropdown",
                                options=[
                                    {"label": c, "value": c}
                                    for c in variables_categorielles
                                ],
                                value=variables_categorielles[0],
                                clearable=False,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "180px"},
                    ),
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
                    html.Div(
                        [
                            html.Label(
                                "Nombre de bins pour la heatmap :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Input(
                                id="bins-input",
                                type="number",
                                min=5,
                                max=100,
                                step=1,
                                value=20,
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
                                "Modalités à inclure :",
                                style={"font-weight": "bold", "color": label_color},
                            ),
                            dcc.Dropdown(
                                id="modalities-dropdown",
                                options=[],  # rempli dynamiquement
                                multi=True,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "min-width": "200px"},
                    ),
                ],
            ),
            html.Br(),
            html.Div(
                [
                    dcc.Graph(
                        id="quant-qual-graph",
                        style={
                            "border-radius": "15px",
                            "background-color": card_bg,
                            "box-shadow": "0 4px 8px rgba(0,0,0,0.05)",
                            "padding": "10px",
                            "height": "800px",
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


def compute_group_tests(df, num_var, cat_var):
    """Calcule les tests statistiques entre groupes."""
    modalities = df[cat_var].unique()
    results = []

    if len(modalities) == 2:
        group1 = df[df[cat_var] == modalities[0]][num_var]
        group2 = df[df[cat_var] == modalities[1]][num_var]

        # Test de Levene pour égalité des variances
        stat_lev, p_lev = levene(group1, group2)
        equal_var = p_lev > 0.05

        # Test t (Student ou Welch)
        stat_t, p_t = ttest_ind(group1, group2, equal_var=equal_var)
        results.append(
            {
                "Test": "T-test : Student" if equal_var else "T-test : Welch",
                "Statistique": round(stat_t, 4), #type: ignore
                "p-value": round(p_t, 4), #type: ignore
            }
        )

        # Test de Mann-Whitney
        stat_mw, p_mw = mannwhitneyu(group1, group2, alternative="two-sided")
        results.append(
            {
                "Test": "Mann-Whitney",
                "Statistique": round(stat_mw, 4),
                "p-value": round(p_mw, 4),
            }
        )

    elif len(modalities) > 2:
        groups = [df[df[cat_var] == m][num_var] for m in modalities]

        # ANOVA
        stat_anova, p_anova = f_oneway(*groups)
        results.append(
            {
                "Test": "ANOVA",
                "Statistique": round(stat_anova, 4),
                "p-value": round(p_anova, 4),
            }
        )

        # Kruskal-Wallis
        stat_kw, p_kw = kruskal(*groups)
        results.append(
            {
                "Test": "Kruskal-Wallis",
                "Statistique": round(stat_kw, 4),
                "p-value": round(p_kw, 4),
            }
        )

    return results


def quantitative_qualitative_analysis(
    df: pd.DataFrame,
    variables_numeriques: List[str],
    variables_categorielles: List[str],
    port: int = 8059,
):
    app = _init_app_quant_qual(variables_numeriques, variables_categorielles)

    @app.callback(
        Output("modalities-dropdown", "options"),
        Output("modalities-dropdown", "value"),
        Input("cat-dropdown", "value"),
    )
    def update_modalities_dropdown(cat_var):
        if cat_var is None:
            return [], []
        modalities = df[cat_var].unique()
        options = [{"label": m, "value": m} for m in modalities]
        return options, list(modalities)

    @app.callback(
        Output("quant-qual-graph", "figure"),
        Input("num-dropdown", "value"),
        Input("cat-dropdown", "value"),
        Input("tukey-toggle", "value"),
        Input("bins-input", "value"),
        Input("modalities-dropdown", "value"),
    )
    def update_graph(num_var, cat_var, exclude_outliers, bins, selected_modalities):
        df_plot = df[[num_var, cat_var]].copy()
        if exclude_outliers:
            df_plot = df_plot[
                tukey_outlier(df_plot[num_var], threshold=1.5)["bool_col"]
            ]
        if selected_modalities:
            df_plot = df_plot[df_plot[cat_var].isin(selected_modalities)]
        if len(df_plot) > MAX_POINTS:
            df_plot = df_plot.sample(MAX_POINTS, random_state=42)

        df_plot["num_bin"] = pd.cut(df_plot[num_var], bins=bins)
        df_plot["num_bin_str"] = df_plot["num_bin"].astype(str)

        df_count = (
            df_plot.groupby(["num_bin_str", cat_var]).size().reset_index(name="count")
        )
        df_total = (
            df_count.groupby("num_bin_str")["count"].sum().reset_index(name="total")
        )
        df_count = df_count.merge(df_total, on="num_bin_str")
        df_count["prop"] = df_count["count"] / df_count["total"]  # normalisation

        modalities = df_plot[cat_var].unique()
        colors = px.colors.qualitative.Plotly
        color_map = {mod: colors[i % len(colors)] for i, mod in enumerate(modalities)}

        # Heatmap empilée normalisée
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,  # x différent pour boxplot
            row_heights=[0.7, 0.2],
            vertical_spacing=0.1,
        )

        # Heatmap empilée normalisée
        for mod in modalities:
            df_mod = df_count[df_count[cat_var] == mod]
            fig.add_trace(
                go.Bar(
                    x=df_mod["num_bin_str"],
                    y=df_mod["prop"],
                    name=str(mod),
                    marker_color=color_map[mod],
                    text=[
                        f"{p:.1f}%<br>({n})"
                        for p, n in zip(df_mod["prop"] * 100, df_mod["count"])
                    ],
                    textposition="inside",  # à l’intérieur de la barre
                    insidetextanchor="middle",
                ),
                row=1,
                col=1,
            )
        fig.update_layout(
            barmode="stack",
            legend_title=cat_var,
            xaxis1=dict(title=""),
            yaxis1=dict(title="", showticklabels=False),
        )
        x_min = df_plot[num_var].min()
        x_max = df_plot[num_var].max()
        fig.update_xaxes(range=[x_min, x_max], row=2, col=1)
        
        # Boxplot : une box par modalité
        for mod in modalities:
            df_mod = df_plot[df_plot[cat_var] == mod]
            fig.add_trace(
                go.Box(
                    x=df_mod[num_var],
                    y=[mod] * len(df_mod),  # x = modalité
                    name=str(mod),
                    marker_color=color_map[mod],
                    showlegend=False,
                    orientation="h",
                ),
                row=2,
                col=1,
            )

        fig.update_layout(height=800, plot_bgcolor="white", showlegend=True)
        return fig

    @app.callback(
        Output("stats-output", "children"),
        Input("num-dropdown", "value"),
        Input("cat-dropdown", "value"),
        Input("tukey-toggle", "value"),
        Input("modalities-dropdown", "value"),
    )
    def update_stats(num_var, cat_var, exclude_outliers, selected_modalities):
        df_stats = df[[num_var, cat_var]].copy()
        if exclude_outliers:
            df_stats = df_stats[
                tukey_outlier(df_stats[num_var], threshold=1.5)["bool_col"]
            ]
        if selected_modalities:
            df_stats = df_stats[df_stats[cat_var].isin(selected_modalities)]

        # Tableau descriptif
        desc_table = dash_table.DataTable(
            columns=[
                {"name": c, "id": c}
                for c in df_stats.groupby(cat_var)[num_var]
                .agg(["count", "mean", "median", "std", "min", "max"])
                .reset_index()
                .columns
            ],
            data=df_stats.groupby(cat_var)[num_var]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
            .round(4)
            .to_dict("records"),
            style_cell={"textAlign": "center", "padding": "5px"},
            style_header={"fontWeight": "bold"},
            style_table={"width": "60%", "margin": "auto"},
        )

        # Tableau des tests statistiques
        tests_results = compute_group_tests(df_stats, num_var, cat_var)
        tests_table = dash_table.DataTable(
            columns=[{"name": k, "id": k} for k in tests_results[0].keys()]
            if tests_results
            else [],
            data=tests_results,
            style_cell={"textAlign": "center", "padding": "5px"},
            style_header={"fontWeight": "bold"},
            style_table={"width": "60%", "margin": "auto", "margin-top": "20px"},
        )

        return html.Div([desc_table, html.Br(), tests_table])

    app.run(debug=True, port=port)


if __name__ == "__main__":
    df = pd.read_csv("./car_insurance.csv")
    quantitative_qualitative_analysis(
        df=df,
        variables_categorielles=["is_claim", "steering_type", "model", "transmission_type"],
        variables_numeriques=["age_of_car", "age_of_policyholder"],
    )
