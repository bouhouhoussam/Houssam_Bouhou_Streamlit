import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
COLORS = {"usa": "#ff4343", "japan": "#74c6f5", "europe": "#0868c9"}

st.set_page_config(page_title="Auto MPG | Analyse et prédiction", page_icon="🚗", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df["weight_kg"] = df["weight"] * 0.45359237
    df["consumption_L_100km"] = 235.214583 / df["mpg"]
    df["year"] = df["model_year"] + 1900
    return df

@st.cache_resource
def train_model(data):
    features = ["cylinders", "displacement", "horsepower", "weight_kg", "acceleration", "year", "origin"]
    labeled = data.dropna(subset=["consumption_L_100km"]).copy()
    X = labeled[features]
    y = labeled["consumption_L_100km"]
    numeric = [col for col in features if col != "origin"]
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("origin", OneHotEncoder(handle_unknown="ignore"), ["origin"]),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("forest", RandomForestRegressor(n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return model, mean_absolute_error(y_test, preds), r2_score(y_test, preds)

try:
    df = load_data()
except Exception as exc:
    st.error("Impossible de charger le CSV. Vérifie ta connexion Internet.")
    st.exception(exc)
    st.stop()

st.title("🚗 Exploration des données automobiles")
st.caption("Auto MPG · Analyse descriptive et prédiction par Random Forest Regressor")
st.info("Un **mpg élevé** signifie une meilleure efficacité ; un **L/100 km faible** signifie une consommation moindre.")

with st.sidebar:
    st.header("Filtres")
    origin = st.selectbox("Origine", ["Toutes"] + sorted(df["origin"].dropna().unique().tolist()))
    cylinders = st.multiselect("Nombre de cylindres", sorted(df["cylinders"].dropna().unique().tolist()), default=sorted(df["cylinders"].dropna().unique().tolist()))
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    years = st.slider("Année du modèle", year_min, year_max, (year_min, year_max))
    minimum_mpg = st.slider("Efficacité minimale (mpg)", float(df["mpg"].min()), float(df["mpg"].max()), float(df["mpg"].min()), 0.5)

filtered = df[
    df["cylinders"].isin(cylinders)
    & df["year"].between(*years)
    & (df["mpg"] >= minimum_mpg)
].copy()
if origin != "Toutes":
    filtered = filtered[filtered["origin"] == origin]

tab_analysis, tab_prediction = st.tabs(["📊 Analyse descriptive", "🤖 Prédiction ML"])

with tab_analysis:
    st.header("1 — Indicateurs et données filtrées")
    st.write(f"**{len(filtered)} observations sur {len(df)}**")
    k1, k2, k3 = st.columns(3)
    k1.metric("Efficacité moyenne", f"{filtered['mpg'].mean():.1f} mpg" if not filtered.empty else "—")
    k2.metric("Consommation moyenne", f"{filtered['consumption_L_100km'].mean():.1f} L/100 km" if not filtered.empty else "—")
    k3.metric("Puissance moyenne", f"{filtered['horsepower'].mean():.1f} hp" if not filtered.empty else "—")
    st.caption("Moyennes calculées à partir des données filtrées, en ignorant les valeurs manquantes.")
    columns = ["name", "origin", "year", "cylinders", "mpg", "consumption_L_100km", "horsepower", "weight_kg", "acceleration"]
    st.dataframe(filtered[columns].round(2), use_container_width=True, height=270)

    st.header("2 — Choisir un véhicule")
    if filtered.empty:
        st.warning("Aucun véhicule ne correspond aux filtres. Modifie les critères dans la barre latérale.")
    else:
        vehicle_id = st.selectbox(
            "Véhicule", filtered.index.tolist(),
            format_func=lambda i: f"{df.loc[i, 'name']} — {int(df.loc[i, 'year'])} (ligne {i + 1})",
        )
        st.dataframe(df.loc[[vehicle_id], columns].round(2), hide_index=True, use_container_width=True)

        st.header("3 — Comparaisons graphiques")
        counts = filtered.groupby("origin", as_index=False).size().rename(columns={"size": "Nombre"})
        fig_bar = px.bar(counts, x="origin", y="Nombre", color="origin", title="Nombre d’observations par origine", color_discrete_map=COLORS)
        fig_bar.update_layout(showlegend=False, xaxis_title="Origine")
        st.plotly_chart(fig_bar, use_container_width=True)

        plot_df = filtered.dropna(subset=["weight_kg", "mpg"])
        fig_scatter = px.scatter(
            plot_df, x="weight_kg", y="mpg", color="origin", hover_name="name",
            title="Poids et efficacité énergétique", color_discrete_map=COLORS,
            labels={"weight_kg": "Poids (kg)", "mpg": "Efficacité (mpg)", "origin": "Origine"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        by_year = filtered.groupby("year", as_index=False)["mpg"].mean()
        fig_line = px.line(by_year, x="year", y="mpg", markers=True, title="Efficacité moyenne par année de modèle", labels={"year": "Année", "mpg": "Efficacité moyenne (mpg)"})
        st.plotly_chart(fig_line, use_container_width=True)

        by_origin = filtered.groupby("origin", as_index=False)[["consumption_L_100km", "horsepower"]].mean()
        fig_origin = px.bar(by_origin, x="origin", y="consumption_L_100km", color="origin", title="Consommation moyenne par origine", color_discrete_map=COLORS, labels={"origin": "Origine", "consumption_L_100km": "Consommation (L/100 km)"})
        fig_origin.update_layout(showlegend=False)
        st.plotly_chart(fig_origin, use_container_width=True)

with tab_prediction:
    st.header("4 — Estimer la consommation d’un véhicule")
    st.write("Le modèle est entraîné sur les données historiques **complètes** (pas seulement celles filtrées).")
    model, mae, r2 = train_model(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE sur le jeu de test", f"{mae:.2f} L/100 km")
    c2.metric("R² sur le jeu de test", f"{r2:.3f}")
    c3.metric("Algorithme", "Random Forest")

    with st.form("predict_form"):
        a, b = st.columns(2)
        with a:
            model_origin = st.selectbox("Origine du véhicule", ["usa", "japan", "europe"])
            model_year = st.number_input("Année du modèle", min_value=1970, max_value=2030, value=1980)
            model_cylinders = st.selectbox("Nombre de cylindres", [3, 4, 5, 6, 8], index=1)
            displacement = st.number_input("Cylindrée (cubic inches)", min_value=30.0, max_value=600.0, value=150.0, step=5.0)
        with b:
            horsepower = st.number_input("Puissance (hp)", min_value=20.0, max_value=400.0, value=100.0, step=5.0)
            weight = st.number_input("Poids (kg)", min_value=400.0, max_value=3000.0, value=1200.0, step=20.0)
            acceleration = st.number_input("Accélération (secondes, 0–60 mph)", min_value=5.0, max_value=30.0, value=15.0, step=0.5)
        submitted = st.form_submit_button("🔮 Prédire la consommation", type="primary")

    if submitted:
        new_vehicle = pd.DataFrame([{
            "cylinders": model_cylinders, "displacement": displacement,
            "horsepower": horsepower, "weight_kg": weight,
            "acceleration": acceleration, "year": model_year, "origin": model_origin,
        }])
        prediction = float(model.predict(new_vehicle)[0])
        st.success(f"Consommation estimée : **{prediction:.2f} L/100 km** (≈ {235.214583 / prediction:.1f} mpg)")
        if model_year > 1982:
            st.warning("Attention : les données d’entraînement couvrent uniquement 1970–1982. Les estimations pour des années plus récentes sont des extrapolations moins fiables.")
    st.caption("Projet pédagogique : cette estimation n’est pas une mesure réelle de consommation.")
