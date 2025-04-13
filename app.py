#bibliotheques:

import streamlit as st
import numpy as np
import pandas as pd
from yahooquery import Ticker
import yfinance as yf
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re


st.title("Analyse et Optimisation ESG de notre Fonds ")

st.write("""
Bienvenue sur notre outil d’analyse ESG.
Dans ce projet, nous avons choisi de partir d’un fonds d’investissement déjà existant afin d’évaluer sa performance selon des critères environnementaux, sociaux et de gouvernance (ESG).
Il s'agit du fonds indiciel coté iShares MSCI World SRI UCITS ETF, conçu pour offrir aux investisseurs une exposition aux actions mondiales respectant des critères ESG rigoureux.


Notre objectif est double :
1. **Mesurer le score ESG moyen du fonds dans son ensemble** pour avoir une première idée de sa performance globale.
2. **Analyser le score ESG moyen par secteur d’activité** pour identifier les secteurs les plus vertueux, mais aussi ceux qui pénalisent le score global.


Enfin, dans un troisième temps, en repérant le ou les secteurs ayant les scores ESG les plus faibles, nous pourrons proposer des ajustements dans la composition du portefeuille afin d’**améliorer son profil ESG** tout en maintenant une diversification cohérente.
Par ailleurs, cette analyse nous permettra de challenger les hypothèses du fonds choisi.
""")

st.markdown("<b><u>1ère partie : Analyse globale de notre fonds :</u></b>", unsafe_allow_html=True)

# Upload du fichier 
uploaded_file = st.file_uploader("Chargez votre fichier Excel (.xlsm)", type=["xls", "xlsx", "xlsm"]) 

if uploaded_file is not None: 
    try: 
        df_positions = pd.read_excel(uploaded_file, engine='openpyxl', skiprows=7) 
        st.success("✅ Fichier chargé avec succès !") 
        st.subheader("Aperçu des données") 
        st.dataframe(df_positions.head()) 
    except Exception as e:    
        st.error(f"Erreur lors de la lecture du fichier : {e}")


# Nettoyage
df_clean = df_positions[['Ticker', 'Nom', 'Secteur', 'Lieu', 'Pondération (%)']].dropna()


# Filtres
st.sidebar.header("Filtres d'affichage")


secteurs = df_clean['Secteur'].unique().tolist()
secteurs_selectionnes = st.sidebar.multiselect("Secteurs :", secteurs, default=secteurs)


lieux = df_clean['Lieu'].unique().tolist()
lieux_selectionnes = st.sidebar.multiselect("Pays :", lieux, default=lieux)


df_filtré = df_clean[
    (df_clean['Secteur'].isin(secteurs_selectionnes)) &
    (df_clean['Lieu'].isin(lieux_selectionnes))
]


if df_filtré.empty:
    st.warning("Aucune donnée correspondante aux filtres sélectionnés.")
else:


    st.subheader("Répartition par Pays 🌍")
    # Regrouper les pays avec moins de 1% sous "Autres"
    pays = df_filtré.groupby('Lieu')['Pondération (%)'].sum().reset_index()
    seuil = 1  # seuil en pourcentage


# Séparer les pays majeurs et mineurs
    pays_majeurs = pays[pays['Pondération (%)'] >= seuil]
    pays_mineurs = pays[pays['Pondération (%)'] < seuil]


# Ajouter ligne "Autres"
    if not pays_mineurs.empty:
        autres = pd.DataFrame({
            'Lieu': ['Autres'],
            'Pondération (%)': [pays_mineurs['Pondération (%)'].sum()]
        })
        pays = pd.concat([pays_majeurs, autres], ignore_index=True)


# Pie chart avec regroupement
    fig_pays = px.pie(
        pays,
        values='Pondération (%)',
        names='Lieu',
        title='Répartition géographique du portefeuille',
        hole=0.3
    )
    fig_pays.update_traces(textinfo='percent')
    st.plotly_chart(fig_pays, use_container_width=True)
    st.write("Les entreprises américaines sont nettement sur représentées.")


    st.subheader("Répartition par Secteur 🏭")
    secteur = df_filtré.groupby('Secteur')['Pondération (%)'].sum().reset_index()
    fig_secteur = px.pie(secteur, values='Pondération (%)', names='Secteur', title='Répartition par secteur', hole=0.3)
    st.plotly_chart(fig_secteur, use_container_width=True)


    st.write("On observe une assez grande diversité quant à la répartition sectorielle.")


# === Analyse ESG globale et par secteur ===
st.markdown("<b><u>1ère partie : Analyse globale ESG :</u></b>", unsafe_allow_html=True)


df = pd.read_excel(uploaded_file, engine='openpyxl', skiprows=7) 


tickers = df.iloc[:, 0].dropna().unique().tolist()
tickers_valides = [str(t).strip() for t in tickers if isinstance(t, str)]  # Vérifie que c'est une chaîne de caractères
tickers_clean = [ticker.strip().upper() for ticker in tickers_valides]
# Filtrer les suffixes comme ":xpar"
tickers_clean = [re.sub(r':.*', '', ticker) for ticker in tickers_clean]



try:
    @st.cache_data
    def get_esg_scores(tickers):
        ticker_data = Ticker(tickers)
        return ticker_data.esg_scores
    esg_scores_data = get_esg_scores(tickers)
except Exception as e:
    st.error(f"Erreur lors de la récupération des scores ESG : {e}")
    esg_scores_data = {}

    # Vérifier chaque ticker et récupérer son score ESG
    for ticker in tickers_clean:
        esg_data = get_esg_score(ticker)
        if esg_data is not None:
            esg_scores[ticker] = esg_data
        else:
            st.warning(f"⚠️ Aucun score ESG récupéré pour {ticker}")
        
    # Si aucun score ESG n'a été récupéré
    if not esg_scores:
        st.warning("⚠️ Aucun score ESG récupéré pour les tickers disponibles.")
    else:
        st.write("Scores ESG récupérés :")
        st.write(esg_scores)


esg_data_dict = {}


for ticker in tickers:
    esg_info = esg_scores_data.get(ticker, None)
    if isinstance(esg_info, dict):
        esg_score = esg_info.get("totalEsg")
        e_score = esg_info.get("environmentScore")
        s_score = esg_info.get("socialScore")
        g_score = esg_info.get("governanceScore")


        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            rendement = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100 if not hist.empty else None
        except Exception:
            rendement = None


        if esg_score is not None:
            esg_data_dict[ticker] = {
                "esg": esg_score,
                "e": e_score,
                "s": s_score,
                "g": g_score,
                "rendement": rendement
            }


if not esg_data_dict:
    st.warning("⚠️ Aucun score ESG récupéré pour les tickers disponibles.")
    st.stop()


esg_scores = [data["esg"] for data in esg_data_dict.values()]
moyenne_esg = sum(esg_scores) / len(esg_scores)
st.success(f"**Score ESG moyen du fonds : {moyenne_esg:.2f}**")


def safe_average(lst):
    clean = [x for x in lst if x is not None]
    return sum(clean) / len(clean) if clean else None


secteurs = df.iloc[:, 2].dropna().unique()
secteurs_scores = []


for secteur in secteurs:
    tickers_du_secteur = df[df.iloc[:, 2] == secteur].iloc[:, 0].dropna()
    tickers_du_secteur = [str(t).strip() for t in tickers_du_secteur]


    esg_scores, e_scores, s_scores, g_scores, rendements = [], [], [], [], []


    for ticker in tickers_du_secteur:
        if ticker in esg_data_dict:
            d = esg_data_dict[ticker]
            esg_scores.append(d.get("esg"))
            e_scores.append(d.get("e"))
            s_scores.append(d.get("s"))
            g_scores.append(d.get("g"))
            if d.get("rendement") is not None:
                rendements.append(d.get("rendement"))


    if esg_scores:
        secteurs_scores.append([
            secteur,
            safe_average(esg_scores),
            safe_average(e_scores),
            safe_average(s_scores),
            safe_average(g_scores),
            safe_average(rendements),
            len([x for x in esg_scores if x is not None])
        ])


secteurs_df = pd.DataFrame(
    secteurs_scores,
    columns=["Secteur", "Score ESG moyen", "Score E moyen", "Score S moyen", "Score G moyen", "Rendement moyen (%)", "Nombre d'entreprises"]
)


st.dataframe(secteurs_df)


st.subheader("Classement des secteurs selon le score ESG")


# Trie des secteurs du meilleur au pire (du plus bas au plus haut ESG)
secteurs_df_sorted = secteurs_df.sort_values(by="Score ESG moyen", ascending=True)


# Création du graphique Plotly
fig_secteurs = px.bar(
    secteurs_df_sorted,
    x="Score ESG moyen",
    y="Secteur",
    orientation='h',
    color="Score ESG moyen",
    color_continuous_scale="RdYlGn_r",
    text=secteurs_df_sorted["Score ESG moyen"].apply(lambda x: f"{x:.2f}")
)


fig_secteurs.update_layout(
    title_text=None,
    xaxis_title="Score ESG moyen",
    yaxis_title="Secteur",
    title_x=0.5,
    height=400
)


fig_secteurs.update_traces(textposition='outside')


st.plotly_chart(fig_secteurs, use_container_width=True)




st.write("""
Cette première analyse du fonds nous permet de réaliser une première conclusion.
Bien que le fonds iShares MSCI World SRI UCITS ETF mette en avant des critères ESG exigeants, son approche demeure largement perfectible. La composition du portefeuille reste dominée par de grandes multinationales généralistes comme Microsoft, Apple ou Nestlé, qui, malgré de bonnes notations ESG, sont régulièrement critiquées pour leurs impacts environnementaux et sociaux, tels que la consommation excessive de ressources ou des pratiques de chaîne d’approvisionnement controversées.
Si le fonds applique bien une logique d’exclusion des secteurs et des acteurs les plus problématiques, il ne cible pas spécifiquement des entreprises à impact positif. Il se limite à un filtrage relatif plutôt qu'à une véritable dynamique de transformation vers un modèle durable.
Par ailleurs, la stratégie du fonds repose essentiellement sur des notations externes, souvent fondées sur des données auto-déclarées ou incomplètes. Cela soulève des interrogations quant à la fiabilité et à la profondeur réelle de l’analyse ESG effectuée.
L’absence d’alignement significatif avec la taxonomie verte européenne et le manque d’engagement direct auprès des entreprises renforcent encore ces limites. Le fonds se contente de répliquer passivement son indice de référence sans dialogue actif avec les sociétés en portefeuille.
Finalement, malgré ses ambitions affichées, ce fonds semble davantage répondre à une logique de conformité réglementaire et d’image que s’inscrire dans une véritable stratégie d’investissement durable et de soutien actif à la transition écologique.
""")


if not secteurs_df.empty:
    secteurs_df_sorted = secteurs_df.sort_values(by="Score ESG moyen", ascending=True)
    meilleur_secteur = secteurs_df_sorted.iloc[0]      # Score ESG le plus bas
    pire_secteur = secteurs_df_sorted.iloc[-1]         # Score ESG le plus haut


    st.markdown("<b><u>2ème partie : Analyse sectorielle :</u></b>", unsafe_allow_html=True)
    st.write(f"✅ **Secteur le plus vertueux** : **{meilleur_secteur['Secteur']}** avec un score ESG moyen de **{meilleur_secteur['Score ESG moyen']:.2f}**")
  
    st.write(""" Bien que ne représentant qu'une très faible proportion dans le fonds, le secteur immobilier est souvent perçu comme l’un des plus vertueux en matière ESG car il dispose de leviers d’amélioration très concrets et immédiats.
           L’efficacité énergétique des bâtiments peut être améliorée rapidement grâce à des rénovations, l’utilisation de matériaux durables et l’adoption de normes environnementales reconnues (comme HQE, BREEAM ou LEED).
           De plus, le cadre réglementaire est de plus en plus strict, notamment en Europe, où les réglementations thermiques et environnementales imposent des seuils ambitieux de performance énergétique.
           Les entreprises du secteur ont ainsi tout intérêt à s’adapter pour rester compétitives et attirer des investisseurs sensibles aux critères ESG.
           Par ailleurs, les impacts sociaux et de gouvernance sont généralement moins exposés que dans d’autres secteurs : la chaîne d’approvisionnement est plus locale et maîtrisée, et les risques de controverses majeures (droits humains, travail forcé) sont plus faibles que dans les industries extractives ou manufacturières.""")
  
    st.write(f"⚠️ **Secteur le moins vertueux** : **{pire_secteur['Secteur']}** avec un score ESG moyen de **{pire_secteur['Score ESG moyen']:.2f}**")


    st.write(""" Le secteur de l’énergie, à l’inverse, est structurellement désavantagé dans les notations ESG du fait de sa dépendance historique aux énergies fossiles.
            Même si de nombreux acteurs commencent à investir dans les énergies renouvelables, le cœur de leur activité repose encore largement sur l’extraction, la transformation et la distribution de pétrole, de gaz et de charbon — activités fortement émettrices de gaz à effet de serre.
            La transition énergétique dans ce secteur est complexe, coûteuse et progressive, ce qui limite à court terme l'amélioration des performances ESG.
            En parallèle, le secteur est confronté à de nombreuses controverses, qu’elles soient environnementales (pollutions, marées noires, atteintes à la biodiversité) ou sociales (atteintes aux droits des populations locales, conditions de travail dans les zones d’extraction).
            Enfin, certains grands groupes énergétiques continuent de financer des activités ou de mener des actions de lobbying défavorables à la transition écologique, ce qui pénalise encore davantage leurs scores ESG.""")


   # Fonction d'analyse détaillée secteur
    def analyse_secteur(titre, secteur_cible, phrase_personnalisee):
        st.markdown(f"### 🔍 Détail des entreprises du secteur **{secteur_cible}**")
        entreprises = df[df.iloc[:, 2] == secteur_cible]
        entreprises_scores = []


        for _, row in entreprises.iterrows():
            ticker = str(row[0]).strip()
            if ticker in esg_data_dict:
                d = esg_data_dict[ticker]
                entreprises_scores.append([
                    ticker,
                    d.get("esg"),
                    d.get("e"),
                    d.get("s"),
                    d.get("g"),
                    d.get("rendement"),
                    row['Lieu'] 
                ])


        entreprises_df = pd.DataFrame(
            entreprises_scores,
            columns=["Entreprise", "Score ESG", "Score E", "Score S", "Score G", "Rendement sur 1 an (%)", "Pays"]
        )


        st.dataframe(entreprises_df)


        if not entreprises_df.empty:
            entreprises_df_sorted = entreprises_df.sort_values(by="Score ESG", ascending=True)
            fig = px.bar(
                entreprises_df_sorted,
                x="Score ESG",
                y="Entreprise",
                orientation='h',
                color="Score ESG",
                color_continuous_scale="RdYlGn_r",
                title=f"{titre} — Score ESG par entreprise (plus le score est bas, meilleure est la performance ESG)",
                category_orders={'Entreprise': entreprises_df_sorted['Entreprise'].tolist()}
            )    
            st.plotly_chart(fig, use_container_width=True)


            # ✅ Phrase personnalisée
            st.markdown(f"📝 {phrase_personnalisee}")


           # ✅ Décomposition E/S/G du secteur
            score_e_moy = entreprises_df['Score E'].mean()
            score_s_moy = entreprises_df['Score S'].mean()
            score_g_moy = entreprises_df['Score G'].mean()


            st.markdown(f"### 🧩 Décomposition du score ESG global du secteur **{secteur_cible}**")
            st.write(f"- **Score E (Environnemental)** : {score_e_moy:.2f}")
            st.write(f"- **Score S (Social)** : {score_s_moy:.2f}")
            st.write(f"- **Score G (Gouvernance)** : {score_g_moy:.2f}")
            st.write("A noter que le score ESG total n'est pas une moyenne des trois scores. Cela s'explique par la pondération différente des facteurs et l'intégration d'autres éléments tels que le niveau de controverse qui peuvent augmenter ou diminuer le score total.​")
  


           # ✅ Camembert répartition géographique
            st.markdown(f"### 🌍 Répartition géographique des entreprises du secteur **{secteur_cible}**")
            pays_repartition = entreprises_df['Pays'].value_counts().reset_index()
            pays_repartition.columns = ['Pays', 'Nombre d\'entreprises']
            fig_pie = px.pie(
                pays_repartition,
                values='Nombre d\'entreprises',
                names='Pays',
                title=f"Répartition géographique du secteur {secteur_cible}",
                hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)


   # Analyse du secteur le plus vertueux
    analyse_secteur(
        "🌟 Secteur le plus vertueux",
        meilleur_secteur['Secteur'],
        "L'entreprise ayant la meilleure performance extra financière est le groupe CBRE avec un score ESG de 8,96. "
        "A l'inverse, la moins bonne performance extra financière est réalisée par l'entreprise canadienne FirstService avec un score ESG de 15,96 "
        "soit une meilleure performance que la 'meilleure' entreprise du secteur de l'Énergie."
    )


   # Analyse du secteur le moins vertueux
    analyse_secteur(
        "🚨 Secteur le moins vertueux",
        pire_secteur['Secteur'],
        "L'entreprise ayant la meilleure performance extra financière est Baker Hughes avec un score ESG de 19,12. "
        "A l'inverse, la moins bonne performance extra financière est réalisée par l'entreprise américaine Phillips Energy avec un score ESG de 35,92."
    )


else:
    st.warning("Aucun secteur disponible pour l'analyse.")






# Reconstruction du portefeuille personnalisé




st.markdown("<b><u>3ème partie : Simulation de reconstruction du portefeuille</u></b>", unsafe_allow_html=True)
st.markdown("Les deux premières parties du projet nous ont permis de mieux comprendre le fonds et notamment d'observer le secteur le plus performant ainsi que le moins performant."
        
"Dans cette dernière partie, nous utilisons nos observations précédentes afin de construire un portefeuille plus performant vis-à-vis des critères ESG. "




"Si le secteur Énergie est choisi, nous laissons la possibilité d'exclure certaines entreprises. Nous avons fait la même chose pour le secteur des matériaux qui regroupe plus d'entreprises et qui est le deuxième secteur de notre fonds ayant le pire score ESG "




"Enfin, nous comparons ce nouveau portefeuille avec la performance ESG du fonds de départ. ")




# Initialisation de la session state
if 'secteurs_selectionnes' not in st.session_state:
    st.session_state['secteurs_selectionnes'] = []




if 'entreprises_selectionnees_energie' not in st.session_state:
    st.session_state['entreprises_selectionnees_energie'] = []




# 📋 FORMULAIRE
with st.form("form_portefeuille"):
  # 1️⃣ Sélection des secteurs
    secteurs_disponibles = df['Secteur'].dropna().unique().tolist()
    secteurs_selectionnes = st.multiselect(
        "Sélectionnez les secteurs que vous souhaitez inclure dans votre portefeuille :",
        secteurs_disponibles,
        default=st.session_state['secteurs_selectionnes']
    )


  # Sauvegarde temporaire dans la session
    st.session_state['secteurs_selectionnes'] = secteurs_selectionnes


  # 2️⃣ Si "Énergie" est dans les secteurs sélectionnés, afficher la sélection des entreprises
    secteur_energie = next((sect for sect in secteurs_selectionnes if "ener" in sect.lower()), None)
    secteur_materiaux = next((sect for sect in secteurs_selectionnes if "matériaux" in sect.lower() or "materiaux" in sect.lower()), None)


    entreprises_selectionnees_energie = []
    entreprises_selectionnees_materiaux = []


    if secteur_energie:
        entreprises_energie = df[df['Secteur'] == secteur_energie].iloc[:, 0].dropna().unique().tolist()
        entreprises_selectionnees_energie = st.multiselect(
            f"Sélectionnez les entreprises à inclure dans le secteur {secteur_energie} :",
            entreprises_energie,
            default=entreprises_energie
        )
        st.session_state['entreprises_selectionnees_energie'] = entreprises_selectionnees_energie


    if secteur_materiaux:
        entreprises_materiaux = df[df['Secteur'] == secteur_materiaux].iloc[:, 0].dropna().unique().tolist()
        entreprises_selectionnees_materiaux = st.multiselect(
            f"Sélectionnez les entreprises à inclure dans le secteur {secteur_materiaux} :",
            entreprises_materiaux,
            default=entreprises_materiaux
        )
        st.session_state['entreprises_selectionnees_materiaux'] = entreprises_selectionnees_materiaux


  # 📍 Bouton de validation du formulaire
    submit = st.form_submit_button("Construire le portefeuille")


# 🚀 Calcul uniquement après soumission du formulaire
if submit:


    if not secteurs_selectionnes:
        st.warning("⚠️ Veuillez sélectionner au moins un secteur pour construire le portefeuille.")
    else:
        entreprises_selectionnees = []


      # Collecte des entreprises des autres secteurs (hors énergie et matériaux)
        autres_secteurs = [sect for sect in secteurs_selectionnes if sect not in [secteur_energie, secteur_materiaux]]
        entreprises_autres = df[df['Secteur'].isin(autres_secteurs)].iloc[:, 0].dropna().unique().tolist()


      # Finalisation des entreprises
        entreprises_selectionnees.extend(entreprises_autres)
        entreprises_selectionnees.extend(st.session_state.get('entreprises_selectionnees_energie', []))
        entreprises_selectionnees.extend(st.session_state.get('entreprises_selectionnees_materiaux', []))


      # Filtrage final
        df_personnalise = df[df.iloc[:, 0].isin(entreprises_selectionnees)]


        if df_personnalise.empty:
            st.warning("⚠️ Aucun titre sélectionné dans votre portefeuille personnalisé.")
        else:
            esg_personnalise = []
            for _, row in df_personnalise.iterrows():
                ticker = str(row[0]).strip()
                if ticker in esg_data_dict:
                    d = esg_data_dict[ticker]
                    esg_personnalise.append({
                        "Ticker": ticker,
                        "Secteur": row['Secteur'],
                        "Score ESG": d.get("esg"),
                        "Score E": d.get("e"),
                        "Score S": d.get("s"),
                        "Score G": d.get("g"),
                        "Rendement (%)": d.get("rendement")
                    })


            df_esg_personnalise = pd.DataFrame(esg_personnalise)


            if df_esg_personnalise.empty:
                st.warning("⚠️ Aucun score ESG disponible pour les entreprises sélectionnées.")
            else:
                st.markdown("### 🌿 Performance du portefeuille personnalisé")


                score_esg_moyen = df_esg_personnalise['Score ESG'].mean()
                score_e_moyen = df_esg_personnalise['Score E'].mean()
                score_s_moyen = df_esg_personnalise['Score S'].mean()
                score_g_moyen = df_esg_personnalise['Score G'].mean()
                rendement_moyen = df_esg_personnalise['Rendement (%)'].mean()


                st.success(f"**Score ESG moyen du portefeuille personnalisé : {score_esg_moyen:.2f}**")
                st.write(f"- Score E moyen : {score_e_moyen:.2f}")
                st.write(f"- Score S moyen : {score_s_moyen:.2f}")
                st.write(f"- Score G moyen : {score_g_moyen:.2f}")
                if rendement_moyen is not None:
                    st.write(f"- Rendement moyen : {rendement_moyen:.2f}%")


                # Comparaison avec le portefeuille initial
                st.markdown("### ⚖️ Comparaison avec le portefeuille initial")
                ecart_esg = score_esg_moyen - moyenne_esg
                if ecart_esg < 0:
                    st.success(f"✅ Amélioration ESG de {abs(ecart_esg):.2f} points par rapport au portefeuille initial.")
                else:
                    st.warning(f"⚠️ Dégradation ESG de {ecart_esg:.2f} points par rapport au portefeuille initial.")



# Comparaison avec le portefeuille initial
                ecart_esg = score_esg_moyen - moyenne_esg
                rendement_moyen = rendement_moyen if rendement_moyen is not None else 0  # Gérer les valeurs manquantes

                # Affichage des résultats calculés (déjà sur le doc) 
                st.success(f"**Score ESG moyen du portefeuille personnalisé : {score_esg_moyen:.2f}**")
                st.write(f"- Score E moyen : {score_e_moyen:.2f}")
                st.write(f"- Score S moyen : {score_s_moyen:.2f}")
                st.write(f"- Score G moyen : {score_g_moyen:.2f}")
                st.write(f"- Rendement moyen : {rendement_moyen:.2f}%")

                # Commentaires
            st.markdown(f"""
            Dans cette dernière étape, vous avez choisi de composer votre portefeuille à partir des secteurs {', '.join(secteurs_selectionnes)}.
                
            Cette personnalisation a conduit à un portefeuille affichant un score ESG moyen de **{score_esg_moyen:.2f}**, soit une {'légère dégradation' if ecart_esg > 0 else 'amélioration'} de **{abs(ecart_esg):.2f}** points par rapport au portefeuille initial.
                
            Le rendement moyen obtenu est de **{rendement_moyen:.2f}%**, ce qui reste {'en dessous' if rendement_moyen < 0 else 'au-dessus'} du portefeuille optimal identifié précédemment, démontrant que le choix des entreprises, même au sein de secteurs pertinents, peut impacter significativement la performance financière tout en ne garantissant pas nécessairement une amélioration ESG.
                
            De manière plus générale, notre analyse met en évidence une tension persistante entre performance financière et durabilité. Alors que certains secteurs parviennent à combiner scores ESG élevés et rendements solides, d’autres affichent de très bonnes performances financières malgré de faibles scores ESG. Cette dualité souligne l’importance d’un arbitrage éclairé entre impact durable et objectifs de rentabilité. La construction d’un portefeuille optimal nécessite donc non seulement une sélection sectorielle rigoureuse, mais aussi une analyse fine au niveau des entreprises pour concilier ces deux dimensions.
            """)

