---
name: spss-medical-analysis
description: >
  Expert senior en analyse statistique SPSS pour la recherche médicale et biomédicale.
  Utilise cette compétence dès que l'utilisateur mentionne un fichier .sav, .spv, ou toute
  demande d'analyse statistique médicale incluant : statistiques descriptives, t-test, ANOVA,
  régression linéaire ou logistique, analyse de survie (Kaplan-Meier, Cox), corrélations,
  chi², ou toute interprétation de résultats SPSS. Active aussi pour toute demande de
  "rapport statistique", "analyse de données médicales", "résultats de thèse", "interprétation
  SPSS", ou lorsque l'utilisateur veut transformer des données brutes en publication scientifique.
  Produit un fichier Word (.docx) professionnel avec tableaux APA, figures et texte d'interprétation
  de niveau thèse/mémoire universitaire. Ne pas ignorer ce skill pour des analyses simples —
  même une demande comme "analyse mon fichier SPSS" doit utiliser ce skill.
---

# Expert Senior en Analyse SPSS pour la Recherche Médicale

## Vue d'ensemble

Ce skill transforme des fichiers SPSS (.sav) en rapports scientifiques complets au standard
thèse/mémoire universitaire. Il couvre l'ensemble du pipeline analytique :
lecture des données → analyses statistiques → figures → tableaux APA → interprétation rédigée → .docx.

---

## Workflow Principal

```
1. LECTURE     → Charger et explorer le fichier .sav
2. NETTOYAGE   → Vérifier types, valeurs manquantes, outliers
3. ANALYSES    → Choisir et exécuter les tests appropriés
4. VISUALISATION → Générer figures (matplotlib/seaborn)
5. RAPPORT     → Assembler le .docx avec docx-js
6. LIVRAISON   → Présenter le fichier à l'utilisateur
```

---

## Étape 1 — Lecture du Fichier SPSS

### Installation des dépendances

```bash
pip install pyreadstat scipy statsmodels lifelines matplotlib seaborn pandas numpy --break-system-packages
npm install -g docx
```

### Chargement du fichier .sav

```python
import pyreadstat
import pandas as pd
import numpy as np

df, meta = pyreadstat.read_sav("/mnt/user-data/uploads/fichier.sav",
                                apply_value_labels=True)

# Explorer la structure
print(df.shape)
print(df.dtypes)
print(df.describe())
print("Valeurs manquantes:\n", df.isnull().sum())
print("Labels des variables:", meta.column_labels)
```

**Si le fichier .sav n'est pas encore uploadé**, demander à l'utilisateur de le fournir.
**Si c'est un .spv** (output SPSS), lire la note dans `references/spv-reading.md`.

---

## Étape 2 — Diagnostic des Données

Toujours effectuer avant toute analyse :

```python
# Résumé automatique de qualité des données
def data_quality_report(df, meta):
    report = {}
    report['n_observations'] = len(df)
    report['n_variables'] = len(df.columns)
    report['missing'] = df.isnull().sum().to_dict()
    report['missing_pct'] = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    
    # Identifier variables numériques vs catégorielles
    report['numeric_vars'] = df.select_dtypes(include='number').columns.tolist()
    report['categorical_vars'] = df.select_dtypes(exclude='number').columns.tolist()
    
    return report
```

**Règles de décision pour les valeurs manquantes :**
- < 5% : exclure les cas (listwise)
- 5–20% : envisager imputation multiple
- > 20% : signaler à l'utilisateur

---

## Étape 3 — Analyses Statistiques

### 3A. Statistiques Descriptives

```python
from scipy import stats

def descriptives_table(df, variables, group_var=None):
    """Génère un tableau descriptif complet style APA."""
    results = []
    for var in variables:
        col = df[var].dropna()
        row = {
            'Variable': var,
            'N': len(col),
            'Moyenne': round(col.mean(), 2),
            'ÉT': round(col.std(), 2),
            'Médiane': round(col.median(), 2),
            'Min': round(col.min(), 2),
            'Max': round(col.max(), 2),
            'Asymétrie': round(col.skew(), 3),
            'Kurtosis': round(col.kurt(), 3)
        }
        # Test de normalité (Shapiro si n<50, K-S sinon)
        if len(col) < 50:
            stat, p = stats.shapiro(col)
            row['Test normalité'] = f"Shapiro-Wilk W={stat:.3f}, p={p:.3f}"
        else:
            stat, p = stats.kstest(col, 'norm', args=(col.mean(), col.std()))
            row['Test normalité'] = f"K-S D={stat:.3f}, p={p:.3f}"
        results.append(row)
    return pd.DataFrame(results)
```

### 3B. Tests Paramétriques

```python
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- t-test indépendant ---
def run_ttest(df, var, group_var):
    groups = [g[var].dropna() for _, g in df.groupby(group_var)]
    # Vérifier homogénéité des variances (Levene)
    lev_stat, lev_p = stats.levene(*groups)
    equal_var = lev_p > 0.05
    t_stat, p_val = stats.ttest_ind(*groups, equal_var=equal_var)
    # Cohen's d
    n1, n2 = len(groups[0]), len(groups[1])
    pooled_std = np.sqrt(((n1-1)*groups[0].std()**2 + (n2-1)*groups[1].std()**2) / (n1+n2-2))
    cohens_d = (groups[0].mean() - groups[1].mean()) / pooled_std
    return {'t': round(t_stat, 3), 'df': n1+n2-2, 'p': round(p_val, 3),
            "Cohen's d": round(cohens_d, 3), 'Levene p': round(lev_p, 3)}

# --- ANOVA à un facteur ---
def run_anova(df, var, group_var):
    groups = [g[var].dropna().values for _, g in df.groupby(group_var)]
    f_stat, p_val = stats.f_oneway(*groups)
    # Eta-carré
    grand_mean = df[var].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
    eta_sq = ss_between / ss_total
    # Post-hoc Tukey si significatif
    posthoc = None
    if p_val < 0.05:
        tukey = pairwise_tukeyhsd(df[var].dropna(), df.loc[df[var].notna(), group_var])
        posthoc = str(tukey)
    return {'F': round(f_stat, 3), 'p': round(p_val, 3),
            'η²': round(eta_sq, 3), 'Post-hoc': posthoc}
```

### 3C. Régressions

```python
import statsmodels.formula.api as smf

# --- Régression Linéaire Multiple ---
def run_linear_regression(df, outcome, predictors):
    formula = f"{outcome} ~ {' + '.join(predictors)}"
    model = smf.ols(formula, data=df).fit()
    results = {
        'R²': round(model.rsquared, 3),
        'R² ajusté': round(model.rsquared_adj, 3),
        'F': round(model.fvalue, 3),
        'F p-value': round(model.f_pvalue, 4),
        'AIC': round(model.aic, 2),
        'Coefficients': model.summary2().tables[1].round(3)
    }
    return model, results

# --- Régression Logistique ---
def run_logistic_regression(df, outcome, predictors):
    formula = f"{outcome} ~ {' + '.join(predictors)}"
    model = smf.logit(formula, data=df).fit(disp=False)
    # Odds Ratios avec IC 95%
    odds_ratios = pd.DataFrame({
        'OR': np.exp(model.params),
        'IC 95% inf': np.exp(model.conf_int()[0]),
        'IC 95% sup': np.exp(model.conf_int()[1]),
        'p-value': model.pvalues
    }).round(3)
    # Pseudo R² (Nagelkerke)
    n = len(df.dropna(subset=[outcome]+predictors))
    r2_cox = 1 - np.exp(-(model.llf - model.llnull) * 2 / n)
    r2_nagelkerke = r2_cox / (1 - np.exp(2 * model.llnull / n))
    return model, odds_ratios, round(r2_nagelkerke, 3)
```

### 3D. Analyses de Survie

```python
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# --- Kaplan-Meier ---
def run_kaplan_meier(df, duration_col, event_col, group_col=None):
    kmf = KaplanMeierFitter()
    results = {}
    if group_col:
        groups = df[group_col].unique()
        fitters = {}
        for g in groups:
            mask = df[group_col] == g
            kmf_g = KaplanMeierFitter()
            kmf_g.fit(df[mask][duration_col], df[mask][event_col], label=str(g))
            fitters[g] = kmf_g
        # Log-rank test
        if len(groups) == 2:
            g1, g2 = groups
            results['logrank'] = logrank_test(
                df[df[group_col]==g1][duration_col], df[df[group_col]==g2][duration_col],
                df[df[group_col]==g1][event_col],   df[df[group_col]==g2][event_col]
            )
        results['fitters'] = fitters
    else:
        kmf.fit(df[duration_col], df[event_col])
        results['fitter'] = kmf
        results['median_survival'] = kmf.median_survival_time_
    return results

# --- Modèle de Cox ---
def run_cox_model(df, duration_col, event_col, covariates):
    cph = CoxPHFitter()
    cols = [duration_col, event_col] + covariates
    cph.fit(df[cols].dropna(), duration_col=duration_col, event_col=event_col)
    return cph
```

---

## Étape 4 — Génération des Figures

Toujours sauvegarder les figures en PNG haute résolution (300 dpi) avant insertion dans le Word.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Style académique cohérent
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
PALETTE = ['#2E75B6', '#ED7D31', '#70AD47', '#FFC000', '#FF0000']

FIGURES_DIR = "/home/claude/output_figures/"
import os; os.makedirs(FIGURES_DIR, exist_ok=True)

# Voir references/figures-templates.md pour les templates complets de chaque type de figure
```

→ Consulter `references/figures-templates.md` pour les templates complets :
  boîtes à moustaches, histogrammes, barres d'erreur, courbes K-M, forêt plot (Cox), scatter plots.

---

## Étape 5 — Assemblage du Rapport Word

Utiliser le skill **docx** pour générer le `.docx`. Structure standard :

```
Page de titre
Résumé analytique
1. Caractéristiques de l'échantillon (Tableau 1)
2. Analyses bivariées (Tableaux 2–N)
3. Analyses multivariées (Tableaux N+1–M)
4. Analyses de survie (si applicable)
5. Figures (numérotées, avec légendes)
6. Interprétation et discussion
```

→ Consulter `references/docx-report-template.md` pour le code docx-js complet
  incluant styles APA, tableaux formatés, et insertion d'images.

---

## Étape 6 — Règles d'Interprétation (Niveau Thèse)

### Seuils de significativité
| p-value | Mention |
|---------|---------|
| < 0.001 | p < 0,001 |
| < 0.01  | p < 0,01 |
| < 0.05  | p < 0,05 |
| ≥ 0.05  | p = [valeur exacte] (non significatif) |

### Tailles d'effet
| Test | Mesure | Petit | Moyen | Grand |
|------|--------|-------|-------|-------|
| t-test | Cohen's d | 0.2 | 0.5 | 0.8 |
| ANOVA | η² | 0.01 | 0.06 | 0.14 |
| Régression | R² | 0.02 | 0.13 | 0.26 |
| Chi² | V de Cramér | 0.1 | 0.3 | 0.5 |

### Formulation APA des résultats
```
t-test   : t(df) = X.XX, p = .XXX, d = X.XX [IC 95%: X.XX, X.XX]
ANOVA    : F(dfb, dfw) = X.XX, p = .XXX, η² = .XXX
Régression linéaire : β = X.XX, t(df) = X.XX, p = .XXX
Régression logistique : OR = X.XX [IC 95%: X.XX–X.XX], p = .XXX
Log-rank : χ²(1) = X.XX, p = .XXX
Cox HR   : HR = X.XX [IC 95%: X.XX–X.XX], p = .XXX
```

### Rédigé type pour l'interprétation
- **Toujours mentionner** : statistique, degrés de liberté, p-value exacte, taille d'effet
- **Contextualiser** cliniquement : que signifie ce résultat pour la pratique médicale ?
- **Limites à mentionner** : taille d'échantillon, cross-sectional vs longitudinal, biais potentiels
- **Ne jamais écrire** "p = 0.000" → écrire "p < 0,001"

---

## Checklist Qualité Avant Livraison

- [ ] Vérification des hypothèses de chaque test respectée et documentée
- [ ] Valeurs manquantes rapportées dans le texte
- [ ] Tous les tableaux numérotés avec titre **au-dessus** (style APA)
- [ ] Toutes les figures numérotées avec légende **en dessous** (style APA)
- [ ] Unités et décimales cohérentes dans tout le document
- [ ] Tests post-hoc inclus pour toute ANOVA significative
- [ ] IC 95% fournis pour toutes les estimations principales
- [ ] Fichier validé avec le script docx/validate.py

---

## Références

- `references/figures-templates.md` — Code matplotlib complet pour chaque type de figure
- `references/docx-report-template.md` — Template complet du rapport Word en docx-js
- `references/spv-reading.md` — Lecture des fichiers output .spv de SPSS
- `references/assumptions-checks.md` — Guide complet de vérification des hypothèses statistiques
