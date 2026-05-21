# Regresión Lasso — Predicción de Enfermedades Cardíacas por Condado

> Pipeline de regresión sobre datos sociodemográficos y de salud a nivel de condado en EE.UU.: poda intensa de características en un dataset amplio, prefiltrado con SelectKBest y un modelo Lasso (regularización L1) que pone a cero los predictores débiles — demostrando por qué la regularización importa cuando las características superan a la señal.

---

## Problema

Se han recopilado datos sociodemográficos y de recursos sanitarios a nivel de condado en todo Estados Unidos (2018–2019). El objetivo es determinar si existe una relación significativa entre factores sociodemográficos — pobreza, educación, distribución de edad, raza, empleo — y la carga de enfermedades cardíacas. El target es un recuento bruto: total de personas por condado diagnosticadas con enfermedades cardíacas.

## Dataset

- **Fuente:** Dataset de salud demográfica a nivel de condado de EE.UU. (2018–2019)
- **Target:** `Heart disease_number` — total de casos de enfermedades cardíacas por condado (continuo)
- **Forma original:** Dataset amplio con ~50+ columnas que incluyen demografía de población, ingresos, educación, empleo, distribución racial y prevalencia/recuentos de enfermedades

## Pipeline de EDA y Preprocesamiento

**Paso 1 — Poda agresiva de columnas** antes de cualquier análisis:

| Categoría eliminada | Motivo |
|---|---|
| `Heart disease_prevalence`, límites de IC | Fuga de datos — derivada del target |
| Recuentos brutos y prevalencia de otras enfermedades | Variables resultado, no predictores |
| Todas las columnas de límites de IC | Redundantes con las estimaciones puntuales |
| Recuentos brutos de población por edad/raza | Se retienen los equivalentes en % |
| Recuentos brutos de empleo/pobreza | Se retienen los equivalentes en tasa/% |

| Paso | Acción |
|---|---|
| Duplicados | Ninguno encontrado |
| Manejo de nulos | Filas con target nulo eliminadas; nulos restantes rellenados con la mediana de la columna |
| Capping de outliers | Método IQR en TOT_POP, MEDHHINC_2018, GQ_ESTIMATES_2018 |
| Columnas de texto | Cualquier columna de tipo objeto restante (p. ej. nombre del condado) eliminada |
| Escalado | MinMaxScaler en todas las columnas de características |
| Selección de características | SelectKBest (f_regression, k=15) — k generoso porque Lasso realiza una poda adicional durante el entrenamiento |
| División | 80/20 entrenamiento/prueba |

**Principales predictores identificados (por correlación con el target):**

| Característica | Dirección |
|---|---|
| TOT_POP (población del condado) | Fuerte positiva — los condados más grandes tienen más casos por definición |
| Obesity_prevalence | Positiva |
| % Black-alone | Positiva |
| PCTPOVALL_2018 (tasa de pobreza) | Positiva |
| 80+ y/o % of total pop | Positiva |
| MEDHHINC_2018 (ingreso medio del hogar) | Negativa |
| % con grado universitario | Negativa |
| Médicos activos por 100k | Negativa |

## Modelo

**Dos modelos comparados:**

| Modelo | Descripción |
|---|---|
| `LinearRegression` | Línea base — sin regularización |
| `Lasso(alpha=1.0)` | Regularización L1 — reduce los coeficientes débiles exactamente a cero |

Lasso es la elección correcta para este dataset porque: (1) el conjunto de características es amplio en relación a la señal, (2) muchos indicadores sociodemográficos están correlacionados entre sí, y (3) la regularización L1 produce un modelo disperso que identifica automáticamente qué predictores son genuinamente útiles.

El modelo Lasso entrenado se guarda en `models/lasso_alpha-1.0.sav`.

## Conclusiones Clave

- **La población no es una característica — es un factor de confusión:** TOT_POP debe incluirse porque un condado de 1 millón de habitantes siempre tendrá más casos de enfermedades cardíacas que uno de 5.000, independientemente de los factores sociodemográficos. Sin él, el modelo aprende el tamaño del condado, no los factores de riesgo de salud.
- **La fuga de datos en datasets amplios está por todas partes:** La prevalencia de enfermedades cardíacas, los límites de IC y las columnas de recuentos derivados tenían que eliminarse antes de cualquier análisis — codifican la respuesta en lugar de predecirla.
- **Regularización L1 vs L2:** Lasso (L1) lleva los coeficientes débiles a cero, produciendo un modelo disperso e interpretable. Ridge (L2) reduce todos los coeficientes pero los mantiene. Cuando el objetivo es identificar qué factores sociodemográficos importan, la dispersión de Lasso es más útil.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_LinReg_Regularization_MTK.git
cd ML_LinReg_Regularization_MTK
pip install -r requirements.txt
python src/app.py
```

## Próximos Pasos

- Optimizar `alpha` con validación cruzada (`LassoCV`) en lugar de usar un valor fijo de 1.0 — la fuerza de regularización óptima depende de los datos
- Ingenierizar un target **per cápita** (`heart_disease_number / TOT_POP`) para estudiar los factores de riesgo de salud independientemente del tamaño de la población
- Comparar Lasso con **ElasticNet** (híbrido L1 + L2) para ver si combinar ambas penalizaciones mejora el R² fuera de muestra

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
