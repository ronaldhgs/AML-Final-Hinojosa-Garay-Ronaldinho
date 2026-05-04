# Proyecto Final – Tasa personalizada por perfil mediante IA

## CENTRUM PUCP · Deep Learning · Módulo 4

---

### Descripción del proyecto

Este proyecto desarrolla un modelo predictivo que genera una **tasa de interés (TEA) personalizada** para cada cliente de Préstamos con Garantía Hipotecaria (PGH), reemplazando el tarifario estático tradicional por uno dinámico basado en Machine Learning y Deep Learning.

Se utilizó un dataset sintético de 3,800 contratos con 160 variables que incluyen perfil del cliente, características del préstamo e inmueble, y comportamiento de pago mes a mes (análisis de cosechas).

### Objetivos

- Predecir la tasa TEA óptima ajustada por riesgo para cada perfil individual.
- Comparar el rendimiento entre modelos de Machine Learning y Deep Learning.
- Demostrar la superioridad del pricing dinámico sobre el tarifario estático por segmentos.
- Analizar la interpretabilidad del modelo mediante SHAP.

### Modelos implementados

1. **XGBoost** (Gradient Boosting) — Baseline de Machine Learning
2. **Red Neuronal Profunda (DNN)** — Con BatchNormalization, Dropout y regularización L2
3. **DNN con Embeddings Categóricos** — Representaciones aprendidas de variables como zona, sector laboral y tipo de inmueble

Se aplicó ingeniería de features sobre las cosechas de pago (promedios de atraso, volatilidad, tendencias) y variables de interacción (score × LTV, score × DTI).

### Resultados

Comparación de modelos:

| Modelo | MAE (pp) | RMSE (pp) | R² | MAPE |
|--------|----------|-----------|------|------|
| XGBoost | 0.3705 | 0.4786 | 0.9485 | 3.10% |
| DNN Simple | 0.5558 | 0.6978 | 0.8905 | 4.55% |
| DNN + Embeddings | 0.5320 | 0.6621 | 0.9014 | 4.36% |

**XGBoost** obtuvo el mejor desempeño, consistente con la literatura para datos tabulares con volúmenes moderados. El tarifario dinámico reduce el error en un **74.8%** frente al enfoque estático por segmentos (MAE de 2.11 pp → 0.37 pp).

### Interpretabilidad

Se utilizó **SHAP** con XGBoost para analizar la importancia de las variables. Las features más determinantes fueron: score crediticio, LTV, comportamiento de atraso temprano, ratio deuda/ingreso y zona geográfica.

### Estructura del repositorio

```
├── notebooks/
│   └── final_project.ipynb      # Notebook principal ejecutable
├── data/
│   └── pgh_dataset_sintetico.csv # Dataset sintético (Google Drive)
├── README.md                     # Este archivo
```

### Tecnologías utilizadas

- Python 3.12
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- SHAP
- Matplotlib, Seaborn
- Pandas, NumPy

### Posibles mejoras

- Aumentar el volumen del dataset para potenciar los modelos de Deep Learning.
- Incorporar variables macroeconómicas externas (tasa de referencia BCRP, inflación).
- Hyperparameter tuning más exhaustivo con Optuna o Bayesian Optimization.
- Implementar A/B testing para medir impacto real en conversión y margen.
- Aplicar técnicas de fairness para evitar sesgos en la asignación de tasas.

### Conclusión

El proyecto demuestra que un tarifario dinámico basado en IA supera significativamente al enfoque estático, permitiendo asignar tasas personalizadas que reflejan el riesgo real de cada perfil. XGBoost resultó el modelo más preciso para este volumen de datos, aunque las redes neuronales con embeddings mostraron capacidad de capturar relaciones complejas entre variables categóricas, con potencial de escalar mejor ante datasets más grandes.

---

**Alumno:** Hinojosa Garay, Ronaldiñho  
**Programa de Especialización en IA aplicada a los negocios:** Módulo de Deep Learning – CENTRUM PUCP  
**Fecha:** Mayo 2026
