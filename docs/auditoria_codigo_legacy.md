# Auditoría del código legacy

## Alcance

Este documento compara `legacy/modelo_gnn_siames.py` con la arquitectura
objetivo del repositorio.

Reglas de interpretación:

- el archivo legacy se trata como referencia histórica;
- no debe presentarse como implementación definitiva;
- la presencia de una función o rama no demuestra por sí sola que esté
  correctamente entrenada, conectada al optimizador o alineada con la
  especificación;
- varias secciones dependen del orden de ejecución de notebook, variables
  globales y rutas de Colab, por lo que parte del comportamiento es solo
  parcialmente verificable desde lectura estática.

## Observaciones generales

- El archivo está ya ubicado correctamente en `legacy/`.
- Tiene aproximadamente 26k líneas y conserva sintaxis de notebook como
  `!pip`, por lo que no es importable como módulo Python limpio sin
  preprocesado.
- Mezcla entrenamiento, auditoría de HDF5, exportación, clustering, análisis
  de ablaciones y parches post hoc en un único archivo.
- Usa rutas específicas de Colab y Google Drive, por ejemplo
  `/content/drive/MyDrive/GNN_siames_colab`.
- Depende de muchas variables globales y helpers definidos en celdas previas.

## Tabla comparativa

| componente | estado | ubicación en el código | comportamiento actual | diferencia respecto a la especificación | riesgo | posibilidad de reutilización | prioridad de migración |
|---|---|---|---|---|---|---|---|
| lectura y validación HDF5 | existente | `validate_hdf5_file`, `load_graph_safe`, `list_*`, `read_graph_globals` en `legacy/modelo_gnn_siames.py:1186`, `1498`, `1411`, `1435` | abre HDF5, valida grupos, construye `x`, `edge_index`, `edge_attr`, lee `graph_features` | está embebido en script monolítico y usa filtros ad hoc; no está modularizado ni probado en `src/` | alto | alta | inmediata |
| construcción de grafos PyG | existente | `load_graph_safe` en `legacy/modelo_gnn_siames.py:1498` | crea `Data(x, edge_index, edge_attr)` con filtrado de `mask_*`, `var_*` e `is_truncation_node` | no conserva explícitamente el esquema objetivo ni availability masks separadas como primer ciudadano del paquete | medio | alta | inmediata |
| emparejamiento Mutante–WT | existente | `build_wt_index` en `legacy/modelo_gnn_siames.py:1612`; uso en `PairDataset.__getitem__` en `2085` | indexa WT por posición y lo usa como companion contextual | la estrategia real queda simplificada a posición; la especificación exige validación explícita por `chain + position + wt_aa` y trazabilidad | medio | alta | inmediata |
| creación de `is_mutation` | existente | `_find_variant_index_from_hdf5` y `_add_is_mutation_channel` en `1887`, `1919` | deriva el nodo mutado desde `diff_*` y añade el canal al final de `x` | encaja conceptualmente, pero depende de heurística implícita y del orden de columnas del notebook | medio | alta | inmediata |
| augmentations | existente | `PairDataset._augment` en `legacy/modelo_gnn_siames.py:2006` | aplica feature dropout, jitter y edge dropout suaves con columnas protegidas | no está desacoplado ni configurable por YAML; usa lógica inline y estado global | medio | alta | inmediata |
| `PairDataset` | existente | `legacy/modelo_gnn_siames.py:1986` | genera pares `aug-aug` o `mut-WT` según `lambda_wt`, añade `diff_struct` e `is_mutation`, devuelve referencia WT auxiliar | mezcla varias responsabilidades y extensiones experimentales en un único dataset | alto | media | inmediata |
| `pair_collate` | existente | `legacy/modelo_gnn_siames.py:2230` | agrupa batches, aplica z-score y lee globales para encoder y losses | depende de buffers globales `_ZSCORE_*`, `_GLOBALS_*`, `_LOSS_GLOBALS_*` | alto | media | inmediata |
| dataset para inferencia | existente | `H5GraphDataset` y `embed_all_with_loader` en `4093`, `4134` | carga grafos individuales y exporta embeddings para análisis | fuerza `return_space=\"bio\"` y hereda la semántica legacy de `proj_bio` | alto | media | corto plazo |
| encoder edge-aware | existente | `EdgeAwareEncoder` en `legacy/modelo_gnn_siames.py:3044` | usa `NNConv`, `edge_attr`, dropout residual, atención local y opcionalmente globales | no sigue la ruta objetivo `H_mut/H_WT -> poolings -> LayerNorm -> MLP_fusion -> h_encoder_*`; mezcla pooling y projection heads dentro del encoder | alto | media | inmediata |
| poolings existentes | parcial | `global_mean_pool`, `h_mut`, `h_neigh` en `3136-3189` | combina pooling global, nodo mutado y vecindad con atención | faltan `LayerNorm`, `MLP_fusion`, `availability_mask` y separación explícita entre `H_*` y `h_encoder_*` | alto | media | inmediata |
| representación pre-proyección verdadera | parcial | no existe `h_encoder_mut` / `h_encoder_wt` explícito; el encoder devuelve `z_instance`, `z_bio` desde `3200-3205` | el vector pooled `g` existe internamente, pero no se exporta ni nombra como representación principal | contradice la especificación, que exige exportar y usar `h_encoder_mut` / `h_encoder_wt` como espacio principal | alto | baja | inmediata |
| `projection_instance` | parcial | `proj_instance` en `3079`, usado en `3200` | existe una cabeza lineal para `z_instance` | está embebida en el encoder y no está separada semánticamente como módulo independiente | medio | media | inmediata |
| `proj_bio` / `z_bio` | parcial | `proj_bio` en `3080`, `z_bio` en `3201`, pérdidas auxiliares en `3321-3418` | participa en pérdidas auxiliares solo si pesos como `_SEV_LOSS_WEIGHT`, `_WT_AUX_LOSS_WEIGHT`, `_BIO_SUPCON_WEIGHT` o `_BIO_PSEUDO_WEIGHT` son mayores que cero | no hay auditoría general que demuestre gradientes, cambio de pesos o que deba interpretarse como representación biológica | alto | baja | corto plazo |
| `r_delta` | ausente | no localizado | no hay concatenación obligatoria de cinco bloques `h_mut`, `h_wt`, diferencia, valor absoluto y producto | incumple una pieza central de la especificación | alto | nula | inmediata |
| `MLP_delta` / `z_delta` | ausente | no localizado | no existe transformación aprendida relacional explícita | incumple la ruta relacional futura, aunque en la arquitectura objetivo sigue desactivada por defecto | medio | nula | medio plazo |
| `projection_pair` / `z_instance_pair` | ausente | no localizado | no existe cabeza relacional separada para el par completo | no cumple la separación obligatoria entre contraste individual y relacional | medio | nula | medio plazo |
| `L_delta` | ausente | no localizado | no existe pérdida explícita para `MLP_delta` | no hay ruta relacional aprendida | medio | nula | medio plazo |
| pérdida principal NT-Xent | existente | `nt_xent_loss` en `3221`; uso en `3312-3313` | implementa SimCLR/InfoNCE estándar entre dos vistas | sirve como baseline, pero no hay integración de máscaras de falsos negativos | medio | alta | inmediata |
| máscaras de falsos negativos | ausente | no localizado | no hay `W_ij`, `same_position`, `structural_hard` ni `structural_soft` | contradice la prioridad metodológica actual | alto | nula | inmediata después del baseline |
| alternativas sin negativos explícitos | ausente | no localizado | no hay VICReg ni Barlow Twins | faltan controles si NT-Xent degenera | medio | nula | corto plazo |
| regularización WT | parcial | `lambda_wt` en `2114`; `loss_wt_aux` en `3355-3370`; `loss_sev` en `3335-3352` | puede usar WT como positivo con probabilidad `lambda_wt`; además añade pérdidas auxiliares respecto a WT companion | la especificación prohíbe WT como positivo fuerte por defecto; aquí el default es `0.0`, pero la ruta existe y debe tratarse como ablación explícita | alto | media | inmediata |
| uso del WT como positivo fuerte | parcial | `use_wt = ... random.random() < self.lambda_wt` en `2114`; CLI `--lambda-wt` en `5078` | con `lambda_wt > 0` el segundo elemento del par puede ser WT companion | solo es conforme cuando se trate como control explícito; no debe confundirse con baseline | alto | baja | inmediata |
| `custom_structure_energy` en el encoder | parcial | globals configurables en `5316-5336`; `g = torch.cat([g, film_G_use], dim=1)` en `3197`; defaults con `custom_structure_energy` en `5185`, `5210` | si se activan globales, `custom_structure_energy` puede entrar por concatenación o FiLM | contradice la configuración base objetivo, donde energía debe quedar fuera del encoder | alto | baja | inmediata |
| `log(N)` en el encoder | existente | `torch.log1p(n_nodes)` en `3193` | concatena tamaño del grafo antes de las proyecciones | contradice la vigilancia explícita sobre proxies de tamaño/cobertura | alto | baja | inmediata |
| reconstrucción enmascarada / decoder | ausente | no localizado | no hay decoder ni pérdida reconstructiva | no existe la rama de corto plazo definida ahora en la documentación | medio | nula | corto plazo |
| auditoría general de gradientes y cambio de pesos | parcial | hay clip y EMA en `3426-3440`, métricas de checkpoint en `6010-6216` | controla entrenamiento básico y checkpointing, pero no audita módulo por módulo gradientes, `None`, NaN/Inf, cambio relativo de pesos o estado `trained/inactive/failed` | insuficiente respecto a la auditoría obligatoria actual | alto | media | inmediata |
| checkpointing | existente | `best_model_raw.pt`, `best_model_full.pt` en `5964-6216` | guarda `state_dict` y un payload extendido con metadatos de globales | útil, pero no equivale a `run_manifest.json` ni a auditoría completa de módulos | medio | alta | inmediata |
| `run_manifest.json` | ausente | no localizado | no genera manifiesto reproducible por run | incumple requisito central de reproducibilidad | alto | nula | inmediata |
| configuración | parcial | `argparse` monolítico en `5055-5400` y variables globales al inicio | parametriza muchas rutas y flags, pero desde CLI Colab y globals, no desde YAML resuelto | no coincide con el esquema de configuración declarativa del repositorio | alto | media | inmediata |
| splits | parcial | `grouped_split_by_position` en `5631`; `grouped_split_by_pocket` y `lopo_kfold_indices` como utilidades en `5679`, `5694` | el split efectivo del entrenamiento sí es agrupado por posición | no existe `leave-neighborhood-out` operativo; `grouped_split_by_pocket` es utilitario y no integrado | medio | alta | inmediata |
| exportación de embeddings | existente | `embed_all_with_loader` y guardado `.npy/.json` en `4134`, `6255-6347` | exporta embeddings latentes y manifiesto específico de exportación | exporta el espacio `bio` legacy, no las representaciones objetivo `h_encoder_*`, `r_delta`, `severity`, `mechanism_direction` | alto | media | corto plazo |
| clustering | parcial | `scan_silhouette_kmeans`, `run_umap_kmeans` en `4238`, `4341` | calcula KMeans/Agglomerative sobre `Z_all` original y exporta UMAP/PCA para visualización | el clustering no está formalizado en torno al espacio objetivo; además el análisis downstream reusa CSV de UMAP/clusters | medio | media | corto plazo |
| clustering sobre UMAP o espacio original | parcial | KMeans sobre `Z_all` en `4460-4490`; UMAP solo para exportación en `4356-4520` | el clustering principal mostrado en `run_umap_kmeans` ocurre sobre el embedding original, no sobre UMAP | esto es más cercano a la especificación que otros componentes, pero sigue acoplado a un pipeline de visualización legacy | medio | alta | corto plazo |
| auditorías anti-shortcut | existente | `compute_feature_importance_perm` en `3916`; shortcut audit desde `23650` en adelante | realiza permutation importance y auditorías específicas para `custom_structure_energy` | es extensa, pero está especializada, duplicada y depende de artefactos legacy como CSV de UMAP | medio | media | corto plazo |
| código duplicado | existente | múltiples bloques repetidos, p. ej. `_checkpoint_global_audit`, shortcut audits y parches de globals en zonas `18940`, `22993`, `24468`, `24859` | hay funciones duplicadas o parcheadas varias veces | eleva el riesgo de divergencia silenciosa y resultados inconsistentes | alto | baja | inmediata |
| dependencias globales y orden de notebook | existente | `_GLOBALS_*`, `_ZSCORE_*`, `_SEV_*`, `_BIO_*`, fallbacks en `4387-4420`, entradas interactivas en `5532-5539` | muchas funciones asumen variables globales o helpers ya definidos | impide reutilización directa como paquete y dificulta pruebas reproducibles | alto | baja | inmediata |

## Evaluación puntual pedida

### Representación pre-proyección

- No se exporta una representación pre-proyección verdadera conforme a la
  especificación.
- El tensor pooled `g` existe dentro de `EdgeAwareEncoder`, pero se consume
  inmediatamente para producir `z_instance` y `z_bio`.
- No aparecen `h_encoder_mut` ni `h_encoder_wt` como salidas públicas
  auditables.

### `proj_bio`

- `proj_bio` existe.
- Puede recibir gradientes solo cuando alguna loss auxiliar está activa:
  `loss_sev`, `loss_wt_aux`, `bio_supcon` o `bio_pseudo`.
- Con lectura estática no puede demostrarse que esas ramas estuvieran activas en
  runs concretos ni que `proj_bio` cambiara sus pesos.
- Por tanto no debe tratarse como espacio biológico validado.

### Ruta relacional

- `r_delta`: ausente.
- `MLP_delta`: ausente.
- `L_delta`: ausente.
- `projection_pair`: ausente.

### WT como positivo fuerte

- Existe esa posibilidad mediante `lambda_wt`.
- El valor por defecto es `0.0`, así que el baseline por defecto no lo activa.
- Aun así, la ruta existe en el código y debe tratarse como control explícito,
  no como comportamiento estándar.

### Máscaras de falsos negativos

- No existen máscaras reales de falsos negativos.
- No existe matriz `W_ij`.
- No hay modos `same_position`, `structural_hard` ni `structural_soft`.

### Reconstrucción enmascarada

- No existe decoder ni pérdida reconstructiva.
- No existe enmascaramiento explícito para una tarea de reconstrucción.

### Auditoría de gradientes y pesos

- No existe una auditoría general de gradientes y cambio de pesos por módulo
  equivalente a la requerida por la especificación.
- Sí existen checkpointing, clipping, EMA y métricas de entrenamiento.

### Reproducibilidad

- No existe `run_manifest.json`.
- Sí existe un `embedding_export_manifest.json` específico de exportación, que
  no sustituye al manifiesto de run requerido.

### Splits

- El entrenamiento usa realmente un split agrupado por posición.
- No existe `leave-neighborhood-out` operativo.
- Hay utilidades opcionales de pocket y K-fold LOPO, pero no forman parte del
  pipeline principal conforme.

### Energía, tamaño y clustering

- `custom_structure_energy` puede entrar en el encoder si se activan globales.
- `log(N)` entra explícitamente en el encoder.
- El clustering principal en `run_umap_kmeans` se calcula sobre el embedding
  original `Z_all`; UMAP se usa para proyección y exportación visual.
- Sin embargo, parte de las auditorías posteriores leen CSV con columnas UMAP y
  clusters ya exportados, lo que mantiene un acoplamiento fuerte con la capa de
  visualización legacy.

## Componentes con mayor valor de reutilización

- parsing de claves HDF5 y aminoácidos;
- validación de HDF5 y lectura segura;
- construcción de grafos `Data` de PyG;
- localización del nodo mutado y creación de `is_mutation`;
- pairing Mutante–WT por posición como punto de partida;
- augmentations conservadoras;
- split agrupado por posición;
- encoder `NNConv` sensible a `edge_attr`;
- exportación básica de embeddings y checkpointing.

## Componentes críticos ausentes o no conformes

- `h_encoder_mut` / `h_encoder_wt` exportables;
- `r_delta`;
- `MLP_delta`, `L_delta`, `projection_pair`;
- máscaras de falsos negativos;
- reconstrucción enmascarada;
- auditoría completa de gradientes y cambio de pesos por módulo;
- `run_manifest.json`;
- eliminación de rutas Colab/Drive y del estado global;
- separación modular entre dataset, encoder, pérdidas, exportación y análisis.

## Orden recomendado de migración

1. infraestructura mínima: configuración, seeds, logging y esquema inicial de
   `run_manifest.json`;
2. utilidades puras reutilizables: parsing HDF5, validación, `load_graph_safe`,
   `build_wt_index`, `is_mutation`;
3. dataset y collate modulares, eliminando dependencias globales;
4. split agrupado por posición con tests anti-leakage y registro en el
   manifiesto;
5. encoder edge-aware compartido, pero sin `proj_bio`, `log(N)` ni globales en
   la configuración base;
6. ruta explícita de representación pre-proyección `H_* -> h_encoder_*`;
7. representaciones deterministas `r_delta`, `severity` y
   `mechanism_direction`;
8. baseline NT-Xent limpio;
9. auditoría inicial y actualización del manifiesto durante el run;
10. exportación conforme de `h_encoder_*`, `r_delta`, `severity`,
    `mechanism_direction`;
11. máscaras de falsos negativos;
12. `leave-neighborhood-out`;
13. reconstrucción enmascarada;
14. solo después, componentes relacionales aprendidos y extensiones.
