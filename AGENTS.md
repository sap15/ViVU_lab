# AGENTS.md

## 1. Propósito del repositorio

Este repositorio implementa una GNN siamesa sensible a atributos de arista para estudiar las alteraciones estructurales, bioquímicas y relacionales producidas por mutaciones missense de PKP2 mediante comparación pareada Mutante–WT.

El objetivo científico de la fase actual no es predecir directamente benignidad, patogenicidad, LoF, GoF o WT-like. El objetivo es aprender representaciones reproducibles que permitan:

- medir la desviación de cada mutante respecto al WT;
- comparar mutantes entre sí;
- separar magnitud de cambio y patrón de cambio;
- identificar agrupamientos mecanísticos provisionales;
- auditar atajos, confusores, leakage y ramas no entrenadas;
- preparar una fase futura con datos experimentales continuos y etiquetas funcionales fiables.

El código debe funcionar:

- localmente en Linux;
- en Google Colab;
- en CPU de forma obligatoria;
- en GPU de forma opcional;
- sin rutas personales fijas;
- con configuración mediante YAML y argumentos de línea de comandos.

---

## 2. Fuente de verdad y orden de lectura obligatorio

Antes de crear, modificar o revisar código, Codex debe leer, en este orden:

1. `AGENTS.md`
2. `docs/especificacion_modelo.md`
3. `docs/informe_modelo.docx`
4. `docs/decisiones_arquitectonicas.md`
5. `configs/base.yaml`
6. `sample_data/sample_schema.json`
7. `sample_data/README.md`
8. cualquier código existente en `legacy/`
9. los tests relacionados con el módulo que se vaya a modificar

La especificación científica principal está en:

- `docs/especificacion_modelo.md`
- `docs/informe_modelo.docx`

`docs/decisiones_arquitectonicas.md` registra decisiones posteriores, excepciones justificadas y cambios aprobados.

Si existe contradicción:

1. prevalece `AGENTS.md` para reglas operativas;
2. prevalece `docs/especificacion_modelo.md` para definiciones implementables;
3. prevalece `docs/informe_modelo.docx` para el contexto científico;
4. cualquier discrepancia debe documentarse antes de modificar la arquitectura.

No se debe inferir el esquema real del HDF5 cuando no esté documentado. Debe inspeccionarse `sample_data/sample_schema.json` y, cuando sea necesario, los ejemplos mínimos permitidos.

---

## 3. Alcance y estrategia de implementación

No generar el modelo completo en una sola tarea.

El desarrollo debe ser incremental, con cambios pequeños, pruebas específicas y commits revisables.

Orden recomendado:

1. infraestructura, configuración y reproducibilidad;
2. auditoría de HDF5 y emparejamiento Mutante–WT;
3. Dataset de PyTorch Geometric;
4. splits sin leakage;
5. encoder compartido y pooling;
6. representaciones pre-proyección;
7. `r_delta`, `severity` y `mechanism_direction`;
8. `projection_instance` y contraste individual;
9. entrenamiento, checkpoints y auditoría de gradientes;
10. exportación de embeddings;
11. máscaras de falsos negativos inmediatamente después de validar el baseline NT-Xent, la auditoría básica y la exportación inicial;
12. clustering y auditorías anti-shortcut;
13. `L_relative_WT`;
14. reconstrucción enmascarada después del baseline reproducible, smoke test, checkpointing y auditoría básica de gradientes;
15. `MLP_delta`, `L_delta` y `projection_pair`;
16. dominios, shells, métricas de red, dinámica, ESM/LLR y datos experimentales.

La fase 1 debe incluir explícitamente:

- carga y validación de configuración;
- seeds reproducibles;
- logging;
- esquema inicial de `run_manifest.json`;
- creación del manifiesto antes de cargar o entrenar el modelo;
- mecanismo de actualización del manifiesto durante y al final del run.

Cada tarea debe:

- modificar el menor número razonable de archivos;
- añadir o actualizar tests;
- ejecutar los tests relevantes;
- mostrar el diff;
- documentar riesgos y limitaciones;
- no cambiar decisiones científicas sin autorización.

---

## 4. Nomenclatura obligatoria

La nomenclatura siguiente no debe alterarse ni reutilizarse con otro significado.

### 4.1. Grafos y matrices nodales

- `G_mut`: grafo del mutante.
- `G_WT`: grafo WT companion.
- `H_mut`: matriz de embeddings nodales del mutante producida por la última capa del encoder antes del pooling.
- `H_WT`: matriz de embeddings nodales del WT producida por la última capa del encoder antes del pooling.

`H_mut` y `H_WT` siempre representan matrices nodales. Nunca deben utilizarse para representar vectores de grafo.

### 4.2. Representaciones de grafo pre-proyección

- `h_encoder_mut`: representación vectorial del grafo mutante después de pooling, fusión y normalización.
- `h_encoder_wt`: representación vectorial del grafo WT después de pooling, fusión y normalización.
- `h_mut`: alias permitido de `h_encoder_mut`.
- `h_wt`: alias permitido de `h_encoder_wt`.

No usar el símbolo genérico `h_encoder` sin indicar explícitamente la rama mutante o WT.

### 4.3. Representación relacional determinista

`r_delta` es la representación relacional Mutante–WT determinista y de referencia.

Debe definirse exactamente como:

```python
r_delta = concat(
    h_mut,
    h_wt,
    h_mut - h_wt,
    abs(h_mut - h_wt),
    h_mut * h_wt,
)
```

Reglas:

- `r_delta` no contiene parámetros entrenables;
- `r_delta` debe poder calcularse aunque `MLP_delta` esté desactivada;
- `r_delta` es la representación relacional principal mientras `MLP_delta` no haya sido validada;
- debe exportarse en las ejecuciones relacionales;
- cualquier cambio en su definición requiere una decisión arquitectónica explícita.

### 4.4. Representación relacional aprendida

- `MLP_delta`: transformación paramétrica aplicada a `r_delta`.
- `z_delta`: salida de `MLP_delta(r_delta)`.

`z_delta` solo puede denominarse representación aprendida, exportarse como espacio biológico o utilizarse como entrada principal de análisis cuando se demuestre que `MLP_delta`:

1. está activa;
2. está incluida en el optimizador;
3. participa en una pérdida explícita identificada;
4. recibe gradientes no nulos;
5. no presenta gradientes NaN o Inf;
6. modifica sus pesos;
7. supera la comparación con `r_delta`;
8. queda registrada como `trained` en `run_manifest.json`.

Si estas condiciones no se cumplen:

- `z_delta` no debe exportarse como representación aprendida;
- no debe utilizarse para clustering principal;
- no debe describirse como entrenada;
- la configuración debe usar `r_delta`.

### 4.5. Projection heads

- `projection_instance`: cabeza de proyección individual.
- `z_instance`: salida de `projection_instance`.
- `projection_pair`: cabeza de proyección relacional del par Mutante–WT.
- `z_instance_pair`: salida de `projection_pair`.

Flujo individual:

```text
h_encoder_mut
→ projection_instance
→ z_instance
```

Flujo relacional:

```text
r_delta
→ projection_pair
→ z_instance_pair
```

o, únicamente con `MLP_delta` validada:

```text
z_delta
→ projection_pair
→ z_instance_pair
```

`projection_instance` y `projection_pair` son módulos semánticamente diferentes y no deben compartir pesos por defecto.

No debe pasarse `r_delta` a `projection_instance`.

No debe pasarse una representación individual `h_mut` a `projection_pair` para afirmar que se está contrastando el par Mutante–WT.

### 4.6. Magnitud y dirección del cambio

- `severity`: magnitud de la desviación Mutante–WT.
- `mechanism_direction`: dirección normalizada del cambio.

Definiciones conceptuales:

```python
severity = distance(h_mut, h_wt)
mechanism_direction = normalize(h_mut - h_wt)
```

Debe existir manejo explícito de valores de `severity` próximos a cero para evitar divisiones inestables.

`severity` y `mechanism_direction` son variables derivadas post-encoder. No deben reintroducirse como inputs del mismo entrenamiento.

### 4.7. Representaciones adicionales

- `z_bio`: solo puede utilizarse si una cabeza biológica explícita participa en una pérdida, recibe gradientes y cambia sus pesos.
- `h_pair_delta`: no debe introducirse como alias ambiguo de `r_delta` o `z_delta`. Si se usa en una extensión multiescala futura, su definición debe aparecer en la especificación y en el manifiesto.
- `z_instance_pair`: se reserva al contraste de dos vistas del par completo Mutante–WT.

---

## 5. Reglas arquitectónicas no negociables

### 5.1. Encoder compartido

Los grafos Mutante y WT deben procesarse con exactamente la misma instancia del encoder.

No se permiten dos encoders independientes salvo ablación explícita aprobada.

Los tests deben comprobar:

- identidad de parámetros compartidos;
- actualización conjunta;
- gradients en el encoder;
- funcionamiento con grafos de tamaños diferentes;
- sensibilidad a `edge_attr`;
- funcionamiento en CPU.

### 5.2. Encoder sensible a atributos de arista

La arquitectura base debe utilizar NNConv u otra operación equivalente que emplee atributos de arista de forma efectiva.

No debe ignorarse silenciosamente `edge_attr`.

Debe existir al menos un test que demuestre que modificar `edge_attr` puede modificar la salida manteniendo constantes los nodos.

### 5.3. Fusión de poolings

La representación de grafo debe admitir, según disponibilidad:

- pooling global;
- embedding del nodo mutado;
- pooling de vecinos inmediatos;
- pooling local;
- pooling regional;
- `availability_mask`.

La ruta base debe ser:

```text
bloques de pooling
→ concatenación
→ LayerNorm
→ MLP_fusion
→ h_encoder_mut / h_encoder_wt
```

Los bloques no disponibles deben marcarse mediante máscara. No deben sustituirse de forma indistinguible por un valor que pueda interpretarse como ausencia real de cambio.

### 5.4. Mutante y WT

Mutante y WT no deben tratarse como positivos contrastivos fuertes por defecto.

Los positivos principales de la fase autosupervisada son dos vistas aumentadas del mismo mutante.

El WT se utiliza como:

- referencia relacional;
- baseline;
- término auxiliar suave;
- tarea de margen, ranking o predicción;
- control experimental explícito.

WT como positivo fuerte solo puede aparecer como control metodológico nombrado y separado.

### 5.5. Contraste individual y contraste relacional

El contraste individual debe operar sobre dos vistas del mismo mutante:

```text
h_encoder_mut_view1 → projection_instance → z_instance_view1
h_encoder_mut_view2 → projection_instance → z_instance_view2
```

El contraste relacional debe operar sobre dos vistas del par completo:

```text
r_delta_view1 o z_delta_view1 validado
→ projection_pair
→ z_instance_pair_view1

r_delta_view2 o z_delta_view2 validado
→ projection_pair
→ z_instance_pair_view2
```

No mezclar ambas rutas ni reutilizar una cabeza para afirmar que se ejecuta la otra.

### 5.6. Energía global

`custom_structure_energy` no debe entrar en el encoder base.

Debe utilizarse inicialmente como:

- confusor;
- variable de auditoría;
- variable de estratificación;
- baseline;
- señal para permutation tests;
- ablación tardía.

No debe utilizarse como:

- objetivo principal de `L_delta`;
- pseudo-label principal;
- señal dominante del contraste;
- variable oculta introducida por concatenación o FiLM en la configuración base.

Cualquier ejecución que use energía como input debe declararlo explícitamente en el YAML y en `run_manifest.json`.

### 5.7. Calidad estructural

Los indicadores de calidad estructural no entran en el encoder base.

Se usan primero como:

- máscaras;
- pesos;
- filtros;
- criterios de exclusión;
- variables de estratificación;
- variables de auditoría.

La calidad como input directo constituye una ablación independiente.

Deben auditarse, cuando estén disponibles:

- cobertura Mutante–WT;
- residuos ausentes;
- confianza local;
- RMSD local;
- clash score;
- geometría anómala;
- validez del emparejamiento;
- aristas comparables.

### 5.8. UMAP y clustering

El clustering principal debe realizarse en el espacio original normalizado.

UMAP se utilizará principalmente para visualización.

PCA puede utilizarse para análisis lineal.

No se debe seleccionar una arquitectura únicamente porque produzca una visualización UMAP visualmente atractiva.

Si se clusteriza sobre una reducción dimensional, debe justificarse y compararse con el espacio original.

---

## 6. Reglas sobre datos y HDF5

### 6.1. No asumir el esquema

El código no debe asumir nombres, shapes o dtypes que no estén documentados o validados.

Antes de implementar el Dataset debe comprobarse:

- jerarquía del HDF5;
- nombres de datasets;
- shapes;
- dtypes;
- `edge_index`;
- orientación de `edge_index`;
- node features;
- edge features;
- graph features;
- identificadores;
- posición;
- aminoácido WT;
- aminoácido mutante;
- nodo mutado;
- máscaras;
- features diferenciales;
- targets, si existen;
- emparejamiento con WT companion.

### 6.2. Datos del proyecto

Los archivos mutantes siguen el patrón conceptual:

```text
pos_<posición>_<AA_WT>_<AA_MUTANTE>.pdb
```

La proteína WT de referencia se denomina:

```text
PKP2_WT.pdb
```

La cadena de referencia del proyecto es:

```text
A
```

La implementación debe conservar:

- `variant_id`;
- `position`;
- `wt_aa`;
- `mut_aa`;
- identificador del WT companion;
- `is_mutation`;
- máscaras de disponibilidad;
- metadatos de cobertura y calidad.

### 6.3. Features base y grupos

Los grupos deben seleccionarse desde configuración.

Grupos previstos:

- estructura;
- bioquímica;
- `diff_bioq`;
- `diff_struct_node`;
- dominio/región;
- shells;
- red estructural;
- dinámica;
- comunicación mecánica;
- ESM/LLR;
- energía;
- calidad.

La configuración base no debe activar automáticamente todos los grupos.

Las variables exactas incluidas en cada grupo deben registrarse en `run_manifest.json`.

### 6.4. Features y variables a vigilar

No introducir automáticamente como señal biológica:

- `custom_structure_energy`;
- número de nodos;
- número de aristas;
- `log(N)`;
- cobertura;
- residuos ausentes;
- truncación;
- calidad;
- dominio;
- región;
- rutas locales del archivo.

Estas variables deben tratarse como posibles confusores y auditarse.

### 6.5. Valores ausentes

Un valor no calculable no debe confundirse con un cambio real igual a cero.

Debe conservarse una máscara de disponibilidad cuando corresponda.

El Dataset debe rechazar o enmascarar explícitamente:

- NaN;
- Inf;
- shapes incompatibles;
- índices fuera de rango;
- ausencia del nodo mutado;
- más de un nodo mutado cuando no esté permitido;
- pares Mutante–WT no resolubles;
- incompatibilidades de correspondencia.

---

## 7. Splits y prevención de leakage

Los splits deben generarse a partir de grupos, nunca mediante división aleatoria simple de variantes cuando eso permita que posiciones equivalentes aparezcan en train y validación.

Splits previstos:

- `leave_position_out`;
- `leave_neighborhood_out`;
- `leave_domain_out`;
- `leave_structural_region_out`.

Reglas:

- todas las mutaciones de una misma posición deben permanecer en el mismo fold;
- en `leave_neighborhood_out`, posiciones estructuralmente próximas deben permanecer en el mismo grupo;
- los grupos y sus miembros deben exportarse;
- train, validation y test no deben compartir posiciones o grupos prohibidos;
- los tests deben detectar leakage deliberado;
- la máscara de falsos negativos no sustituye al split por vecindario.

Umbrales iniciales orientativos, configurables y auditables:

- vecindad para máscara dentro del batch: Cα–Cα ≤ 8 Å o contacto pesado ≤ 4.5 Å;
- vecindad para split: Cα–Cα ≤ 12 Å o distancia de grafo ≤ 2.

No codificar estos valores de forma fija. Deben estar en YAML.

---

## 8. Pérdidas y condiciones de uso

### 8.1. Contraste individual

La primera versión debe implementar NT-Xent entre dos augmentations del mismo mutante.

Debe registrar:

- temperatura;
- tamaño de batch;
- número de negativos válidos;
- presencia de colapso;
- embeddings con NaN o Inf.

### 8.2. Máscaras de falsos negativos

NT-Xent estándar sin máscara es el baseline inicial obligatorio.

Las máscaras de falsos negativos son el control metodológico prioritario
inmediatamente posterior a ese baseline. No deben tratarse como una extensión
remota ni como sustituto del split por vecindario.

Modalidades previstas:

- sin máscara;
- misma posición;
- vecindad estructural hard;
- vecindad estructural soft.

Comparaciones mínimas obligatorias:

1. NT-Xent sin máscara.
2. máscara para sustituciones diferentes en la misma posición.
3. máscara estructural dura.
4. máscara estructural ponderada.

Umbrales iniciales orientativos, configurables y sometidos a ablación:

- Cα–Cα ≤ 8 Å;
- contacto pesado ≤ 4.5 Å;
- `alpha` de soft masking en `{0.25, 0.5}`.

Para cada batch debe poder construirse conceptualmente una matriz `W_ij`:

- `W_ij = 1` para negativos válidos;
- `W_ij = 0` para hard masking;
- `W_ij = alpha` para soft masking.

Los positivos entre las dos vistas del mismo mutante nunca deben enmascararse.

Debe registrarse para cada ancla:

- negativos potenciales;
- negativos válidos;
- proporción de negativos conservados;
- fracción enmascarada;
- número de pares eliminados por cada criterio;
- número de pares ponderados por cada criterio;
- motivo del enmascaramiento.

Si un ancla conserva menos de 8 negativos válidos o menos del 25 % de los
negativos potenciales, el batch debe reconstituirse mediante muestreo por
posición o vecindario. Si el problema persiste, debe considerarse una
alternativa sin negativos explícitos, como VICReg o Barlow Twins, en lugar de
calcular una NT-Xent degenerada.

La máscara de falsos negativos actúa dentro del batch. `leave-neighborhood-out`
actúa en la partición de train, validation y test. Son controles distintos y
ninguno sustituye al otro.

Criterios documentales de aceptación:

- el baseline NT-Xent sin máscara funciona y supera el smoke test;
- las cuatro configuraciones pueden compararse manteniendo encoder,
  augmentations, split, seeds y presupuesto de entrenamiento constantes;
- se registra el número de negativos válidos por ancla;
- no aparecen anchors con denominadores degenerados;
- las máscaras no provocan colapso;
- la configuración seleccionada mejora o mantiene la estabilidad entre seeds;
- se preservan suficientes negativos informativos;
- la comparación se realiza también frente a VICReg o Barlow Twins si el número
  efectivo de negativos resulta insuficiente.

### 8.3. `L_relative_WT`

Modalidades permitidas:

- `none`;
- `margin`;
- `ranking`;
- `predictive`;
- `strong_positive_control`.

La configuración base debe usar:

```text
lambda_wt = 0
```

La pérdida debe documentar:

- espacio de aplicación;
- distancia;
- margen;
- stop-gradient;
- coeficiente;
- dirección del efecto.

No usar una pérdida atractiva simple que fuerce indiscriminadamente `h_mut ≈ h_wt`.

### 8.4. `L_delta`

`L_delta` solo debe activarse cuando `MLP_delta` esté activa.

Puede incluir:

- consistencia entre vistas;
- varianza/covarianza;
- predicción de descriptores de cambio;
- reconstrucción de deltas validados.

`custom_structure_energy` no debe ser el objetivo principal.

Debe existir al menos un coeficiente de entrenamiento de `MLP_delta` mayor que cero.

### 8.5. Reconstrucción enmascarada

La reconstrucción enmascarada no forma parte de la primera implementación
mínima del encoder, pero es una incorporación de corto plazo y prioridad
media-alta. Debe implementarse después de disponer de:

- Dataset validado;
- encoder compartido funcional;
- baseline autosupervisado reproducible;
- auditoría básica de gradientes;
- smoke test y checkpointing funcionales.

Puede incluir, mediante configuraciones separadas:

- enmascaramiento de una fracción de features nodales;
- predicción de las features nodales enmascaradas;
- reconstrucción de categorías de contacto;
- reconstrucción de bins de distancia de aristas;
- otros descriptores estructurales validados.

No debe reconstruir automáticamente:

- identificadores;
- posiciones absolutas;
- `is_mutation`;
- máscaras de disponibilidad;
- targets;
- `custom_structure_energy`;
- confusores globales;
- variables cuya reconstrucción cree un shortcut trivial.

Debe existir una máscara explícita que distinga:

- valores observados;
- valores ocultados para la tarea;
- valores originalmente ausentes o no calculables.

La pérdida reconstructiva debe calcularse únicamente sobre elementos ocultados
y válidos.

Debe auditarse:

- decoder en optimizador;
- término de pérdida que entrena el decoder;
- gradientes del decoder;
- ausencia de gradientes `None`, NaN o Inf en el decoder;
- gradientes hacia el encoder;
- retropropagación de `L_masked_reconstruction` a decoder y encoder;
- ausencia de `detach` accidental entre encoder y decoder;
- cambio de pesos;
- pérdida reconstructiva;
- ausencia de reconstrucción trivial de confusores;
- estado `trained`, `inactive`, `failed` o `not_applicable` en
  `run_manifest.json`.

Comparaciones mínimas obligatorias:

1. NT-Xent sin reconstrucción.
2. NT-Xent más reconstrucción enmascarada.
3. el mejor régimen sin negativos explícitos sin reconstrucción.
4. ese mismo régimen más reconstrucción enmascarada.

Todas las comparaciones deben mantener constantes dataset, split, seeds,
encoder, augmentations, batch size o presupuesto efectivo, número de épocas o
pasos y política de early stopping.

Criterios documentales de aceptación:

- el decoder pertenece al optimizador;
- recibe gradientes no nulos;
- cambia sus pesos;
- la pérdida retropropaga al encoder;
- la reconstrucción supera un baseline trivial;
- no introduce NaN, Inf ni colapso;
- mejora o mantiene la estabilidad entre seeds;
- preserva o mejora el contenido estructural del embedding;
- no aumenta la dependencia respecto a energía, tamaño, cobertura o calidad;
- su utilidad se demuestra frente al mismo modelo sin reconstrucción;
- si no añade valor reproducible, debe poder desactivarse sin alterar el
  pipeline principal.

---

## 9. Auditoría obligatoria de módulos entrenables

Toda rama paramétrica activa debe auditarse.

Módulos mínimos:

- encoder compartido;
- `MLP_fusion`;
- `projection_instance`;
- `projection_pair`, si está activa;
- `bio_head`, si existe;
- `MLP_delta`, si está activa;
- decoder de reconstrucción;
- cualquier módulo WT auxiliar;
- cabezas supervisadas futuras.

Para cada módulo registrar:

- `parameter_count`;
- `optimizer_group`;
- `connected_losses`;
- `mean_gradient_norm`;
- `median_gradient_norm`;
- `min_gradient_norm`;
- `max_gradient_norm`;
- `zero_gradient_fraction`;
- `none_gradient_fraction`;
- `has_nan_or_inf`;
- `relative_weight_change`;
- `status`.

Estados permitidos:

- `trained`;
- `inactive`;
- `failed`;
- `not_applicable`.

Una rama no se considera entrenada por el mero hecho de estar instanciada.

Una salida de una rama `inactive` o `failed` no debe utilizarse como representación biológica.

---

## 10. Representaciones que deben exportarse

Exportar, cuando existan y sean válidas:

- `h_encoder_mut`;
- `h_encoder_wt`;
- `r_delta`;
- `severity`;
- `mechanism_direction`;
- `z_instance`;
- metadatos de variante;
- split;
- seed;
- identificador del checkpoint.

Exportar condicionalmente:

- `z_delta`, solo con `MLP_delta` entrenada y auditada;
- `z_instance_pair`, solo con `projection_pair` entrenada;
- `z_bio`, solo si la cabeza biológica está entrenada;
- `H_mut` y `H_WT`, solo cuando sean necesarios para análisis nodales.

No utilizar `z_instance` como espacio biológico principal por defecto.

---

## 11. Auditorías anti-shortcut

Cada run candidato debe evaluar, cuando estén disponibles:

- `custom_structure_energy`;
- número de nodos;
- número de aristas;
- densidad;
- `log(N)`;
- cobertura;
- residuos ausentes;
- calidad estructural;
- dominio;
- posición;
- vecindario;
- región estructural;
- severidad.

Pruebas esperadas:

- correlaciones con componentes principales;
- correlaciones con UMAP como diagnóstico visual;
- asociación con `severity`;
- asociación con distancias Mutante–WT;
- energía eliminada;
- energía permutada;
- globales-only;
- comparación con y sin calidad;
- detección de clusters degenerados;
- tamaños mínimo y máximo de cluster;
- número de singletons;
- estabilidad tras retirar confusores.

No interpretar toda correlación como shortcut automáticamente. Debe evaluarse si una sola variable explica por sí sola la geometría latente.

---

## 12. Tests obligatorios

Antes de aceptar un cambio, ejecutar como mínimo:

```bash
pytest -q
python -m compileall src
```

Cuando corresponda:

```bash
python scripts/audit_dataset.py --config configs/base.yaml
python scripts/train.py --config configs/base.yaml --smoke-test
```

### 12.1. Tests de datos

Comprobar:

- carga de una muestra;
- carga de un batch;
- shapes;
- dtypes;
- índices válidos;
- NaN e Inf;
- nodo mutado;
- máscaras;
- emparejamiento Mutante–WT;
- caso con valores ausentes;
- grafos con diferente número de nodos;
- rechazo de casos inválidos.

### 12.2. Tests del encoder

Comprobar:

- pesos compartidos;
- sensibilidad a `edge_attr`;
- shapes de `H_mut` y `H_WT`;
- shapes de `h_encoder_mut` y `h_encoder_wt`;
- gradientes;
- CPU;
- batch heterogéneo;
- availability mask.

### 12.3. Tests relacionales

Comprobar numéricamente:

- definición exacta de `r_delta`;
- ausencia de parámetros en `r_delta`;
- dimensión esperada;
- `severity`;
- `mechanism_direction`;
- estabilidad cuando `severity ≈ 0`;
- activación/desactivación de `MLP_delta`;
- prohibición de exportar `z_delta` no validado.

### 12.4. Tests de pérdidas

Comprobar:

- positivos correctos;
- Mutante y WT no usados como positivo fuerte por defecto;
- máscaras hard y soft;
- número de negativos válidos;
- batches degenerados;
- gradientes de cada rama;
- ausencia de NaN e Inf;
- `lambda_wt=0` como baseline;
- `MLP_delta` sin gradientes cuando `lambda_delta=0`;
- `MLP_delta` con gradientes cuando `L_delta` está activa.

### 12.5. Smoke test

Ejecutar con 4–8 pares Mutante–WT, batch pequeño y 2 épocas.

Debe comprobar:

- carga;
- forward;
- backward;
- optimizador;
- gradientes;
- cambio de pesos;
- checkpoint;
- reanudación;
- exportación;
- `run_manifest.json`;
- CPU;
- ausencia de NaN e Inf.

### 12.6. Overfit controlado

Debe existir un test de overfit sobre aproximadamente 4 muestras para detectar:

- pérdidas desconectadas;
- errores de batching;
- parámetros fuera del optimizador;
- tensores detached;
- augmentations incompatibles;
- dimensiones erróneas.

---

## 13. Configuración

Todos los hiperparámetros, rutas y grupos de features deben declararse mediante:

- archivos YAML;
- argumentos CLI;
- configuración resuelta guardada por run.

No introducir rutas absolutas como:

```text
/content/drive/MyDrive/...
/home/usuario/...
C:\Users\...
```

Las rutas deben pasarse mediante configuración.

El notebook puede montar Google Drive, pero la ruta debe ser configurable.

Cada run debe guardar una copia de la configuración resuelta.

Los valores por defecto deben ser conservadores:

- CPU compatible;
- energía global desactivada;
- calidad como input desactivada;
- `MLP_delta` desactivada;
- `projection_pair` desactivada;
- `lambda_wt = 0`;
- `lambda_delta = 0`;
- split por posición;
- seeds explícitos.

---

## 14. Reproducibilidad y `run_manifest.json`

Cada auditoría, entrenamiento, ablación, baseline, exportación o clustering debe generar o actualizar:

```text
run_manifest.json
```

Flujo obligatorio del manifiesto:

1. crear el esquema y el escritor inicial de `run_manifest.json`;
2. iniciar el manifiesto antes de cargar o entrenar el modelo;
3. actualizarlo durante la ejecución;
4. cerrarlo después de auditorías y exportaciones.

Debe incluir, como mínimo:

### Identificación

- `run_id`;
- fecha y hora de inicio;
- fecha y hora de finalización;
- commit de Git;
- estado del working tree;
- Python;
- PyTorch;
- PyTorch Geometric;
- dispositivo;
- dataset;
- esquema HDF5;
- checksums cuando sea posible.

Al inicio del run deben quedar registrados, como mínimo:

- `run_id`;
- configuración serializada;
- dataset y versión de esquema;
- seed global;
- seeds específicos;
- split seleccionado;
- grupos de train, validation y test cuando ya estén resueltos;
- grupos de features solicitados;
- node features solicitadas;
- edge features solicitadas;
- globales solicitados;
- módulos que se pretenden activar;
- pérdidas que se pretenden utilizar.

### Datos y features

- node features exactas;
- edge features exactas;
- graph features exactas;
- grupos activos;
- variables de `diff_bioq`;
- variables de `diff_struct_node`;
- shells;
- deltas de aristas;
- dominio;
- red;
- dinámica;
- ESM/LLR;
- energía;
- política de calidad;
- política de missing values;
- reglas de pairing.

### Arquitectura

- tipo de encoder;
- capas;
- dimensiones;
- poolings;
- método de fusión;
- estado de `projection_instance`;
- estado de `projection_pair`;
- estado de `MLP_delta`;
- estado del decoder;
- módulos entrenables;
- módulos congelados.

Durante el run debe poder actualizarse con:

- dimensiones reales detectadas;
- features realmente cargadas;
- módulos realmente instanciados;
- módulos activos e inactivos;
- parámetros incluidos en el optimizador;
- pérdidas conectadas a cada módulo;
- checkpoints generados;
- progreso de entrenamiento;
- incidencias o warnings relevantes.

### Entrenamiento

- pérdida principal;
- pérdidas auxiliares;
- lambdas;
- modalidad de `L_relative_WT`;
- máscara de falsos negativos;
- thresholds;
- optimizador;
- scheduler;
- batch size;
- epochs;
- early stopping;
- mixed precision;
- gradient clipping.

### Split y seeds

- tipo de split;
- definición de grupos;
- train/validation/test;
- seeds de Python;
- NumPy;
- PyTorch;
- CUDA;
- sampler;
- augmentations;
- clustering.

### Auditorías

- gradientes;
- cambio de pesos;
- NaN/Inf;
- anti-shortcut;
- clustering degenerado;
- singletons;
- tamaños de clusters;
- estado de aceptación;
- motivo de rechazo.

Al final del run debe actualizarse con:

- checkpoint seleccionado;
- checkpoint final;
- métricas;
- resultados de auditoría de gradientes;
- cambio relativo de pesos;
- detección de NaN, Inf o colapso;
- embeddings exportados;
- representación utilizada para clustering;
- resultados de clustering;
- auditorías anti-shortcut;
- artefactos generados;
- estado final del run: `accepted`, `rejected`, `failed` o `incomplete`;
- motivo de rechazo o fallo cuando corresponda.

### Artefactos

- checkpoints;
- métricas;
- embeddings;
- clustering;
- figuras;
- vecinos;
- informes de calidad;
- auditorías.

Un run sin manifiesto completo no se considera reproducible.

Ningún entrenamiento futuro se considerará completamente reproducible si el
manifiesto solo se genera al final o si no permite reconstruir:

- qué datos se usaron;
- qué features entraron;
- qué módulos estuvieron activos;
- qué pérdidas entrenaron cada rama;
- qué split y seeds se utilizaron;
- qué representaciones se exportaron;
- qué auditorías se superaron o fallaron.

---

## 15. Estructura de resultados

Los artefactos no deben escribirse dentro de `src/`.

Usar una estructura configurable como:

```text
runs/<run_id>/
├── run_manifest.json
├── config_resolved.yaml
├── train.log
├── metrics.csv
├── gradient_audit.json
├── dataset_audit.json
├── checkpoints/
├── embeddings/
├── clustering/
├── shortcuts/
└── figures/
```

Los artefactos grandes deben quedar fuera de Git.

---

## 16. Reglas de Git

Antes de modificar:

```bash
git status
git branch --show-current
```

Después de modificar:

```bash
git diff --stat
git diff
pytest -q
python -m compileall src
```

Reglas:

- usar ramas pequeñas por tarea;
- no mezclar arquitectura, datos, notebooks, pérdidas y análisis en un único commit;
- no modificar datos originales;
- no borrar código existente sin justificarlo;
- no reformatear archivos no relacionados;
- no cambiar nombres públicos sin migración;
- no hacer commit de HDF5 grandes, PDB, checkpoints, resultados o credenciales;
- no hacer `git push` ni fusionar ramas sin petición explícita;
- mostrar el diff antes de dar la tarea por terminada.

Nombres de ramas recomendados:

```text
feature/hdf5-audit
feature/mut-wt-pairing
feature/shared-encoder
feature/r-delta
feature/instance-contrast
feature/gradient-audit
feature/false-negative-mask
feature/relative-wt
feature/mlp-delta
feature/colab-notebook
```

Commits pequeños y descriptivos, por ejemplo:

```text
Implement HDF5 audit and validation report
Add shared edge-aware encoder
Implement deterministic Mut-WT relational representation
Add gradient audit for trainable branches
```

---

## 17. Google Colab

Los notebooks deben ser delgados.

No copiar el modelo completo dentro de celdas.

El flujo debe ser:

```text
repositorio Git
→ instalación editable o importación desde src
→ carga de YAML
→ llamada a scripts o funciones del paquete
→ guardado de artefactos
```

El notebook de entrenamiento debe permitir:

- comprobar entorno;
- clonar o actualizar el repositorio;
- instalar dependencias;
- montar Google Drive opcionalmente;
- definir rutas configurables;
- auditar dataset;
- smoke test;
- entrenar;
- reanudar checkpoint;
- exportar embeddings;
- clustering;
- guardar manifiesto;
- comprimir resultados.

Debe funcionar sin GPU.

Mixed precision solo se activa cuando el dispositivo y la configuración lo permitan.

---

## 18. Elementos prohibidos en la configuración base

No introducir en el modelo base:

- `custom_structure_energy` como input;
- calidad estructural como input directo;
- WT como positivo contrastivo fuerte;
- pseudo-labels de energía;
- supervised contrastive con bins de energía;
- CADD como señal dominante;
- ESM/LLR como señal dominante;
- `z_delta` sin pérdida y auditoría;
- `z_bio` no entrenado;
- rutas personales;
- split aleatorio con leakage de posiciones;
- clustering principal sobre UMAP sin justificación;
- etiquetas LoF, GoF o WT-like sin datos experimentales fiables;
- objetivos futuros presentados como ya implementados.

---

## 19. Criterios para aceptar una modificación

Una tarea solo se considera terminada cuando:

1. respeta este archivo;
2. respeta la especificación;
3. no introduce nomenclatura ambigua;
4. añade o actualiza tests;
5. pasan los tests relevantes;
6. compila `src/`;
7. funciona en CPU;
8. no introduce rutas fijas;
9. no introduce energía o calidad de forma oculta;
10. registra la configuración;
11. actualiza documentación cuando corresponda;
12. muestra el diff;
13. identifica limitaciones pendientes.

Para cambios de arquitectura, además:

- debe existir una ablación;
- debe existir un criterio de aceptación;
- debe existir auditoría de gradientes;
- debe existir comparación con el baseline anterior;
- debe actualizarse `docs/decisiones_arquitectonicas.md`.

---

## 20. Conducta esperada de Codex

Antes de escribir código:

- resumir el objetivo;
- indicar los archivos que se modificarán;
- identificar datos faltantes;
- señalar supuestos;
- proponer tests;
- no inventar el esquema HDF5.

Durante la implementación:

- realizar cambios mínimos;
- mantener interfaces claras;
- usar type hints cuando aporten claridad;
- manejar errores con mensajes informativos;
- evitar `except Exception` silenciosos;
- evitar estados globales ocultos;
- separar código, configuración y artefactos.

Después:

- ejecutar tests;
- mostrar resultados;
- mostrar diff;
- indicar riesgos;
- no afirmar que una rama está entrenada sin evidencia;
- no afirmar que un cluster es funcional sin validación experimental.

---

## 21. Preguntas bloqueantes

Codex debe detener la implementación y pedir aclaración cuando falte información crítica sobre:

- esquema real del HDF5;
- nombres reales de features;
- orientación de `edge_index`;
- identificación del nodo mutado;
- emparejamiento Mutante–WT;
- máscaras;
- cobertura;
- tratamiento de residuos ausentes;
- dimensiones esperadas;
- objetivos de una pérdida;
- procedencia de una etiqueta;
- compatibilidad PyTorch/PyG;
- significado de una variable global;
- criterio de split;
- criterio de aceptación de una ablación.

No debe resolver estas incertidumbres mediante supuestos silenciosos.

---

## 22. Regla científica crítica final

`r_delta` es la representación relacional determinista Mutante–WT de referencia.

`z_delta` solo se considera aprendida cuando `MLP_delta`:

- participa en una pérdida explícita;
- está en el optimizador;
- recibe gradientes;
- cambia sus pesos;
- no colapsa;
- supera las auditorías;
- se compara con `r_delta`;
- queda registrada como entrenada en `run_manifest.json`.

Hasta entonces, usar `r_delta`.
