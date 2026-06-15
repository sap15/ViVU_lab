# Decisiones arquitectónicas del modelo GNN siamés Mutante–WT para PKP2

## Estado del documento

- **Tipo:** registro vivo de decisiones arquitectónicas.
- **Proyecto:** GNN siamesa para análisis mecanístico de mutantes missense de PKP2.
- **Ubicación:** `docs/decisiones_arquitectonicas.md`.
- **Estado inicial:** activo.
- **Finalidad:** registrar decisiones aprobadas, justificación, consecuencias, alternativas y criterios de revisión.

Este archivo debe actualizarse cuando se modifique una decisión científica o técnica relevante, se active una nueva pérdida, se cambie el espacio de clustering, se incorpore un nuevo grupo de features, se cambie el split o una ablación justifique reemplazar una decisión previa.

## Fuentes de verdad relacionadas

1. `AGENTS.md`
2. `docs/especificacion_modelo.md`
3. `docs/informe_modelo.docx`
4. `configs/base.yaml`
5. `sample_data/sample_schema.json`
6. código validado mediante tests y ejecuciones reproducibles

En caso de contradicción:

- `AGENTS.md` prevalece para reglas operativas;
- `docs/especificacion_modelo.md` prevalece para definiciones implementables;
- este archivo explica el porqué de las decisiones;
- el código solo se considera conforme cuando existen tests y artefactos reproducibles.

## Convención de estados

- `propuesta`
- `aceptada`
- `implementada`
- `validada`
- `rechazada`
- `sustituida`
- `aplazada`

Una decisión no debe marcarse como `implementada` sin evidencia de código y tests. No debe marcarse como `validada` sin una ejecución reproducible y criterios de aceptación satisfechos.

---

## Resumen de decisiones iniciales

| ID | Decisión | Estado |
|---|---|---|
| ADR-001 | NNConv como encoder base sensible a aristas | aceptada |
| ADR-002 | Pesos compartidos entre Mutante y WT | aceptada |
| ADR-003 | `h_encoder_mut` y `h_encoder_wt` como representaciones pre-proyección principales | aceptada |
| ADR-004 | `r_delta` como baseline relacional determinista | aceptada |
| ADR-005 | `MLP_delta` desactivada en la configuración base | aceptada |
| ADR-006 | `z_delta` solo válida si está entrenada y auditada | aceptada |
| ADR-007 | `projection_instance` y `projection_pair` son cabezas distintas | aceptada |
| ADR-008 | Dos vistas del mismo mutante son los positivos principales | aceptada |
| ADR-009 | WT como referencia relacional, no como positivo fuerte por defecto | aceptada |
| ADR-010 | `custom_structure_energy` fuera del encoder base | aceptada |
| ADR-011 | Calidad estructural como máscara, filtro o covariable | aceptada |
| ADR-012 | Fusión de poolings mediante concatenación + LayerNorm + MLP | aceptada |
| ADR-013 | Clustering en el espacio original normalizado | aceptada |
| ADR-014 | UMAP se reserva principalmente para visualización | aceptada |
| ADR-015 | Split inicial `leave-position-out` | aceptada |
| ADR-016 | `leave-neighborhood-out` se implementará después | aceptada |
| ADR-017 | Separación entre `severity` y `mechanism_direction` | aceptada |
| ADR-018 | Auditoría de gradientes para toda rama paramétrica | aceptada |
| ADR-019 | Configuración y rutas mediante YAML/CLI | aceptada |
| ADR-020 | Colab como entorno de ejecución, no como fuente principal del código | aceptada |
| ADR-021 | Ablaciones acumulativas de grupos de features | aceptada |
| ADR-022 | Energía, tamaño y cobertura como posibles confusores | aceptada |
| ADR-023 | `z_instance` se reserva al objetivo contrastivo y control técnico | aceptada |
| ADR-024 | Etiquetas LoF/GoF/WT-like aplazadas hasta disponer de evidencia experimental | aceptada |
| ADR-025 | El código existente se audita antes de decidir qué reutilizar | aceptada |
| ADR-026 | NT-Xent sin máscara es el baseline inicial y las máscaras de falsos negativos son el control metodológico inmediato posterior | aceptada |
| ADR-027 | La reconstrucción enmascarada es una incorporación de corto plazo tras baseline reproducible y auditoría básica | aceptada |
| ADR-028 | `run_manifest.json` tiene ciclo de vida completo desde el inicio del run | aceptada |
| ADR-029 | `r_delta`, `severity` y `mechanism_direction` preceden a las máscaras de falsos negativos en el orden de migración | aceptada |

---

# ADR-001 — NNConv como encoder base sensible a atributos de arista

- **Estado:** aceptada
- **Ámbito:** arquitectura del encoder

## Contexto

Los grafos del proyecto contienen atributos de arista relevantes, como distancia, contacto y otras relaciones estructurales. El modelo debe utilizarlos de forma efectiva durante el paso de mensajes.

## Decisión

La arquitectura base utilizará `NNConv` o una operación Edge-Aware funcionalmente equivalente.

## Justificación

NNConv permite que `edge_attr` module la transformación de mensajes y es compatible con el código previo del proyecto.

## Consecuencias

- `edge_attr` debe utilizarse realmente;
- el encoder debe declarar `edge_dim`;
- debe existir un test de sensibilidad a atributos de arista;
- no se aceptará una implementación que cargue `edge_attr` pero lo ignore.

## Alternativas consideradas

- GCN sin atributos de arista;
- GraphSAGE;
- GAT sin integración explícita de `edge_attr`;
- SE(3)-GNN completa.

## Criterios de revisión

Solo se sustituirá si otra arquitectura mantiene compatibilidad con CPU y Colab, usa atributos de arista explícitamente, supera NNConv en varios seeds y no introduce shortcuts.

---

# ADR-002 — Encoder exactamente compartido entre Mutante y WT

- **Estado:** aceptada
- **Ámbito:** arquitectura siamesa

## Decisión

`G_mut` y `G_WT` se procesarán con la misma instancia del encoder y los mismos parámetros.

```text
G_mut ─┐
       ├── Shared Edge-Aware Encoder
G_WT ──┘
```

## Justificación

Permite que ambas ramas se representen en el mismo espacio latente y hace interpretable la comparación Mutante–WT.

## Consecuencias

- no se instanciarán dos encoders independientes;
- los tests deben demostrar identidad de parámetros;
- cualquier rama específica deberá añadirse como ablación explícita.

---

# ADR-003 — Representaciones pre-proyección principales

- **Estado:** aceptada
- **Ámbito:** nomenclatura y análisis

## Decisión

Las representaciones principales del grafo son:

```text
h_encoder_mut
h_encoder_wt
```

Alias permitidos:

```text
h_mut = h_encoder_mut
h_wt = h_encoder_wt
```

## Justificación

La salida pre-proyección conserva la información del encoder sin estar especializada exclusivamente para la pérdida contrastiva.

## Consecuencias

- `h_encoder_mut` será el espacio principal de clustering individual;
- `h_encoder_wt` se exportará como referencia;
- no se usará `h_encoder` sin indicar rama;
- `z_instance` no sustituirá automáticamente a `h_encoder_mut`.

---

# ADR-004 — `r_delta` como baseline relacional determinista

- **Estado:** aceptada
- **Ámbito:** representación Mutante–WT

## Decisión

```python
r_delta = concat(
    h_mut,
    h_wt,
    h_mut - h_wt,
    abs(h_mut - h_wt),
    h_mut * h_wt,
)
```

## Justificación

Integra información individual, diferencia dirigida, diferencia absoluta e interacción elemento a elemento sin introducir parámetros adicionales.

## Consecuencias

- debe calcularse aunque `MLP_delta` esté desactivada;
- debe exportarse como baseline;
- cualquier `z_delta` debe compararse con `r_delta`;
- su dimensión esperada es `5 * graph_dim`.

---

# ADR-005 — `MLP_delta` desactivada en la configuración base

- **Estado:** aceptada
- **Ámbito:** configuración base

## Decisión

```yaml
model:
  mlp_delta:
    enabled: false
```

## Justificación

Una MLP no produce una representación útil si no está conectada a una pérdida explícita.

## Consecuencias

- el baseline relacional será `r_delta`;
- `z_delta` no se exportará inicialmente;
- `projection_pair` permanecerá desactivada hasta que exista una ruta relacional válida.

---

# ADR-006 — Condiciones estrictas para aceptar `z_delta`

- **Estado:** aceptada
- **Ámbito:** entrenamiento y auditoría

## Decisión

`z_delta` solo se considera aprendida cuando `MLP_delta`:

1. está activa;
2. está incluida en el optimizador;
3. participa en una pérdida explícita;
4. recibe gradientes no nulos;
5. modifica sus pesos;
6. no colapsa;
7. supera las auditorías;
8. se compara favorablemente con `r_delta`.

## Consecuencias

Si no se cumplen estas condiciones:

- `z_delta` no se usará para clustering principal;
- no se exportará como espacio biológico;
- no se describirá como entrenada;
- el manifiesto la marcará como `inactive`, `failed` o `active_unvalidated`.

---

# ADR-007 — `projection_instance` y `projection_pair` son cabezas distintas

- **Estado:** aceptada
- **Ámbito:** projection heads

## Decisión

Ruta individual:

```text
h_encoder_mut
→ projection_instance
→ z_instance
```

Ruta relacional:

```text
r_delta o z_delta validada
→ projection_pair
→ z_instance_pair
```

## Justificación

Las entradas tienen semántica y dimensiones diferentes.

## Consecuencias

- no comparten pesos por defecto;
- no debe pasarse `r_delta` a `projection_instance`;
- no debe afirmarse que se contrasta el par si solo se proyecta `h_mut`.

---

# ADR-008 — Dos vistas del mismo mutante son los positivos principales

- **Estado:** aceptada
- **Ámbito:** aprendizaje autosupervisado

## Decisión

La ruta inicial utilizará dos augmentations del mismo mutante como par positivo.

## Consecuencias

Las augmentations:

- no deben eliminar el nodo mutado;
- no deben destruir señales `diff_*`;
- deben ser reproducibles;
- deben evitar perturbaciones estructurales poco plausibles.

---

# ADR-009 — WT como referencia relacional, no como positivo fuerte por defecto

- **Estado:** aceptada
- **Ámbito:** relación Mutante–WT

## Decisión

El WT se utiliza como referencia estructural, entrada de `r_delta`, regularización suave, tarea de margen, ranking o predicción.

WT como positivo fuerte solo será un control metodológico.

## Justificación

Forzar Mutante y WT a coincidir puede eliminar la señal diferencial buscada.

## Configuración base

```yaml
loss:
  lambda_wt: 0.0
```

---

# ADR-010 — `custom_structure_energy` fuera del encoder base

- **Estado:** aceptada
- **Ámbito:** features globales

## Decisión

`custom_structure_energy` no entra en el encoder principal de la configuración base.

## Justificación

Puede dominar componentes latentes y actuar como shortcut.

## Usos permitidos

- confusor;
- covariable;
- auditoría;
- baseline tabular;
- permutation test;
- ablación tardía.

## Usos prohibidos inicialmente

- input oculto;
- pseudo-label principal;
- objetivo principal de `L_delta`;
- señal dominante de supervised contrastive.

---

# ADR-011 — Calidad estructural como máscara, filtro o covariable

- **Estado:** aceptada
- **Ámbito:** control de calidad

## Decisión

En la configuración base, la calidad estructural se utiliza para enmascarar, ponderar, filtrar, excluir, estratificar y auditar.

No entra como input directo del encoder.

## Variables previstas

- confianza local;
- RMSD local;
- residuos ausentes;
- clash score;
- geometría anómala;
- cobertura Mutante–WT;
- validez del emparejamiento.

La calidad como input directo será una ablación independiente.

---

# ADR-012 — Fusión de poolings por concatenación, LayerNorm y MLP

- **Estado:** aceptada
- **Ámbito:** representación de grafo

## Decisión

```text
pool_global
+ nodo mutado
+ vecinos
+ shell local
+ shell regional
+ availability_mask
→ concat
→ LayerNorm
→ MLP_fusion
```

## Justificación

La concatenación conserva la identidad de cada bloque y facilita ablaciones.

## Alternativas

- suma;
- gating;
- atención;
- pooling global único.

Estas alternativas quedan reservadas para ablaciones.

---

# ADR-013 — Clustering en el espacio original normalizado

- **Estado:** aceptada
- **Ámbito:** análisis

## Decisión

El clustering principal se realizará sobre la representación original normalizada.

## Justificación

La reducción dimensional puede distorsionar distancias, densidades y vecindarios.

---

# ADR-014 — UMAP no es el espacio principal de clustering

- **Estado:** aceptada

## Decisión

UMAP se utilizará principalmente para visualización, diagnóstico y exploración cualitativa.

No será el espacio principal de selección de clusters.

---

# ADR-015 — Split inicial `leave-position-out`

- **Estado:** aceptada
- **Ámbito:** validación

## Decisión

Todas las sustituciones de una misma posición permanecerán en el mismo fold.

## Justificación

Evita leakage directo entre sustituciones de la misma posición.

## Limitación

No evita leakage entre posiciones estructuralmente vecinas.

---

# ADR-016 — `leave-neighborhood-out` se implementará posteriormente

- **Estado:** aceptada
- **Ámbito:** validación estricta

## Decisión

Después de validar el pipeline base, se incorporará `leave-neighborhood-out`.

## Definición inicial

Agrupación sobre la estructura WT con umbrales configurables, por ejemplo:

```text
Cα–Cα ≤ 12 Å
```

o:

```text
distancia de grafo ≤ 2
```

## Consecuencias

- menor tamaño efectivo de train;
- necesidad de exportar grupos;
- comparación obligatoria con `leave-position-out`.

---

# ADR-017 — Separación entre severidad y mecanismo

- **Estado:** aceptada
- **Ámbito:** representación e interpretación

## Decisión

```python
severity = distance(h_mut, h_wt)
mechanism_direction = normalize(h_mut - h_wt)
```

## Justificación

Permite distinguir cuánto cambia un mutante de cómo cambia.

## Consecuencias

- se auditarán clusters radiales;
- se comparará magnitud, dirección y representación relacional completa;
- se controlarán casos con `severity ≈ 0`.

---

# ADR-018 — Auditoría de toda rama paramétrica

- **Estado:** aceptada
- **Ámbito:** entrenamiento

## Módulos

- encoder;
- `MLP_fusion`;
- `projection_instance`;
- `projection_pair`;
- `bio_head`;
- `MLP_delta`;
- decoder;
- módulos WT auxiliares;
- futuras cabezas supervisadas.

## Campos mínimos

- parámetros;
- grupo del optimizador;
- pérdidas conectadas;
- norma de gradiente;
- fracción de gradiente cero;
- fracción `None`;
- NaN/Inf;
- cambio relativo de pesos;
- estado.

## Regla

Una rama instanciada no implica una rama entrenada.

---

# ADR-019 — Configuración por YAML y CLI

- **Estado:** aceptada
- **Ámbito:** reproducibilidad

## Decisión

Rutas, hiperparámetros, features, pérdidas y splits se declararán en YAML y argumentos CLI.

No se permiten rutas personales fijas en el paquete.

---

# ADR-020 — Colab como entorno de ejecución

- **Estado:** aceptada
- **Ámbito:** organización del proyecto

## Decisión

El desarrollo principal se realiza en el repositorio. Colab clona o actualiza el repositorio, instala dependencias, carga configuración, ejecuta scripts y guarda resultados.

## Justificación

Evita duplicar el modelo dentro de notebooks.

---

# ADR-021 — Ablaciones acumulativas

- **Estado:** aceptada
- **Ámbito:** evaluación de features

## Orden previsto

1. `STRUCT_ONLY`
2. `BIOCHEM_NONSTRUCT_ONLY`
3. `DELTA_BASELINE`
4. `STRUCT + BIOCHEM`
5. `STRUCT + BIOCHEM + DELTA`
6. `+ DOMAIN`
7. `+ SHELLS`
8. `+ NETWORK`
9. `+ ESM/LLR`
10. `+ ENERGY`

Cada ampliación se compara con la anterior.

---

# ADR-022 — Energía, tamaño y cobertura como posibles confusores

- **Estado:** aceptada

## Variables

- energía;
- número de nodos;
- número de aristas;
- `log(N)`;
- cobertura;
- residuos ausentes;
- dominio;
- calidad.

## Auditorías

- correlaciones;
- permutation tests;
- runs sin energía;
- energía permutada;
- globales-only;
- estabilidad con y sin confusores.

---

# ADR-023 — `z_instance` como espacio técnico

- **Estado:** aceptada

## Decisión

`z_instance` se utiliza para la pérdida contrastiva y controles técnicos.

No es el espacio biológico principal por defecto.

---

# ADR-024 — Etiquetas funcionales aplazadas

- **Estado:** aceptada

## Decisión

No se asignarán etiquetas LoF, GoF o WT-like en la fase actual.

## Fase futura

Se priorizarán variables continuas:

- estabilidad;
- localización;
- abundancia;
- actividad;
- distancia experimental al WT.

---

# ADR-025 — Auditoría del código heredado antes de reutilizarlo

- **Estado:** aceptada
- **Ámbito:** migración

## Decisión

El archivo existente `legacy/modelo_gnn_siames.py` debe analizarse antes de decidir qué módulos reutilizar.

## Elementos potencialmente reutilizables

- lectura HDF5;
- parsing de variantes;
- validación de grafos;
- `PairDataset`;
- NNConv;
- atención local;
- NT-Xent;
- checkpoints;
- exportación;
- análisis.

## Elementos que requieren revisión

- rutas absolutas de Google Drive;
- lógica de `z_bio`;
- uso de globales;
- `log(N)`;
- pairing Mutante–WT;
- WT como positivo;
- pseudo-labels de energía;
- estructura monolítica;
- ausencia de separación entre `projection_instance` y `projection_pair`;
- ausencia de `r_delta` según la especificación;
- auditoría incompleta de gradientes;
- configuración integrada en código;
- duplicación de lógica entre notebook y paquete.

## Consecuencia

No se copiará el script completo sin modularizarlo y probarlo.

---

# Decisiones no implementadas todavía

## ADR-P01 — Activación de `MLP_delta`

- **Estado:** aplazada
- **Dependencias:** `r_delta`, `L_delta`, auditoría de gradientes y comparación reproducible.

## ADR-P02 — Activación de `projection_pair`

- **Estado:** aplazada
- **Dependencias:** `r_delta` operativo y contraste relacional definido.

## ADR-P03 — Reconstrucción enmascarada

- **Estado:** aceptada
- **Prioridad:** corto plazo, media-alta
- **Dependencias:** Dataset validado, encoder compartido funcional, baseline autosupervisado reproducible, auditoría básica de gradientes, smoke test y checkpointing funcionales.

### Decisión

La reconstrucción enmascarada no forma parte del encoder mínimo inicial, pero
debe incorporarse a corto plazo mediante configuraciones separadas para:

- enmascaramiento de una fracción de features nodales;
- predicción de features nodales enmascaradas;
- reconstrucción de categorías de contacto;
- reconstrucción de bins de distancia;
- otros descriptores estructurales validados.

No se reconstruirán automáticamente identificadores, posiciones absolutas,
`is_mutation`, máscaras de disponibilidad, targets, `custom_structure_energy`
ni confusores globales.

### Consecuencias

- debe existir una máscara explícita que distinga observados, ocultados y
  ausentes;
- `L_masked_reconstruction` solo se calcula sobre elementos ocultados y
  válidos;
- el decoder debe auditarse en optimizador, gradientes, cambio de pesos y
  conexión efectiva con el encoder;
- debe compararse frente al mismo modelo sin reconstrucción y frente al mejor
  régimen sin negativos explícitos.

## ADR-P04 — Máscaras estructurales de falsos negativos

- **Estado:** aceptada
- **Prioridad:** inmediata después del baseline NT-Xent sin máscara
- **Dependencias:** matriz de vecindad WT, sampler por posición o vecindario y logging de negativos por ancla.

### Decisión

NT-Xent estándar sin máscara es el baseline inicial obligatorio. Inmediatamente
después debe implementarse y compararse el control metodológico de falsos
negativos dentro del batch con cuatro configuraciones mínimas:

1. `none`
2. `same_position`
3. `structural_hard`
4. `structural_soft`

Umbrales iniciales de referencia, configurables y sometidos a ablación:

- Cα–Cα ≤ 8 Å;
- contacto pesado ≤ 4.5 Å;
- `alpha` de soft masking en `{0.25, 0.5}`.

Para cada batch debe poder construirse conceptualmente una matriz `W_ij` con
`1` para negativos válidos, `0` para hard masking y `alpha` para soft masking.
Los positivos entre dos vistas del mismo mutante nunca deben enmascararse.

### Consecuencias

- deben registrarse negativos potenciales, válidos, proporción conservada y
  pares eliminados o ponderados por criterio para cada ancla;
- si un ancla conserva menos de 8 negativos válidos o menos del 25 % de los
  negativos potenciales, el batch debe reconstituirse;
- si el problema persiste, la comparación debe incluir VICReg o Barlow Twins en
  lugar de calcular una NT-Xent degenerada;
- esta máscara opera dentro del batch y no sustituye a
  `leave-neighborhood-out`, que actúa sobre la partición.

## ADR-P09 — Ciclo de vida completo de `run_manifest.json`

- **Estado:** aceptada
- **Prioridad:** inmediata desde la infraestructura mínima

### Decisión

`run_manifest.json` no se considera un artefacto exclusivamente final. Debe
iniciarse antes de cargar o entrenar el modelo, actualizarse durante la
ejecución y cerrarse después de auditorías y exportaciones.

### Consecuencias

- la infraestructura mínima debe incluir esquema y escritor inicial del
  manifiesto;
- deben registrarse desde el inicio identificación del run, configuración,
  dataset, esquema HDF5, seeds, split, features solicitadas, módulos previstos
  y pérdidas previstas;
- durante el run debe registrar dimensiones reales, módulos instanciados,
  optimizador, pérdidas conectadas, checkpoints y warnings;
- al cierre debe registrar métricas, auditorías, artefactos y estado final del
  run.

## ADR-P10 — Orden de migración de representaciones deterministas

- **Estado:** aceptada
- **Prioridad:** inmediata antes de máscaras de falsos negativos

### Decisión

Las representaciones deterministas `r_delta`, `severity` y
`mechanism_direction` deben construirse y validarse antes de introducir las
máscaras de falsos negativos.

Orden consolidado:

1. producir `h_encoder_mut` y `h_encoder_wt`;
2. calcular `r_delta`, `severity` y `mechanism_direction`;
3. implementar el baseline NT-Xent sin máscara;
4. auditar y exportar;
5. añadir después las máscaras de falsos negativos.

### Consecuencias

- `r_delta` permanece separada de `z_delta`;
- las máscaras modifican el tratamiento de negativos dentro de NT-Xent, pero no
  son necesarias para definir las representaciones deterministas;
- `MLP_delta`, `L_delta` y `projection_pair` permanecen en una fase posterior.

## ADR-P05 — Calidad estructural como input

- **Estado:** aplazada
- **Dependencias:** auditoría de calidad completa y ablación separada.

## ADR-P06 — Dinámica y comunicación mecánica

- **Estado:** aplazada
- **Dependencias:** ANM/GNM reproducible y correspondencia Mutante–WT.

## ADR-P07 — ESM/LLR

- **Estado:** aplazada
- **Dependencias:** baseline estructural validado.

## ADR-P08 — Cabezas funcionales

- **Estado:** aplazada
- **Dependencias:** datos experimentales continuos y splits estrictos.

---

# Decisiones rechazadas para la configuración base

## R-001 — Dos encoders independientes

- **Estado:** rechazada
- **Motivo:** rompe la comparabilidad directa del espacio.

## R-002 — WT como positivo fuerte por defecto

- **Estado:** rechazada
- **Motivo:** puede borrar la diferencia biológica.

## R-003 — `custom_structure_energy` como input base

- **Estado:** rechazada
- **Motivo:** riesgo elevado de shortcut.

## R-004 — Pseudo-labels principales basados en energía

- **Estado:** rechazada
- **Motivo:** organiza el espacio según una señal computacional.

## R-005 — Clustering principal sobre UMAP

- **Estado:** rechazada
- **Motivo:** posible distorsión de geometría.

## R-006 — Interpretar `z_bio` por nombre

- **Estado:** rechazada
- **Motivo:** una cabeza no es biológica si no está entrenada.

## R-007 — Exportar `z_delta` sin `L_delta`

- **Estado:** rechazada
- **Motivo:** salida no entrenada.

## R-008 — Modelo completo generado en una sola tarea

- **Estado:** rechazada
- **Motivo:** dificulta pruebas, revisión y trazabilidad.

## R-009 — Mantener rutas absolutas en el paquete

- **Estado:** rechazada
- **Motivo:** impide portabilidad.

## R-010 — Copiar el script monolítico sin revisión

- **Estado:** rechazada
- **Motivo:** preservaría ambigüedades y acoplamiento.

---

# Criterios para cambiar una decisión

Una decisión aceptada solo puede cambiar si existe:

1. una nueva entrada ADR;
2. una motivación explícita;
3. una ablación;
4. una comparación con el baseline;
5. tests;
6. auditoría de gradientes cuando corresponda;
7. resultados en varios seeds;
8. análisis anti-shortcut;
9. actualización de `run_manifest.json`;
10. actualización de `docs/especificacion_modelo.md` si cambia la implementación esperada.

No debe sobrescribirse una decisión anterior sin dejar trazabilidad.

---

# Plantilla para nuevas decisiones

```markdown
## ADR-XXX — Título

- **Estado:** propuesta
- **Fecha:** AAAA-MM-DD
- **Autoría:** equipo del proyecto
- **Ámbito:** datos / modelo / pérdidas / entrenamiento / análisis / despliegue

### Contexto

Descripción del problema.

### Decisión

Qué se decide exactamente.

### Justificación

Por qué se adopta.

### Consecuencias positivas

- ...

### Consecuencias negativas o riesgos

- ...

### Alternativas consideradas

- ...

### Evidencia requerida

- tests;
- métricas;
- seeds;
- auditorías;
- ablaciones.

### Criterios de aceptación

- ...

### Criterios de revisión

- ...

### Archivos afectados

- `...`

### Runs relacionados

- `run_id`
```

---

# Historial inicial

## Versión 0.1

Se crea el registro inicial con las decisiones fundacionales:

- NNConv;
- encoder compartido;
- representación pre-proyección;
- `r_delta`;
- `MLP_delta` desactivada;
- separación de projection heads;
- WT como referencia;
- energía fuera del encoder;
- calidad como máscara;
- clustering en espacio original;
- UMAP como visualización;
- `leave-position-out`;
- `leave-neighborhood-out` aplazado;
- auditoría obligatoria;
- Colab como entorno de ejecución;
- revisión del código heredado antes de reutilizarlo.

---

# Regla de cierre

La arquitectura de referencia queda definida inicialmente como:

```text
G_mut ─┐
       ├── Shared NNConv / Edge-Aware Encoder
G_WT ──┘

H_mut / H_WT
→ poolings y fusión
→ h_encoder_mut / h_encoder_wt
→ r_delta
→ análisis relacional

h_encoder_mut
→ projection_instance
→ z_instance
→ pérdida contrastiva
```

`MLP_delta` y `projection_pair` permanecen desactivadas en la configuración base.

`r_delta` es el baseline relacional obligatorio.

`z_delta` solo se acepta cuando exista evidencia de entrenamiento efectivo y comparación favorable frente a `r_delta`.
