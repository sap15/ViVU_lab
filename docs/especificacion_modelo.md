# Especificación implementable del modelo GNN siamés Mutante–WT para PKP2

## 1. Propósito del documento

Este documento convierte el informe técnico del proyecto en una especificación implementable, auditable y utilizable por Codex.

Su función es definir con precisión:

- qué debe construir el repositorio;
- qué nombres deben conservarse;
- qué entradas espera el modelo;
- qué salidas debe producir;
- qué módulos deben entrenarse;
- qué condiciones deben cumplirse antes de interpretar una representación;
- qué pruebas y auditorías son obligatorias;
- qué artefactos debe generar cada ejecución;
- en qué orden debe implementarse el sistema.

Este archivo debe leerse junto con:

- `AGENTS.md`;
- `docs/informe_modelo.docx`;
- `docs/decisiones_arquitectonicas.md`;
- `configs/base.yaml`;
- `sample_data/sample_schema.json`;
- `sample_data/README.md`.

La especificación describe la arquitectura objetivo del proyecto. No implica que todos sus módulos estén ya implementados.

Nota de sincronización documental:

- `docs/informe_modelo.docx` fue revisado como referencia legible desde este
  entorno;
- las correcciones normativas de prioridad metodológica se registran en este
  archivo Markdown y en `docs/decisiones_arquitectonicas.md`;
- el contenido equivalente del informe `.docx` deberá sincronizarse después
  mediante un procedimiento reproducible de edición de ese binario.

---

# 2. Objetivo científico

## 2.1. Objetivo principal

El objetivo del modelo es aprender representaciones de las alteraciones estructurales, bioquímicas y relacionales producidas por mutaciones missense de PKP2 mediante comparación pareada entre cada mutante y su estructura WT de referencia.

El modelo debe permitir:

1. representar cada mutante como un grafo de residuos;
2. representar su WT companion con el mismo esquema;
3. procesar ambos grafos con un encoder compartido;
4. cuantificar la desviación Mutante–WT;
5. distinguir magnitud de cambio y dirección del cambio;
6. comparar mutantes entre sí;
7. identificar patrones mecanísticos provisionales;
8. exportar espacios latentes auditables;
9. evitar que una única variable global domine el espacio;
10. preparar una futura fase supervisada con datos experimentales.

## 2.2. Pregunta científica

La pregunta principal no es:

> ¿Qué mutaciones son benignas o patogénicas?

La pregunta actual es:

> ¿Qué mutantes de PKP2 producen patrones semejantes de alteración estructural, bioquímica, local, regional o relacional respecto al WT?

## 2.3. Interpretación de los clusters

Durante la fase autosupervisada:

- los clusters se consideran agrupaciones mecanísticas provisionales;
- un cluster no se denominará automáticamente LoF;
- un cluster no se denominará automáticamente GoF;
- una posición próxima al WT no se denominará automáticamente WT-like;
- una energía alta no se interpretará automáticamente como pérdida de función.

Las etiquetas funcionales solo se introducirán cuando existan datos experimentales fiables.

---

# 3. Alcance de la fase actual

## 3.1. Incluido

La fase actual incluye:

- validación de HDF5;
- validación del emparejamiento Mutante–WT;
- Dataset de PyTorch Geometric;
- encoder compartido sensible a atributos de arista;
- poolings globales y mutation-centric;
- representación pre-proyección;
- representación relacional determinista `r_delta`;
- `severity`;
- `mechanism_direction`;
- contraste entre dos vistas del mismo mutante;
- auditoría de gradientes;
- exportación de embeddings;
- clustering en el espacio original normalizado;
- auditorías anti-shortcut;
- splits por posición;
- preparación del split por vecindario;
- `run_manifest.json`;
- baselines y ablaciones iniciales.

## 3.2. No incluido como capacidad demostrada inicial

Los elementos siguientes son requisitos futuros o extensiones y no deben presentarse como ya implementados hasta que existan tests y ejecuciones reproducibles:

- `MLP_delta` entrenada;
- `z_delta` validada;
- `projection_pair`;
- contraste relacional del par completo;
- máscaras de falsos negativos más allá del baseline NT-Xent sin máscara;
- reconstrucción enmascarada;
- leave-neighborhood-out;
- dominios;
- shells multiescala;
- métricas de red;
- ANM/GNM;
- perturbation-response scanning;
- comunicación mecánica;
- ESM/LLR;
- cabezas experimentales;
- LoF/GoF/WT-like;
- fine-tuning por familia.

---

# 4. Datos del proyecto

## 4.1. Entidad biológica

Proteína de estudio:

```text
PKP2
```

Proteína WT de referencia:

```text
PKP2_WT.pdb
```

Cadena de referencia:

```text
A
```

## 4.2. Convención de nombres de mutantes

Los archivos mutantes siguen la convención:

```text
pos_<posición>_<AA_WT>_<AA_MUTANTE>.pdb
```

Ejemplo:

```text
pos_102_S_F.pdb
```

La implementación debe conservar, cuando estén disponibles:

- `variant_id`;
- `position`;
- `wt_aa`;
- `mut_aa`;
- `chain_id`;
- `pdb_id`;
- identificador del WT companion;
- tipo de variante;
- metadatos de calidad;
- metadatos de cobertura.

## 4.3. Naturaleza de los archivos PDB

Los PDB mutantes ya contienen la estructura mutada.

Por tanto, cuando se utilicen herramientas de generación o validación derivadas de DeepRank2:

```text
apply_variant = false
```

No debe volver a aplicarse la sustitución sobre una estructura ya mutada.

---

# 5. Esquema de entrada HDF5

## 5.1. Regla general

El esquema real del HDF5 no debe deducirse a partir de esta especificación.

La implementación debe leer:

```text
sample_data/sample_schema.json
```

y validar ejemplos mínimos antes de fijar nombres o dimensiones.

## 5.2. Información mínima que debe poder recuperarse

Cada muestra debe proporcionar, directa o indirectamente:

### Identificación

- identificador del mutante;
- posición;
- aminoácido WT;
- aminoácido mutante;
- cadena;
- identificador del WT companion.

### Grafo mutante

- matriz de node features;
- `edge_index`;
- matriz de edge features;
- número de nodos;
- número de aristas;
- nodo mutado;
- máscaras de disponibilidad;
- graph features;
- metadatos.

### Grafo WT companion

- misma clase de estructuras;
- mismo esquema de features;
- correspondencia validada;
- cobertura;
- máscaras.

## 5.3. Componentes esperados

El HDF5 puede contener:

```text
node_features
edge_index
edge_features
graph_features
node_feature_names
edge_feature_names
graph_feature_names
variant_id
position
wt_aa
mut_aa
is_mutation
variant_neighbors_mask
availability_mask
targets
metadata
```

La ubicación exacta de cada elemento debe validarse en el esquema real.

## 5.4. Requisitos de validación

Antes del entrenamiento deben comprobarse:

- existencia de datasets obligatorios;
- shapes;
- dtypes;
- consistencia entre número de nodos y node features;
- consistencia entre número de aristas y edge features;
- orientación de `edge_index`;
- índices dentro de rango;
- NaN;
- Inf;
- nodo mutado;
- exactamente un nodo mutado cuando corresponda;
- correspondencia Mutante–WT;
- cobertura;
- máscaras;
- features ausentes;
- nombres duplicados;
- dimensiones incompatibles;
- casos rechazables.

## 5.5. Política de valores ausentes

Un valor no calculable no debe sustituirse automáticamente por cero cuando cero pueda interpretarse como ausencia real de cambio.

Debe utilizarse:

- máscara de disponibilidad;
- valor de relleno documentado;
- exclusión;
- o política explícita configurable.

La política aplicada debe registrarse en `run_manifest.json`.

---

# 6. Grupos de features

## 6.1. Node features base

La configuración base puede incluir:

- tipo de residuo;
- masa;
- carga;
- punto isoeléctrico;
- tamaño;
- donadores de hidrógeno;
- aceptores de hidrógeno;
- polaridad;
- SASA;
- RSA;
- HSE;
- residue depth;
- estructura secundaria;
- `is_mutation`;
- `diff_bioq`;
- `diff_struct_node`, cuando esté validado.

## 6.2. Edge features base

La configuración base puede incluir:

- distancia;
- contacto;
- relación secuencial;
- covalencia, si está disponible y validada;
- electrostática, si está disponible y validada;
- van der Waals, si está disponible y validada;
- relación de cadena.

En grafos monoméricos de cadena A, `same_chain` puede ser constante y debe auditarse por redundancia.

## 6.3. Features diferenciales bioquímicas

El grupo `diff_bioq` puede incluir:

- `diff_mass`;
- `diff_charge`;
- `diff_pI`;
- `diff_size`;
- `diff_hb_donors`;
- `diff_hb_acceptors`;
- `diff_polarity`.

La lista exacta debe quedar registrada por run.

## 6.4. Features estructurales diferenciales

`diff_struct_node` puede incluir:

- ΔSASA;
- ΔRSA;
- ΔHSE;
- Δresidue depth;
- cambios de estructura secundaria;
- cambios de distancia local;
- contactos ganados o perdidos, cuando sean fiables.

No deben mezclarse implícitamente con:

- `delta_shells`;
- `delta_edges`;
- `edge_gain_loss`.

Cada grupo debe ser independiente y ablatable.

## 6.5. Features globales y confusores

Pueden existir:

- `custom_structure_energy`;
- `custom_complex_energy_phenotype`;
- número de nodos;
- número de aristas;
- `delta_nodes`;
- cobertura;
- residuos ausentes;
- calidad estructural.

En la configuración base:

```text
custom_structure_energy no entra en el encoder
```

Estas variables se usan como:

- auditoría;
- estratificación;
- baseline;
- confusor;
- permutation test;
- ablación.

## 6.6. Grupos futuros

- dominio/región;
- shells;
- red estructural;
- dinámica;
- comunicación mecánica;
- ESM/LLR;
- deltas de aristas;
- calidad como input directo.

Estos grupos deben introducirse por fases y mediante configuraciones separadas.

---

# 7. Nomenclatura obligatoria

## 7.1. Grafos

```text
G_mut
G_WT
```

## 7.2. Matrices nodales

```text
H_mut
H_WT
```

Definición:

- matrices producidas por la última capa de propagación;
- antes del pooling;
- una fila por nodo;
- una columna por dimensión latente.

Nunca deben emplearse como vectores de grafo.

## 7.3. Representaciones de grafo

```text
h_encoder_mut
h_encoder_wt
```

Alias permitidos:

```text
h_mut = h_encoder_mut
h_wt = h_encoder_wt
```

No utilizar `h_encoder` sin indicar la rama.

## 7.4. Representación relacional

```text
r_delta
```

Es determinista y no paramétrica.

## 7.5. Representación relacional aprendida

```text
z_delta
```

Es la salida de:

```text
MLP_delta(r_delta)
```

Solo puede considerarse aprendida si supera las condiciones de la sección 12.

## 7.6. Proyección individual

```text
projection_instance
z_instance
```

## 7.7. Proyección del par

```text
projection_pair
z_instance_pair
```

## 7.8. Severidad y dirección

```text
severity
mechanism_direction
```

## 7.9. Cabeza biológica opcional

```text
bio_head
z_bio
```

`z_bio` solo puede interpretarse si `bio_head` está entrenada y auditada.

---

# 8. Arquitectura general

## 8.1. Flujo principal

```text
G_mut ─┐
       ├── Shared Edge-Aware Encoder
G_WT ──┘
```

Salidas nodales:

```text
G_mut → H_mut
G_WT  → H_WT
```

Después del pooling y fusión:

```text
H_mut → h_encoder_mut
H_WT  → h_encoder_wt
```

## 8.2. Encoder compartido

El encoder debe:

- usar exactamente los mismos pesos para Mutante y WT;
- recibir node features;
- recibir `edge_index`;
- utilizar `edge_attr`;
- funcionar con grafos de distinto tamaño;
- funcionar en CPU;
- ser compatible con batches PyG.

Arquitectura inicial recomendada:

```text
NNConv o equivalente Edge-Aware
```

## 8.3. Requisitos funcionales del encoder

Debe producir:

```text
H_mut: [N_mut, hidden_dim]
H_WT:  [N_wt, hidden_dim]
```

y posteriormente:

```text
h_encoder_mut: [batch_size, graph_dim]
h_encoder_wt:  [batch_size, graph_dim]
```

Las dimensiones exactas se obtienen de la configuración.

---

# 9. Poolings y fusión

## 9.1. Bloques previstos

La representación de cada grafo puede combinar:

- pooling global;
- embedding del nodo mutado;
- pooling de vecinos inmediatos;
- pooling local;
- pooling regional;
- máscaras de disponibilidad.

## 9.2. Ruta base

```text
u_graph = concat(
    pool_global,
    h_node_mutation,
    pool_neighbors,
    pool_shell_local,
    pool_shell_regional,
    availability_mask
)
```

Después:

```text
h_encoder = MLP_fusion(LayerNorm(u_graph))
```

Por tanto:

```text
h_encoder_mut = MLP_fusion(LayerNorm(u_graph_mut))
h_encoder_wt  = MLP_fusion(LayerNorm(u_graph_wt))
```

## 9.3. Reglas

- la operación base es concatenación;
- los bloques deben tener dimensiones compatibles;
- los bloques opcionales deben estar enmascarados;
- no debe asumirse que suma y concatenación son equivalentes;
- gating o atención son ablaciones;
- la contribución de cada bloque debe poder desactivarse por YAML.

---

# 10. Construcción de `r_delta`

## 10.1. Definición obligatoria

```python
r_delta = concat(
    h_mut,
    h_wt,
    h_mut - h_wt,
    abs(h_mut - h_wt),
    h_mut * h_wt,
)
```

## 10.2. Propiedades

- no tiene parámetros;
- no depende de una pérdida específica;
- debe ser reproducible;
- debe calcularse después del encoder;
- usa vectores de grafo, no matrices nodales;
- es la referencia relacional por defecto.

## 10.3. Dimensión

Si:

```text
dim(h_mut) = d
dim(h_wt) = d
```

entonces:

```text
dim(r_delta) = 5d
```

La implementación debe validar esta dimensión.

## 10.4. Uso

`r_delta` debe utilizarse para:

- análisis Mutante–WT;
- clustering relacional;
- comparación con `h_encoder_mut`;
- baseline para evaluar `z_delta`;
- entrada de `projection_pair`;
- cálculo de vecindarios relacionales;
- futuras cabezas supervisadas.

---

# 11. `severity` y `mechanism_direction`

## 11.1. Severidad

```python
severity = distance(h_mut, h_wt)
```

La distancia debe fijarse en configuración:

- euclídea;
- coseno;
- u otra explícitamente documentada.

## 11.2. Dirección

```python
mechanism_direction = normalize(h_mut - h_wt)
```

## 11.3. Estabilidad numérica

Cuando:

```text
severity ≈ 0
```

debe utilizarse epsilon y registrar los casos inestables.

## 11.4. Interpretación

- `severity`: cuánto cambia;
- `mechanism_direction`: cómo cambia.

La separación evita que los clusters estén definidos solo por distancia radial al WT.

---

# 12. Condiciones para utilizar `z_delta`

## 12.1. Regla científica central

`r_delta` es la representación relacional determinista de referencia.

`z_delta` solo se considera aprendida cuando `MLP_delta`:

- participa en una pérdida explícita;
- está incluida en el optimizador;
- recibe gradientes;
- modifica sus pesos;
- supera las auditorías;
- se compara favorablemente con `r_delta`.

## 12.2. Requisitos obligatorios

Para marcar `z_delta` como válida:

1. `MLP_delta.enabled = true`;
2. `MLP_delta` está en un grupo del optimizador;
3. existe al menos una pérdida conectada;
4. `lambda_delta > 0` o término equivalente;
5. los gradientes no son `None`;
6. los gradientes no son siempre cero;
7. no hay NaN ni Inf;
8. los pesos cambian;
9. no existe colapso;
10. el run registra la auditoría;
11. `z_delta` se compara con `r_delta`;
12. la comparación no empeora de forma clara estabilidad e interpretación.

## 12.3. Estados permitidos

```text
inactive
active_unvalidated
trained_validated
failed
```

## 12.4. Restricciones

Si `MLP_delta` no está validada:

- no exportar `z_delta` como espacio biológico;
- no usar `z_delta` para clustering principal;
- no usarla en cabezas futuras;
- no denominarla aprendida;
- utilizar `r_delta`.

---

# 13. Projection heads

## 13.1. `projection_instance`

Entrada:

```text
h_encoder_mut
```

Salida:

```text
z_instance
```

Uso:

- NT-Xent;
- VICReg;
- Barlow Twins;
- control técnico.

No debe ser el espacio biológico principal por defecto.

## 13.2. `projection_pair`

Entrada:

```text
r_delta
```

o:

```text
z_delta validada
```

Salida:

```text
z_instance_pair
```

Uso:

- contraste entre dos vistas del par completo;
- consistencia relacional.

## 13.3. Separación obligatoria

`projection_instance` y `projection_pair`:

- tienen semántica diferente;
- pueden tener dimensiones de entrada diferentes;
- no comparten pesos por defecto;
- deben auditarse por separado.

---

# 14. Pérdidas

## 14.1. Contraste individual

Objetivo principal inicial:

```text
NT-Xent entre dos vistas del mismo mutante
```

No utilizar Mutante y WT como positivos fuertes por defecto.

## 14.2. Máscaras de falsos negativos

NT-Xent estándar sin máscara es el baseline inicial obligatorio.

Las máscaras de falsos negativos son el control metodológico prioritario
inmediatamente posterior a la validación de ese baseline. No deben describirse
como una extensión lejana ni confundirse con el split por vecindario.

Modalidades:

```text
none
same_position
structural_hard
structural_soft
```

Comparaciones mínimas obligatorias:

1. NT-Xent sin máscara.
2. máscara para sustituciones diferentes en la misma posición.
3. máscara estructural dura.
4. máscara estructural ponderada o soft masking.

Umbrales iniciales de referencia, configurables y sometidos a ablación:

- distancia Cα–Cα ≤ 8 Å;
- contacto de átomos pesados ≤ 4.5 Å;
- `alpha` inicial de soft masking en `{0.25, 0.5}`.

Para cada batch debe poder construirse conceptualmente una matriz de pesos
`W_ij`:

- `W_ij = 1` para negativos válidos;
- `W_ij = 0` para hard masking;
- `W_ij = alpha` para soft masking.

Los positivos entre las dos vistas del mismo mutante nunca deben enmascararse.

Debe registrarse:

- negativos potenciales;
- negativos válidos;
- proporción de negativos conservados;
- fracción enmascarada;
- número de pares eliminados por cada criterio;
- número de pares ponderados por cada criterio;
- motivo del enmascaramiento.

Si un ancla conserva menos de 8 negativos válidos o menos del 25 % de los
negativos potenciales, el batch debe reconstituirse mediante muestreo por
posición o vecindario. Si el problema persiste, debe compararse con una
alternativa sin negativos explícitos, como VICReg o Barlow Twins, en lugar de
calcular una NT-Xent degenerada.

La máscara de falsos negativos actúa dentro del batch. `leave-neighborhood-out`
actúa en la partición de train, validation y test. Son controles distintos y
ninguno sustituye al otro.

Criterios documentales de aceptación:

- el baseline NT-Xent sin máscara funciona y supera el smoke test;
- las cuatro configuraciones se comparan manteniendo constantes encoder,
  augmentations, split, seeds y presupuesto de entrenamiento;
- se registra el número de negativos válidos por ancla;
- no aparecen anchors con denominadores degenerados;
- las máscaras no provocan colapso;
- la configuración seleccionada mejora o mantiene la estabilidad entre seeds;
- se preservan suficientes negativos informativos;
- la comparación se realiza también frente a VICReg o Barlow Twins si el
  número efectivo de negativos resulta insuficiente.

## 14.3. `L_relative_WT`

Modalidades:

```text
none
margin
ranking
predictive
strong_positive_control
```

Configuración base:

```text
lambda_wt = 0
```

No utilizar una atracción simple Mutante–WT que colapse las diferencias.

## 14.4. `L_delta`

Opciones:

- consistencia de `z_delta` entre vistas;
- regularización de varianza/covarianza;
- predicción de `diff_bioq`;
- predicción de `diff_struct_node`;
- predicción de deltas de shells;
- predicción de contactos validados.

No usar `custom_structure_energy` como objetivo principal.

## 14.5. Reconstrucción enmascarada

La reconstrucción enmascarada no forma parte de la primera implementación
mínima del encoder, pero sí es una incorporación de corto plazo y prioridad
media-alta. Debe implementarse después de disponer de Dataset validado, encoder
compartido funcional, baseline autosupervisado reproducible, auditoría básica
de gradientes y smoke test con checkpointing funcional.

Puede reconstruir, mediante configuraciones separadas:

- features de nodo;
- categorías de contacto;
- bins de distancia;
- otros descriptores estructurales validados.

No debe reconstruir automáticamente:

- identificadores;
- posiciones absolutas;
- `is_mutation`;
- máscaras de disponibilidad;
- targets;
- `custom_structure_energy`;
- confusores globales;
- variables cuya reconstrucción produzca shortcuts triviales.

Debe existir una máscara explícita que distinga:

- valores observados;
- valores ocultados para la tarea;
- valores originalmente ausentes o no calculables.

La pérdida reconstructiva debe calcularse únicamente sobre elementos ocultados
y válidos.

Auditoría obligatoria del decoder:

- sus parámetros deben pertenecer al optimizador;
- debe registrarse qué término de pérdida lo entrena;
- debe recibir gradientes no nulos;
- no debe presentar gradientes `None`, NaN o Inf;
- sus pesos deben cambiar durante el entrenamiento;
- `L_masked_reconstruction` debe retropropagar tanto al decoder como al
  encoder;
- no debe existir un `detach` accidental entre encoder y decoder;
- la rama reconstructiva debe marcarse como `trained`, `inactive`, `failed` o
  `not_applicable` en `run_manifest.json`.

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

## 14.6. Función híbrida

Una posible formulación:

```text
L_total =
    L_main
  + lambda_rec * L_masked_reconstruction
  + lambda_var * L_variance_covariance
  + lambda_wt * L_relative_WT
  + lambda_delta * L_delta
```

Cada término debe poder desactivarse.

---

# 15. Augmentations

## 15.1. Positivos

Los positivos principales son:

```text
vista_1 del mismo mutante
vista_2 del mismo mutante
```

## 15.2. Requisitos

Las augmentations:

- no deben cambiar la identidad de variante;
- no deben eliminar el nodo mutado;
- no deben romper `edge_index`;
- no deben introducir NaN;
- deben ser reproducibles por seed;
- deben preservar la correspondencia necesaria.

## 15.3. WT

WT no es una augmentation del mutante.

El WT se usa como referencia relacional o auxiliar.

---

# 16. Splits

## 16.1. Split inicial

```text
leave_position_out
```

Todas las mutaciones de una misma posición deben quedar en el mismo fold.

## 16.2. Split por vecindario

```text
leave_neighborhood_out
```

Agrupa posiciones estructuralmente próximas sobre el WT.

Umbral inicial orientativo:

```text
Cα–Cα ≤ 12 Å
```

o:

```text
distancia de grafo ≤ 2
```

Debe ser configurable.

## 16.3. Splits futuros

- leave-domain-out;
- leave-structural-region-out.

## 16.4. Prevención de leakage

Debe comprobarse:

- ninguna posición compartida;
- ningún grupo estructural compartido;
- mismos IDs no duplicados;
- escaladores ajustados solo en train;
- clustering evaluado sin utilizar test para seleccionar hiperparámetros;
- energía no usada para definir etiquetas ocultas.

---

# 17. Auditoría de gradientes

## 17.1. Módulos obligatorios

- encoder;
- `MLP_fusion`;
- `projection_instance`;
- `projection_pair`;
- `MLP_delta`;
- `bio_head`;
- decoder;
- módulo WT auxiliar;
- cabezas futuras.

## 17.2. Campos por módulo

- nombre;
- número de parámetros;
- grupo del optimizador;
- pérdidas conectadas;
- media de norma de gradiente;
- mediana;
- mínimo;
- máximo;
- fracción `None`;
- fracción cero;
- NaN/Inf;
- cambio relativo de pesos;
- estado.

## 17.3. Estados

```text
trained
inactive
failed
not_applicable
```

## 17.4. Regla

Una salida paramétrica no puede interpretarse como aprendida si su módulo no figura como `trained`.

---

# 18. Auditorías anti-shortcut

## 18.1. Confusores

- `custom_structure_energy`;
- `custom_complex_energy_phenotype`;
- número de nodos;
- número de aristas;
- densidad;
- `log(N)`;
- cobertura;
- residuos ausentes;
- dominio;
- posición;
- vecindario;
- calidad;
- severidad.

## 18.2. Pruebas

- Spearman con PC1;
- Spearman con otras PCs;
- correlación con UMAP como diagnóstico;
- asociación con `severity`;
- asociación con distancia Mutante–WT;
- energía eliminada;
- energía permutada;
- globales-only;
- permutation importance;
- asociación cluster–energía;
- asociación cluster–dominio;
- clusters degenerados;
- singletons;
- estabilidad al retirar confusores.

## 18.3. Criterio

Ningún confusor debe explicar por sí solo la geometría del espacio seleccionado.

---

# 19. Exportación de embeddings

## 19.1. Exportar siempre que existan

- `h_encoder_mut`;
- `h_encoder_wt`;
- `r_delta`;
- `severity`;
- `mechanism_direction`;
- `z_instance`;
- metadatos;
- split;
- seed;
- checkpoint.

## 19.2. Exportación condicional

- `z_delta`: solo validada;
- `z_instance_pair`: solo entrenada;
- `z_bio`: solo entrenada;
- `H_mut` y `H_WT`: cuando se solicite análisis nodal.

## 19.3. Formatos

Se permiten:

- `.npy`;
- `.npz`;
- `.csv`;
- `.parquet`.

Debe existir una tabla de metadatos alineada por fila.

---

# 20. Clustering

## 20.1. Espacios a comparar

1. `h_encoder_mut`;
2. `r_delta`;
3. `z_delta` validada;
4. `concat(h_encoder_mut, r_delta)`;
5. `concat(h_encoder_mut, z_delta validada)`;
6. `mechanism_direction`;
7. espacio residualizado respecto a `severity`;
8. `z_instance` como control técnico.

## 20.2. Regla principal

El clustering debe realizarse en el espacio original normalizado.

UMAP es principalmente visualización.

## 20.3. Métricas

- silhouette;
- Davies–Bouldin;
- Calinski–Harabasz;
- tamaños de cluster;
- singletons;
- balance;
- estabilidad entre seeds;
- bootstrap;
- consenso;
- preservación de vecinos.

## 20.4. Rechazo

Una solución debe marcarse como degenerada cuando:

- aparece un singleton dominante;
- un cluster contiene casi todos los casos;
- la estabilidad es muy baja;
- el resultado depende de un único confusor.

---

# 21. `run_manifest.json`

## 21.1. Creación

Debe crearse al inicio del run, actualizarse durante la ejecución y cerrarse al
final.

Flujo obligatorio:

1. crear el esquema y el escritor inicial de `run_manifest.json`;
2. iniciar el manifiesto antes de cargar o entrenar el modelo;
3. actualizarlo durante la ejecución;
4. cerrarlo después de auditorías y exportaciones.

## 21.2. Identificación

- `run_id`;
- inicio;
- fin;
- commit;
- working tree;
- Python;
- PyTorch;
- PyG;
- dispositivo;
- dataset;
- esquema HDF5;
- checksum.

Campos mínimos desde el inicio del run:

- `run_id`;
- fecha y hora de inicio;
- versión o commit del código cuando Git esté disponible;
- versión de Python y dependencias principales;
- dispositivo;
- ruta o identificador del dataset;
- versión del esquema HDF5;
- checksum o identificador de entradas cuando sea posible;
- configuración serializada;
- seed global;
- seeds específicos;
- split seleccionado;
- identificadores o grupos de train, validation y test;
- grupos de features solicitados;
- lista exacta de node features solicitadas;
- lista exacta de edge features solicitadas;
- globales solicitados;
- módulos que se pretenden activar;
- pérdidas que se pretenden utilizar.

## 21.3. Datos

- node features;
- edge features;
- graph features;
- grupos activos;
- máscaras;
- missing values;
- pairing;
- cobertura;
- calidad.

## 21.4. Arquitectura

- encoder;
- capas;
- dimensiones;
- poolings;
- fusión;
- `projection_instance`;
- `projection_pair`;
- `MLP_delta`;
- decoder;
- módulos activos;
- módulos congelados.

Durante el run debe poder actualizarse para registrar:

- dimensiones reales detectadas;
- features realmente cargadas;
- módulos realmente instanciados;
- módulos activos e inactivos;
- parámetros incluidos en el optimizador;
- pérdidas conectadas a cada módulo;
- checkpoints generados;
- progreso y estado del entrenamiento;
- auditorías parciales;
- incidencias o warnings relevantes.

## 21.5. Entrenamiento

- pérdida principal;
- pérdidas auxiliares;
- lambdas;
- optimizador;
- scheduler;
- batch size;
- épocas;
- early stopping;
- mixed precision;
- clipping;
- augmentations;
- máscara de falsos negativos.

## 21.6. Splits y seeds

- tipo de split;
- grupos;
- train;
- validation;
- test;
- seed global;
- seeds por librería;
- seed de clustering.

## 21.7. Auditorías

- gradientes;
- cambio de pesos;
- NaN/Inf;
- anti-shortcut;
- degeneración;
- aceptación;
- motivo de rechazo.

## 21.8. Artefactos

- checkpoints;
- métricas;
- embeddings;
- clustering;
- figuras;
- vecinos;
- auditorías;
- informes.

Al cierre del run debe incorporar, como mínimo:

- fecha y hora de finalización;
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

# 22. Configuración base

La configuración base debe ser conservadora:

```yaml
data:
  chain_id: A
  use_quality_as_input: false
  use_global_energy: false

model:
  encoder_type: nnconv

  projection_instance:
    enabled: true

  mlp_delta:
    enabled: false

  projection_pair:
    enabled: false

loss:
  main: nt_xent
  lambda_rec: 0.0
  lambda_var: 0.0
  lambda_wt: 0.0
  lambda_delta: 0.0

split:
  type: leave_position_out
```

No deben existir rutas personales fijas.

---

# 23. Tests obligatorios

## 23.1. Dataset

- una muestra;
- batch;
- shapes;
- dtypes;
- NaN/Inf;
- `edge_index`;
- nodo mutado;
- pairing;
- máscaras;
- missing values;
- rechazo.

## 23.2. Encoder

- pesos compartidos;
- edge sensitivity;
- CPU;
- grafos heterogéneos;
- gradientes;
- shapes.

## 23.3. Relacional

- definición exacta de `r_delta`;
- dimensión 5d;
- sin parámetros;
- `severity`;
- dirección;
- epsilon;
- `MLP_delta` on/off.

## 23.4. Pérdidas

- positivos correctos;
- WT no positivo fuerte por defecto;
- máscara hard;
- máscara soft;
- gradientes;
- colapso;
- NaN/Inf.

## 23.5. Auditoría

- rama conectada;
- rama desconectada;
- cambio de pesos;
- estado `trained`;
- estado `inactive`.

## 23.6. Smoke test

Con 4–8 pares, 2 épocas y batch pequeño:

- forward;
- backward;
- checkpoint;
- resume;
- exportación;
- manifiesto;
- CPU;
- sin NaN/Inf.

---

# 24. Criterios de aceptación

## 24.1. Dataset

Aceptado si:

- esquema validado;
- pairing correcto;
- node/edge features consistentes;
- máscaras presentes;
- casos inválidos reportados.

## 24.2. Encoder

Aceptado si:

- pesos compartidos demostrados;
- `edge_attr` afecta la salida;
- gradientes correctos;
- CPU funcional;
- batch heterogéneo funcional.

## 24.3. `r_delta`

Aceptado si:

- definición exacta;
- dimensión correcta;
- tests numéricos;
- exportación alineada;
- sin parámetros.

## 24.4. `z_delta`

Aceptada solo si:

- pérdida explícita;
- optimizador;
- gradientes;
- cambio de pesos;
- sin colapso;
- comparación favorable con `r_delta`;
- manifest completo.

## 24.5. Clustering

Aceptado si:

- espacio original normalizado;
- sin solución degenerada;
- estabilidad entre seeds;
- ningún confusor domina;
- enriquecimiento interpretable.

## 24.6. Run

Aceptado si:

- `run_manifest.json` completo;
- tests pasados;
- checkpoint reproducible;
- embeddings exportados;
- auditorías completas;
- sin NaN/Inf;
- sin leakage.

---

# 25. Fases de implementación

## Fase 1. Configuración e infraestructura mínima

Archivos:

- `config.py`;
- `manifest.py`;
- `reproducibility.py`;
- `logging.py`;
- validación de configuración.

Criterio:

- configuración válida;
- seeds registradas;
- `run_id` único;
- manifiesto inicial creado antes de cualquier entrenamiento;
- sin rutas personales fijas;
- funcionamiento en CPU.

## Fase 2. Parsing y validación HDF5

Archivos:

- `validation.py`;
- `audit_dataset.py`;
- utilidades de parsing HDF5;
- tests de HDF5.

Criterio:

- informe reproducible de auditoría;
- entries inválidas identificadas;
- no se modifican los HDF5 originales;
- resultados registrados en el manifiesto.

## Fase 3. Dataset y collate

Archivos:

- `hdf5_dataset.py`;
- `pairing.py`;
- `collate.py`.

Criterio:

- cada muestra devuelve `graph_mut` y `graph_wt`;
- metadatos explícitos;
- batches válidos con grafos de tamaño heterogéneo;
- sin globales implícitos;
- sin `custom_structure_energy`;
- sin `log(N)`;
- sin calidad como input base;
- features realmente utilizadas registradas.

## Fase 4. Splits

Archivos:

- `splits.py`;
- tests anti-leakage.

Criterio:

- ninguna posición aparece en más de una partición;
- grupos y seeds quedan registrados;
- el split puede reproducirse.

## Fase 5. Encoder compartido

Archivos:

- `encoder.py`;
- `pooling.py`;
- `model.py`.

Criterio:

- encoder compartido;
- edge-aware;
- producción de `H_mut` y `H_WT`;
- poolings funcionales;
- fusión explícita;
- producción de `h_encoder_mut` y `h_encoder_wt`;
- sin `proj_bio`;
- sin globales;
- sin energía;
- sin `log(N)`;
- CPU.

## Fase 6. Representaciones deterministas

Archivos:

- `relational.py`;
- tests relacionales;
- exportación preliminar.

Criterio:

- `h_encoder_mut`;
- `h_encoder_wt`;
- `r_delta`;
- `severity`;
- `mechanism_direction`;
- definición exacta de cinco bloques en `r_delta`;
- ausencia de parámetros en `r_delta`;
- resultados deterministas;
- shapes verificadas;
- direcciones inválidas o inestables marcadas explícitamente.

`r_delta`, `severity` y `mechanism_direction` no dependen de las máscaras de
falsos negativos. Deben estar disponibles desde el primer baseline
autosupervisado para comparar espacios.

`mechanism_direction` debe manejar explícitamente los casos con `severity ≈ 0`
mediante epsilon configurable, máscara de validez o indicador de dirección no
fiable.

## Fase 7. Baseline autosupervisado

Archivos:

- `projection.py`;
- `contrastive.py`;
- `trainer.py`;
- `checkpointing.py`;

Criterio:

- dos vistas del mismo mutante;
- NT-Xent;
- smoke test;
- gradientes;
- checkpoints.

## Fase 8. Auditoría inicial

Archivos:

- `gradient_audit.py`;
- `optimizer_audit.py`.

Criterio:

- pertenencia al optimizador;
- conectividad de pérdidas;
- gradientes por módulo;
- cambio relativo de pesos;
- NaN/Inf;
- detección de colapso;
- estado por módulo;
- encoder y `projection_instance` marcados como entrenados;
- módulos inactivos identificados;
- resultados incorporados al manifiesto.

## Fase 9. Exportación conforme

Archivos:

- `export_embeddings.py`;
- `embedding_registry.py`.

Criterio:

- exportación de `h_encoder_mut`, `h_encoder_wt`, `z_instance`, `r_delta`,
  `severity` y `mechanism_direction`;
- shapes y normalización registradas;
- checkpoint de origen registrado;
- IDs y representación usada posteriormente registradas en el manifiesto;
- `z_delta` todavía no exportada.

## Fase 10. Máscaras de falsos negativos

- máscaras de falsos negativos inmediatamente después del baseline NT-Xent sin
  máscara;
- comparación entre `none`, `same_position`, `structural_hard` y
  `structural_soft`;
- control de negativos válidos;
- estrategia ante batches degenerados.

Criterio:

- comparación entre NT-Xent sin máscara, `same_position`, `structural_hard` y
  `structural_soft`;
- negativos válidos por ancla;
- ausencia de denominadores degenerados;
- estabilidad reproducible;
- comparación frente a VICReg o Barlow Twins si el número efectivo de
  negativos es insuficiente.

## Fase 11. `leave-neighborhood-out`

- agrupación estructural;
- separación completa de vecindarios entre particiones;
- comparación frente a `leave-position-out`.

## Fase 12. Reconstrucción enmascarada

- decoder ligero;
- máscara explícita de observados, ocultados y ausentes;
- predicción de features nodales y, por ablación separada, contactos o bins de
  distancia;
- auditoría de optimizador, gradientes y cambio de pesos.

Criterio:

- baseline sin reconstrucción frente a baseline con reconstrucción;
- mejor régimen sin negativos explícitos frente a su versión con
  reconstrucción;
- ausencia de shortcuts y valor añadido reproducible.

## Fase 13. Ruta relacional aprendida

- `MLP_delta`;
- `L_delta`;
- `projection_pair`;
- contraste del par.

Criterio:

- `z_delta` entrenada;
- auditoría completa;
- comparación favorable con `r_delta`.

## Fase 14. Extensiones

- dominios;
- shells;
- red;
- dinámica;
- ESM/LLR;
- datos experimentales.

Cada extensión requiere:

- configuración separada;
- tests;
- ablación;
- criterio de aceptación;
- actualización del manifiesto.

---

# 26. Artefactos esperados por run

```text
runs/<run_id>/
├── run_manifest.json
├── config_resolved.yaml
├── train.log
├── metrics.csv
├── metrics.json
├── gradient_audit.json
├── dataset_audit.json
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── embeddings/
├── clustering/
├── shortcuts/
└── figures/
```

---

# 27. Prohibiciones

No se permite en la configuración base:

- energía global como input;
- calidad como input;
- WT como positivo fuerte;
- pseudo-labels de energía;
- `z_delta` no entrenada;
- `z_bio` no entrenada;
- rutas absolutas;
- split aleatorio con leakage;
- clustering principal sobre UMAP;
- etiquetas funcionales no validadas;
- módulos futuros presentados como implementados.

---

# 28. Regla final de implementación

Ante cualquier ambigüedad:

1. no inventar;
2. consultar `sample_schema.json`;
3. consultar el informe;
4. registrar la duda;
5. solicitar aclaración;
6. implementar el cambio mínimo;
7. añadir tests;
8. ejecutar auditorías;
9. actualizar `run_manifest.json`;
10. comparar con el baseline anterior.

La referencia relacional por defecto es siempre:

```text
r_delta
```

La representación:

```text
z_delta
```

solo se acepta cuando `MLP_delta` ha sido realmente entrenada, auditada y validada frente a `r_delta`.
