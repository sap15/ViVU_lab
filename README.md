# Muestra HDF5 del proyecto GNN siamés PKP2

## 1. Finalidad de esta carpeta

La carpeta `sample_data/` contiene la documentación y los ejemplos mínimos necesarios para implementar y probar:

```text
src/gnn_siamese/data/hdf5_dataset.py
src/gnn_siamese/data/pairing.py
src/gnn_siamese/data/validation.py
tests/test_hdf5_dataset.py
tests/test_pairing.py
```

Los ejemplos permiten comprobar cómo:

- se identifica cada variante;
- se empareja un mutante con su WT companion;
- se recuperan las features de nodos, aristas y grafo;
- se localiza el nodo mutado;
- se interpretan las máscaras de disponibilidad;
- se distinguen datos reales y datos sintéticos;
- se rechazan entradas estructuralmente inválidas.

Este archivo debe leerse junto con:

```text
sample_data/sample_schema.json
docs/especificacion_modelo.md
AGENTS.md
```

`sample_schema.json` es la referencia para la jerarquía, shapes y dtypes. Este README documenta la semántica de los archivos y de los campos.

---

## 2. Archivos reales utilizados como referencia

Los archivos de referencia actualmente disponibles son:

```text
proc_483p.hdf5
wt_companion.hdf5
```

### 2.1. `proc_483p.hdf5`

Tipo:

```text
ejemplo real del proyecto
```

Contenido:

- 484 grafos de mutantes missense de PKP2;
- una entrada HDF5 por variante;
- cadena `A`;
- resolución de residuo;
- features de nodo;
- features de arista;
- features globales;
- grupos de máscaras y diferencias;
- grupo `target_values`, actualmente vacío en las entradas inspeccionadas.

Ejemplos de claves reales:

```text
residue-srv:A:100:Glycine->Aspartate:pos_100_G_D
residue-srv:A:100:Glycine->Serine:pos_100_G_S
residue-srv:A:101:Arginine->Cysteine:pos_101_R_C
residue-srv:A:101:Arginine->Histidine:pos_101_R_H
residue-srv:A:101:Arginine->Leucine:pos_101_R_L
```

Cada clave representa un mutante distinto.

### 2.2. `wt_companion.hdf5`

Tipo:

```text
ejemplo real del proyecto
```

Contenido:

- 356 grafos WT companion;
- una entrada WT contextualizada por posición;
- mismo esquema general de grupos que el archivo de mutantes;
- aminoácido WT repetido como origen y destino;
- identificador final `PKP2_WT`.

Ejemplos de claves reales:

```text
residue-srv:A:100:Glycine->Glycine:PKP2_WT
residue-srv:A:101:Arginine->Arginine:PKP2_WT
residue-srv:A:102:Serine->Serine:PKP2_WT
residue-srv:A:103:Proline->Proline:PKP2_WT
residue-srv:A:105:Proline->Proline:PKP2_WT
```

El WT companion no es una segunda variante. Es la referencia WT correspondiente a la posición del mutante.

---

## 3. Relación entre mutante y WT companion

El emparejamiento se realiza inicialmente por:

```text
chain_id + position + wt_aa
```

Ejemplo:

```text
Mutante:
residue-srv:A:100:Glycine->Aspartate:pos_100_G_D

WT companion:
residue-srv:A:100:Glycine->Glycine:PKP2_WT
```

Los campos coincidentes son:

```text
chain_id = A
position = 100
wt_aa = Glycine / G
```

La diferencia es:

```text
mut_aa = Aspartate / D
```

Para otra sustitución en la misma posición:

```text
residue-srv:A:100:Glycine->Serine:pos_100_G_S
```

se utiliza el mismo WT companion de la posición 100:

```text
residue-srv:A:100:Glycine->Glycine:PKP2_WT
```

En los dos archivos reales inspeccionados, todas las posiciones presentes en `proc_483p.hdf5` disponen de una entrada correspondiente en `wt_companion.hdf5`.

El Dataset debe validar esta correspondencia y no asumirla silenciosamente.

---

## 4. Obtención de `variant_id`

### 4.1. Forma canónica recomendada

La forma canónica de `variant_id` será:

```text
pos_<posición>_<AA_WT_1letra>_<AA_MUT_1letra>
```

Ejemplos:

```text
pos_100_G_D
pos_100_G_S
pos_101_R_C
pos_101_R_H
```

### 4.2. Extracción desde la clave HDF5

La clave:

```text
residue-srv:A:100:Glycine->Aspartate:pos_100_G_D
```

se interpreta como:

```text
prefix      = residue-srv
chain_id    = A
position    = 100
wt_aa_name  = Glycine
mut_aa_name = Aspartate
variant_id  = pos_100_G_D
```

La implementación debe conservar por separado:

```text
variant_id
chain_id
position
wt_aa
mut_aa
hdf5_key
source_hdf5
wt_companion_key
```

### 4.3. Regla de prioridad

Para un mutante:

1. usar el identificador final `pos_<...>` si la clave lo contiene;
2. validar que posición y aminoácidos concuerdan con la parte central de la clave;
3. reconstruir el identificador solo si falta;
4. rechazar o advertir si las dos representaciones son contradictorias.

Para un WT companion no debe utilizarse `PKP2_WT` como `variant_id` del mutante. Debe conservarse como:

```text
wt_companion_id = PKP2_WT
```

junto con la posición contextual.

---

## 5. Localización del nodo mutado

## 5.1. Regla actual de los archivos reales

En los mutantes missense, el nodo mutado se localiza mediante los canales diferenciales almacenados en:

```text
<graph_key>/node_features/
```

La implementación heredada utiliza como sondas, en este orden:

```text
diff_mass
diff_charge
diff_pI
diff_size
```

Se selecciona el primer índice de nodo para el que:

```text
abs(diff_feature) > epsilon
```

con un valor pequeño de `epsilon`.

La posición encontrada debe ser única. Si una sonda tiene más de un nodo no nulo, debe generarse una advertencia o rechazarse la muestra según la política de validación.

## 5.2. Ejemplo real

Para:

```text
residue-srv:A:100:Glycine->Aspartate:pos_100_G_D
```

los canales `diff_mass`, `diff_charge`, `diff_pI`, `diff_size` y `diff_polarity` señalan el mismo nodo.

Ese nodo corresponde al residuo mutado.

## 5.3. Creación de `is_mutation`

El loader debe construir o validar un canal binario:

```text
is_mutation
```

con la semántica:

```text
1.0 = nodo mutado
0.0 = cualquier otro nodo
```

Para un mutante missense válido:

```text
sum(is_mutation) == 1
```

Para un WT companion:

```text
sum(is_mutation) == 0
```

## 5.4. Metadatos de residuo

En estos HDF5:

```text
node_features/_position
```

contiene coordenadas tridimensionales y no debe interpretarse como número de residuo.

El identificador textual de cada nodo se encuentra en:

```text
node_features/_name
```

Ejemplo:

```text
pos_100_G_D A 100
```

De este texto puede recuperarse:

```text
variant_id = pos_100_G_D
chain_id = A
residue_number = 100
```

El loader no debe confundir coordenadas de `_position` con numeración de secuencia.

## 5.5. WT companion

En el WT companion, los canales `diff_*` deben ser cero o estar marcados como no aplicables y:

```text
is_mutation = 0
```

en todos los nodos.

El WT companion está contextualizado por la posición, pero no contiene una mutación real.

---

## 6. Jerarquía de cada entrada HDF5

Cada clave de grafo contiene:

```text
<graph_key>/
├── node_features/
├── edge_features/
├── graph_features/
└── target_values/
```

### 6.1. `node_features`

Contiene:

- metadatos de nodo;
- identidad del residuo;
- propiedades bioquímicas;
- propiedades estructurales;
- features diferenciales;
- máscaras;
- variables estructurales auxiliares;
- codificación del residuo variante.

Lista observada en los archivos reales:

```text
_chain_id
_name
_position
bsa
diff_charge
diff_hb_acceptors
diff_hb_donors
diff_hbond_count
diff_hydrophobicity
diff_mass
diff_pI
diff_polarity
diff_size
hb_acceptors
hb_donors
hse
hydrophobicity
is_truncation_node
mask_diff
mask_diff_charge
mask_diff_hb_acceptors
mask_diff_hb_donors
mask_diff_hbond_count
mask_diff_hydrophobicity
mask_diff_mass
mask_diff_pI
mask_diff_polarity
mask_diff_size
polarity
res_charge
res_depth
res_depth_missing
res_id
res_id_norm
res_mass
res_pI
res_size
res_type
rsa
sasa
sec_struct
var_HSE
var_SASA
var_SSnum
var_contact_count_rings_canon
var_contact_count_rings_coarse
var_contact_count_rings_fine
variant_res
```

### 6.2. Agrupación lógica recomendada de node features

#### Metadatos

```text
_chain_id
_name
_position
```

No deben concatenarse automáticamente en `x`.

#### Estructura

```text
bsa
hse
res_depth
rsa
sasa
sec_struct
```

#### Bioquímica e identidad

```text
hb_acceptors
hb_donors
hydrophobicity
polarity
res_charge
res_mass
res_pI
res_size
res_type
```

#### Diferencias bioquímicas Mutante–WT

```text
diff_charge
diff_hb_acceptors
diff_hb_donors
diff_hbond_count
diff_hydrophobicity
diff_mass
diff_pI
diff_polarity
diff_size
```

#### Máscaras de disponibilidad

```text
mask_diff
mask_diff_charge
mask_diff_hb_acceptors
mask_diff_hb_donors
mask_diff_hbond_count
mask_diff_hydrophobicity
mask_diff_mass
mask_diff_pI
mask_diff_polarity
mask_diff_size
res_depth_missing
```

#### Variables auxiliares estructurales

```text
var_HSE
var_SASA
var_SSnum
var_contact_count_rings_canon
var_contact_count_rings_coarse
var_contact_count_rings_fine
```

#### Localización o identidad de la variante

```text
variant_res
is_truncation_node
```

El canal `is_mutation` puede construirse durante la carga si no está almacenado explícitamente.

### 6.3. `edge_features`

Lista observada:

```text
_index
_name
covalent
distance
electrostatic
same_chain
seq_sep
vanderwaals
```

Interpretación:

```text
_index         conectividad del grafo
_name          metadatos de las aristas
covalent       relación covalente
distance       distancia geométrica
electrostatic  término electrostático
same_chain     indicador de misma cadena
seq_sep        separación en secuencia
vanderwaals    término de van der Waals
```

En estos archivos, la conectividad se almacena en:

```text
edge_features/_index
```

El loader debe aceptar y normalizar:

```text
(E, 2)
```

o:

```text
(2, E)
```

y devolver siempre a PyTorch Geometric:

```text
edge_index.shape == (2, E)
```

### 6.4. `graph_features`

Lista observada:

```text
anchor_position
custom_structure_energy
delta_chain_len
delta_nodes
graph_num_edges
graph_num_nodes
is_truncation
lost_fraction_seq
lost_residues_seq
mean_edge_distance
mut_length
stop_position
tail_lost_wt
wt_length
```

Interpretación general:

- descriptores globales del grafo;
- metadatos de tamaño;
- variables de truncación;
- cobertura o longitud;
- energía estructural computacional.

Regla de la configuración base:

```text
custom_structure_energy no entra en el encoder base
```

Debe conservarse para auditoría, estratificación y ablaciones.

### 6.5. `target_values`

En las entradas reales inspeccionadas:

```text
target_values/
```

está vacío.

El Dataset debe soportar esta situación sin inventar etiquetas.

---

## 7. Datos ausentes y máscaras

## 7.1. Principio general

Un cero puede significar:

1. ausencia real de cambio;
2. valor no disponible;
3. valor no aplicable;
4. relleno técnico.

Por ello, las features `diff_*` deben interpretarse junto con sus máscaras.

## 7.2. Casos reales observados

En los dos HDF5 reales inspeccionados:

- no se detectaron valores `NaN`;
- no se detectaron valores `Inf`;
- no se detectaron grupos de features ausentes entre entradas;
- `res_depth_missing` no marca residuos ausentes en los ejemplos inspeccionados;
- `mask_diff_hbond_count` está a cero en las entradas reales;
- `mask_diff_hydrophobicity` está a cero en las entradas reales.

Por tanto, los canales:

```text
diff_hbond_count
diff_hydrophobicity
```

deben considerarse no disponibles o no validados en esta muestra, aunque su dataset exista.

No deben interpretarse como cambios biológicos nulos.

## 7.3. Caso real representativo con disponibilidad parcial

Se utilizará como caso documentado:

```text
residue-srv:A:100:Glycine->Aspartate:pos_100_G_D
```

Archivo:

```text
proc_483p.hdf5
```

Este caso contiene el grafo completo y las features principales, pero presenta máscaras de disponibilidad nulas para:

```text
diff_hbond_count
diff_hydrophobicity
```

Por tanto, sirve para comprobar que el Dataset:

- lee la feature;
- lee la máscara;
- no confunde indisponibilidad con valor cero;
- puede excluir esos canales mediante configuración;
- conserva la máscara para auditoría.

## 7.4. Caso sintético de dato ausente

Para los tests deberá crearse un archivo pequeño, por ejemplo:

```text
synthetic_missing_feature.hdf5
```

Debe contener una copia mínima de un grafo con uno de estos escenarios controlados:

- falta un dataset opcional;
- una máscara está a cero;
- existe una feature rellenada con cero pero marcada como no disponible;
- falta una correspondencia de residuo;
- una feature tiene shape incompatible.

Este archivo será sintético y solo se usará en tests. No debe mezclarse con ejemplos biológicos reales.

---

## 8. Casos recomendados para la muestra mínima del repositorio

No debe subirse a GitHub el HDF5 completo de producción. Debe prepararse una muestra reducida con 2–5 mutantes y sus WT companions.

Selección recomendada:

| Caso | Clave mutante | WT companion | Tipo |
|---|---|---|---|
| completo 1 | `residue-srv:A:100:Glycine->Aspartate:pos_100_G_D` | `residue-srv:A:100:Glycine->Glycine:PKP2_WT` | real |
| completo 2 | `residue-srv:A:100:Glycine->Serine:pos_100_G_S` | `residue-srv:A:100:Glycine->Glycine:PKP2_WT` | real |
| misma posición | `residue-srv:A:101:Arginine->Cysteine:pos_101_R_C` | `residue-srv:A:101:Arginine->Arginine:PKP2_WT` | real |
| sustitución conservadora/parcial | `residue-srv:A:101:Arginine->Histidine:pos_101_R_H` | `residue-srv:A:101:Arginine->Arginine:PKP2_WT` | real |
| missing controlado | entrada definida en `synthetic_missing_feature.hdf5` | WT sintético correspondiente | sintético |

Esta selección permite probar:

- dos mutaciones de una misma posición;
- reutilización correcta del mismo WT companion;
- posiciones diferentes;
- shapes variables entre grafos;
- masks;
- parsing de identificadores;
- batching heterogéneo.

---

## 9. Archivos reales y sintéticos

## 9.1. Reales

```text
proc_483p.hdf5
wt_companion.hdf5
```

Proceden del pipeline real del proyecto y representan grafos de PKP2.

También son reales las entradas reducidas que se extraigan de ellos sin modificar los valores.

Nombres recomendados para copias reducidas:

```text
sample_mutants_real.hdf5
sample_wt_companions_real.hdf5
```

## 9.2. Sintéticos

Nombres recomendados:

```text
synthetic_missing_feature.hdf5
synthetic_invalid_edge_index.hdf5
synthetic_missing_wt_pair.hdf5
synthetic_multiple_mutation_nodes.hdf5
```

Estos archivos deben:

- ser pequeños;
- generarse mediante fixtures o scripts de tests;
- contener valores artificiales;
- estar claramente marcados como sintéticos;
- no utilizarse para resultados científicos;
- no incluir datos reales modificados sin dejar constancia.

Actualmente, los dos HDF5 de referencia documentados en este README son reales. Los archivos sintéticos descritos son requisitos para la futura batería de tests y todavía deben generarse explícitamente.

---

## 10. Salida esperada de `hdf5_dataset.py`

Para cada mutante, el Dataset debe devolver una estructura equivalente a:

```python
{
    "graph_mut": graph_mut,
    "graph_wt": graph_wt,
    "variant_id": "pos_100_G_D",
    "position": 100,
    "wt_aa": "G",
    "mut_aa": "D",
    "chain_id": "A",
    "mut_hdf5_key": (
        "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
    ),
    "wt_hdf5_key": (
        "residue-srv:A:100:Glycine->Glycine:PKP2_WT"
    ),
    "availability_masks": {...},
    "graph_metadata": {...},
}
```

Cada objeto PyG debe contener, cuando corresponda:

```text
x
edge_index
edge_attr
is_mutation
variant_id
position
wt_aa
mut_aa
source_hdf5
hdf5_key
```

---

## 11. Validaciones obligatorias antes de devolver una muestra

### Identidad

- la clave puede parsearse;
- `variant_id` es coherente;
- posición y aminoácidos son coherentes;
- el WT companion existe.

### Nodos

- las features tienen el mismo número de filas;
- el número de nodos es mayor que cero;
- no hay NaN ni Inf;
- el nodo mutado puede localizarse;
- existe exactamente un nodo mutado en missense;
- el WT companion tiene cero nodos marcados como mutados.

### Aristas

- `_index` existe;
- su orientación puede normalizarse;
- los índices están dentro del rango;
- el número de filas de `edge_attr` coincide con `E`;
- no hay NaN ni Inf en atributos seleccionados.

### Máscaras

- las máscaras tienen longitud compatible;
- un valor no disponible no se interpreta como cero biológico;
- las features diferenciales se seleccionan junto con sus máscaras.

### Emparejamiento

- misma cadena;
- misma posición;
- mismo aminoácido WT;
- esquema compatible;
- cobertura documentada.

---

## 12. Política de selección de features

El Dataset no debe cargar automáticamente todas las features como inputs.

La selección se realiza desde YAML.

Ejemplo conceptual:

```yaml
features:
  node_groups:
    - structure
    - biochemistry
    - diff_bioq

  edge_groups:
    - distance
    - contact
    - sequential_relation

data:
  use_global_energy: false
  use_quality_as_input: false
```

Las features de calidad, máscaras y confusores deben conservarse como metadatos aunque no entren en `x`.

---

## 13. Reglas importantes para Codex

Antes de implementar `hdf5_dataset.py`, Codex debe:

1. leer este README;
2. leer `sample_schema.json`;
3. inspeccionar las claves reales;
4. no asumir que `_position` es el número de residuo;
5. normalizar `edge_features/_index` a `(2, E)`;
6. construir o validar `is_mutation`;
7. conservar las máscaras;
8. separar inputs y confusores;
9. no utilizar `custom_structure_energy` como input base;
10. no inventar targets cuando `target_values` está vacío;
11. producir errores informativos;
12. añadir tests para una muestra y un batch;
13. añadir casos sintéticos inválidos;
14. ejecutar `pytest`.

---

## 14. Limitaciones de la muestra actual

- Los archivos reales disponibles son mayores que una muestra habitual de Git.
- No contienen un caso real con NaN o Inf.
- No contienen truncaciones en `proc_483p.hdf5`.
- Los grupos `target_values` están vacíos.
- Algunos canales diferenciales existen pero sus máscaras indican indisponibilidad.
- El número de nodos puede variar entre mutantes de la misma posición.
- La semántica final de cada feature debe validarse con `sample_schema.json` y el pipeline de generación.

Antes de publicar el repositorio, se recomienda extraer una muestra reducida y mantener los HDF5 completos fuera de Git.

---

## 15. Resumen operativo

```text
proc_483p.hdf5
    → grafos mutantes reales

wt_companion.hdf5
    → grafos WT companion reales

variant_id
    → sufijo pos_<posición>_<WT>_<MUT>

nodo mutado
    → índice único señalado por diff_* y convertido a is_mutation

node_features
    → estructura, bioquímica, diferencias, máscaras y metadatos

edge_features
    → conectividad y atributos de arista

graph_features
    → descriptores globales, calidad, tamaño y confusores

dato ausente real documentado
    → diff_hbond_count y diff_hydrophobicity con máscara 0

archivos sintéticos
    → todavía deben generarse para tests de errores controlados
```
