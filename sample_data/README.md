# sample_data

Este directorio contiene una muestra mínima de archivos HDF5 del proyecto GNN siamés Mutante–WT.

La muestra está pensada para documentar el esquema real de los HDF5, permitir pruebas unitarias, smoke tests y facilitar la revisión del formato sin incluir los datos completos de entrenamiento.

## Origen de los ejemplos

Los ejemplos de `sample_data/examples/` se han extraído de los HDF5 completos locales:

- `local_data/hdf5/proc_483p.hdf5`: HDF5 global con grafos mutantes.
- `local_data/hdf5/wt_companion.hdf5`: HDF5 global con grafos WT companion.

Estos archivos completos están excluidos por `.gitignore` y no deben subirse al repositorio.

## Archivos incluidos

| Archivo | Rol | Posición | WT | Mutante | Nodos | Aristas | Key HDF5 |
|---|---|---:|---|---|---:|---:|---|
| `pos_327_GLYCINE_GLUTAMATE_caso_con_missing_o_stop.hdf5` | Caso con máscara/missing o caso límite | 327 | GLYCINE | GLUTAMATE | 10 | 24 | `residue-srv:A:327:Glycine->Glutamate:pos_327_G_E` |
| `pos_327_GLYCINE_GLUTAMATE_wt_companion.hdf5` | WT companion | 327 | GLYCINE | GLUTAMATE | 11 | 25 | `residue-srv:A:327:Glycine->Glycine:PKP2_WT` |
| `pos_458_ISOLEUCINE_LYSINE_mutante_2.hdf5` | Mutante completo 2 | 458 | ISOLEUCINE | LYSINE | 57 | 309 | `residue-srv:A:458:Isoleucine->Lysine:pos_458_I_K` |
| `pos_458_ISOLEUCINE_LYSINE_wt_companion.hdf5` | WT companion | 458 | ISOLEUCINE | LYSINE | 51 | 264 | `residue-srv:A:458:Isoleucine->Isoleucine:PKP2_WT` |
| `pos_563_CYSTEINE_TRYPTOPHAN_mutante_1.hdf5` | Mutante completo 1 | 563 | CYSTEINE | TRYPTOPHAN | 62 | 340 | `residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W` |
| `pos_563_CYSTEINE_TRYPTOPHAN_wt_companion.hdf5` | WT companion | 563 | CYSTEINE | TRYPTOPHAN | 52 | 264 | `residue-srv:A:563:Cysteine->Cysteine:PKP2_WT` |

## Interpretación de la muestra mínima

- `pos_563_CYSTEINE_TRYPTOPHAN_mutante_1.hdf5` representa un mutante missense completo con su WT companion correspondiente.
- `pos_458_ISOLEUCINE_LYSINE_mutante_2.hdf5` representa un segundo mutante missense completo, útil para probar batching y emparejamiento Mutante–WT.
- `pos_327_GLYCINE_GLUTAMATE_caso_con_missing_o_stop.hdf5` representa el caso con máscara, valor ausente o caso límite seleccionado automáticamente durante la extracción.
- Cada posición incluida tiene su WT companion correspondiente.

## Estructura esperada de cada HDF5 individual

Cada archivo HDF5 individual contiene un único grupo raíz correspondiente a una query `residue-srv`.

Dentro de cada grupo raíz se esperan, como mínimo:

- `node_features/`: atributos de nodo a nivel de residuo.
- `edge_features/`: atributos de arista y conectividad mediante `_index`.
- `graph_features/`: atributos globales del grafo.

El archivo `sample_data/sample_schema.json` describe la jerarquía, shapes y tipos de datos observados en estos ejemplos.

## Archivos auxiliares relacionados

- `reports/sample_hdf5_candidates.tsv`: inventario de candidatos detectados en los HDF5 completos.
- `reports/sample_hdf5_extraction_report.tsv`: trazabilidad de los ejemplos extraídos.
- `scripts/extract_minimal_hdf5_examples.py`: script usado para extraer los grupos individuales.

## Nota importante

La muestra reducida no debe utilizarse como dataset final de entrenamiento. El entrenamiento real debe ejecutarse con los HDF5 completos del proyecto, almacenados fuera del repositorio o en rutas ignoradas por Git.
