# Muestra mínima de HDF5

Esta carpeta contiene una muestra pequeña y versionable del dataset del proyecto
GNN siamés Mutante–WT para PKP2. Su finalidad es permitir que Codex, las pruebas
unitarias y los scripts de auditoría conozcan el formato real de entrada sin
subir a GitHub los HDF5 completos.

## Contenido

```text
sample_data/
├── README.md
├── manifest.json
├── sample_schema.json
└── examples/
    ├── mutante_1.hdf5
    ├── wt_companion_1.hdf5
    ├── mutante_2.hdf5
    ├── wt_companion_2.hdf5
    └── caso_con_missing.hdf5
```

## Ejemplos incluidos

| Archivo | Papel | Posición | Nodos | Aristas |
|---|---|---:|---:|---:|
| `mutante_1.hdf5` | Mutante missense normal G100D | 100 | 22 | 69 |
| `wt_companion_1.hdf5` | WT companion de G100D | 100 | 27 | 95 |
| `mutante_2.hdf5` | Mutante C563W con grafo grande | 563 | 62 | 340 |
| `wt_companion_2.hdf5` | WT companion de C563W | 563 | 52 | 264 |
| `caso_con_missing.hdf5` | Caso límite M1T con grafo pequeño y máscaras | 1 | 10 | 31 |

Los dos pares presentan números de nodos distintos entre sí y también entre
Mutante y WT companion. El Dataset no debe asumir igualdad de `N` o `E`.

## Caso con información no disponible

Los HDF5 originales no contienen NaN. La información no calculable se representa
mediante datasets `mask_*`. En `caso_con_missing.hdf5`,
`mask_diff_hbond_count` y `mask_diff_hydrophobicity` valen cero en todos los
nodos. Esto permite probar que el Dataset:

1. conserva las máscaras;
2. distingue un valor no disponible de un cambio real igual a cero;
3. no introduce NaN artificiales;
4. no usa las máscaras como features del encoder base salvo configuración
   explícita.

## Estructura de cada archivo

Cada HDF5 de muestra contiene exactamente un grupo raíz, cuyo nombre es el
`variant_id`. Dentro aparecen:

- `node_features/`;
- `edge_features/`;
- `graph_features/`.

La conectividad se almacena en `edge_features/_index` con shape `[E, 2]`.
Para PyTorch Geometric debe transponerse a `[2, E]`.

`is_mutation` no está almacenado como dataset. El loader actual lo construye en
tiempo de ejecución y lo añade como última columna de `x`.

El emparejamiento con el WT se realiza por posición. Los nombres completos de
aminoácidos y la posición se parsean desde la clave raíz.

## Reglas de uso

- Usar estos archivos únicamente para desarrollo, tests, smoke tests y auditoría.
- No estimar métricas científicas ni entrenar el modelo final con esta muestra.
- No sustituir `sample_schema.json` por deducciones realizadas desde el informe.
- No subir `proc_483p.hdf5` ni `wt_companion.hdf5` completos al repositorio.
- No tratar `custom_structure_energy` como target ni como input base automático.
- Validar la dimensionalidad de las features antes de formar batches.

## Comprobación rápida

```python
from pathlib import Path
import h5py

for path in sorted(Path("sample_data/examples").glob("*.hdf5")):
    with h5py.File(path, "r") as handle:
        keys = list(handle.keys())
        assert len(keys) == 1
        graph = handle[keys[0]]
        assert "node_features" in graph
        assert "edge_features" in graph
        assert "graph_features" in graph
        assert "_index" in graph["edge_features"]
        print(path.name, keys[0])
```

Consulta `sample_schema.json` para ver la jerarquía, los nombres y el orden de
features, dtypes, shapes, máscaras, reglas de `is_mutation` y emparejamiento
Mutante–WT.
