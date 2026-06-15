# Legacy

Esta carpeta conserva código histórico del proyecto para auditoría y migración
gradual.

## Archivo principal

- `legacy/modelo_gnn_siames.py`

## Estado

- procede de un notebook o script monolítico anterior orientado a Colab;
- contiene código potencialmente reutilizable para parsing HDF5, construcción
  de grafos, augmentations, encoder `NNConv` y exportación de embeddings;
- no representa todavía la arquitectura modular objetivo descrita en
  `docs/especificacion_modelo.md`;
- no constituye evidencia de que todas sus ramas estén correctamente
  entrenadas, conectadas a pérdidas o auditadas;
- no debe modificarse durante esta tarea;
- se conservará como referencia histórica para la futura migración a `src/`.

## Uso esperado

- leerlo como fuente histórica para identificar componentes reutilizables;
- contrastarlo con `AGENTS.md`, `docs/especificacion_modelo.md` y
  `docs/decisiones_arquitectonicas.md`;
- no presentarlo como implementación definitiva del repositorio.
