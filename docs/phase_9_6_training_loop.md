# Subfase 9.6.E: training loop mínimo de producción

Esta fase implementa un loop multiepoch mínimo en `gnn_siamese.training` para
usar `TotalLossAssembler` y `training_step` en un entrenamiento reproducible y
auditable sin abrir todavía una infraestructura pesada de producción.

## Qué implementa

- `TrainingLoopConfig`: configuración mínima del loop.
- `TrainingLoopOutput`: salida estructurada del entrenamiento.
- `fit`: loop multiepoch con forward, backward, `optimizer.step`, clipping
  opcional, scheduler opcional y parada por loss no finita.
- `build_run_manifest`: manifiesto ligero con estado del run y de las ramas de
  pérdida.
- `batch_adapter`: punto de adaptación simple para desacoplar el loop del
  dataset real en esta fase.

## Conexión con `TotalLossAssembler`

El loop no reimplementa la lógica de pérdidas. La composición sigue siendo
explícita en `TotalLossAssembler`:

- `nt_xent` es el baseline por defecto.
- `relative_wt` solo se considera activa si su peso es mayor que cero y su modo
  no es `none`.
- `delta` solo se considera activa si su peso es mayor que cero, su modo no es
  `none` y existe `z_delta`.

El loop registra componentes `active`, `inactive` y `skipped` a partir de la
salida del ensamblador.

## Qué significa baseline en esta fase

El baseline reproducible sigue siendo conservador:

- solo `nt_xent` activa por defecto;
- `relative_wt` desactivada por `lambda_wt = 0` o `mode: none`;
- `delta` desactivada por `lambda_delta = 0`, `enabled: false` o `mode: none`;
- `z_delta` no se interpreta como espacio aprendido cuando `delta` no está
  activa;
- WT no actúa como positivo fuerte por defecto;
- `custom_structure_energy` no se usa como objetivo principal.

## Qué registra el manifest

`build_run_manifest` registra de forma ligera:

- `run_name`;
- épocas configuradas y completadas;
- `num_steps`;
- componentes de loss `active`, `inactive` y `skipped`;
- estado de activación de `relative_wt` y `delta`;
- detección de losses no finitas;
- `seed`, si existe;
- estado de reconstrucción como `disabled/pending`;
- timestamp;
- versión simple del código mediante commit Git y estado sucio del árbol cuando
  se puede obtener.

## Qué queda fuera

Esta subfase no implementa todavía:

- reconstrucción enmascarada;
- checkpointing avanzado;
- validación/early stopping completo;
- auditoría profunda de gradientes por módulo;
- exportación de embeddings de producción;
- evaluación biológica o clustering final;
- integración profunda con Colab;
- entrenamiento largo o experimentos reales.

La reconstrucción enmascarada sigue pendiente y permanece fuera del baseline de
esta fase.
