# Auditoría post-merge de fases 9.5 y 9.6

## Propósito

Este documento fija el estado real del repositorio tras el merge de las fases
9.5 y 9.6 en `main` y la apertura de la rama correctiva
`fix/post-merge-phase-9-5-9-6-audit`.

No reinterpreta la arquitectura objetivo como capacidad ya demostrada. Su
función es separar:

- lo implementado y cubierto por tests;
- lo declarado en YAML o documentación como objetivo futuro;
- lo que todavía falta para considerar cerrada la fase 9.6.

## Estado real de la fase 9.5

La fase 9.5 queda sustancialmente implementada en código:

- `InstanceProjectionHead` existe en `src/gnn_siamese/models/projection.py`.
- `PairProjectionHead` existe en `src/gnn_siamese/models/projection.py`.
- ambas heads no comparten pesos por defecto;
- existe validación de dimensiones;
- existe validación semántica por `input_name`;
- `SharedSiameseEncoderModel` expone:
  - `h_encoder_mut`;
  - `h_encoder_wt`;
  - `r_delta`;
  - `z_delta`;
  - `z_instance`;
  - `z_instance_pair`;
  - `severity`;
  - `mechanism_direction`.

La corrección post-merge cierra el hueco de integración añadiendo cobertura
para:

- modelo completo con encoder compartido, módulo relacional,
  `projection_instance` y `projection_pair` activos;
- forward en CPU con batch de varios grafos;
- coherencia dimensional `r_delta.shape[-1] == 5 * graph_dim`;
- desactivación segura de `projection_pair` sin romper `z_instance`;
- ruta `pair_projection_source="z_delta"` cuando `z_delta` no está validado;
- ruta `pair_projection_source="z_delta"` cuando `z_delta` sí está validado.

## Estado real de la fase 9.6

La fase 9.6 no está completa.

Lo único implementado de forma real y operativa es el baseline
`NTXentLoss` en `src/gnn_siamese/losses/contrastive.py`.

No están implementados todavía:

- false-negative masking operativo;
- `L_relative_WT` operativo;
- `L_delta` operativa;
- ensamblado real de `L_total`;
- paquete `src/gnn_siamese/training/`;
- training loop real gobernado por YAML para combinar pérdidas;
- auditoría de entrenamiento por rama conectada a una ejecución real.

La reconstrucción enmascarada permanece correctamente aplazada y desactivada.

## Qué está implementado realmente

- encoder compartido sensible a `edge_attr`;
- `r_delta`, `severity` y `mechanism_direction`;
- `MLP_delta` como módulo opcional;
- validación de `z_delta` mediante evidencia externa del manifiesto;
- `projection_instance`;
- `projection_pair`;
- baseline `NTXentLoss` para dos vistas del mismo mutante.

## Qué solo está declarado en YAML

En `configs/base.yaml` aparecen declarados, pero no deben interpretarse como
capacidades operativas completas:

- `loss.false_negative_mask`;
- `loss.relative_wt`;
- `loss.delta`;
- `training.*` como configuración objetivo;
- `model.reconstruction_decoder`;
- comparativas futuras de reconstrucción y regímenes sin negativos.

Su presencia en YAML significa intención arquitectónica y configuración
objetivo, no implementación demostrada.

## Qué falta

Para considerar cerrada la fase 9.6 faltan como mínimo:

- implementación real del enmascaramiento de falsos negativos dentro del batch;
- trazabilidad por ancla de negativos válidos, enmascarados y ponderados;
- implementación real de `L_relative_WT`;
- implementación real de `L_delta` conectada a `MLP_delta`;
- ensamblado explícito de `L_total` según YAML;
- paquete de entrenamiento con loop, optimizador, scheduler y checkpointing;
- tests de integración y smoke tests que prueben esas rutas;
- auditoría de gradientes y cambio de pesos por rama en entrenamiento real.

## Riesgo de asumir que la fase 9.6 está completa

El riesgo principal es metodológico y documental:

- se puede leer `configs/base.yaml` como si el repositorio ya soportara pérdidas
  compuestas, máscaras y ensamblado completo;
- se puede interpretar `z_delta` como representación aprendida operativa aunque
  no exista todavía un `L_delta` real en entrenamiento;
- se puede asumir erróneamente que WT ya participa en una pérdida auxiliar real;
- se puede presentar `L_total` como disponible cuando aún no existe un ensamblaje
  ejecutable gobernado por YAML.

Esto produciría una apariencia falsa de madurez experimental y de cobertura de
fases que el código todavía no demuestra.

## Plan de implementación por subfases

### A. False-negative masking

- implementar hard y soft masking dentro del batch;
- registrar `W_ij`, negativos válidos por ancla y causas de enmascaramiento;
- comparar baseline sin máscara frente a misma posición y vecindad estructural.

### B. `L_relative_WT`

- definir espacio de aplicación, distancia, margen y `stop_gradient`;
- mantener `lambda_wt = 0` como baseline;
- añadir tests que prueben que WT no se usa como positivo fuerte por defecto.

### C. `L_delta`

- conectar `MLP_delta` a una pérdida explícita;
- verificar inclusión en optimizador, gradientes, cambio de pesos y auditoría;
- impedir interpretar `z_delta` como espacio biológico sin esa evidencia.

### D. `L_total` y training loop

- crear `src/gnn_siamese/training/`;
- ensamblar pérdidas desde YAML;
- registrar módulos activos, pérdidas conectadas y auditoría de gradientes;
- soportar smoke test, checkpointing y manifiesto actualizado durante el run.

### E. Reconstrucción enmascarada futura

- mantenerla fuera del baseline actual;
- activarla solo después de baseline reproducible, smoke test y auditoría básica;
- comparar contra baseline con y sin negativos explícitos sin alterar el encoder
  base ni introducir shortcuts triviales.
