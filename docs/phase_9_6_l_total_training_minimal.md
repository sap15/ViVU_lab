# Subfase 9.6.D: `L_total` mínimo y paquete `training`

Esta subfase implementa un ensamblaje mínimo y explícito de `L_total` sobre las
pérdidas ya existentes:

- `L_nt_xent`
- `L_relative_WT`
- `L_delta`

La composición actual es:

```text
L_total =
  nt_xent_weight * L_nt_xent +
  relative_wt_weight * L_relative_WT +
  delta_weight * L_delta
```

## Estados de componentes

- `active`: el componente tiene peso mayor que cero y su modo no está en `none`.
- `inactive`: el componente tiene modo `none`, aunque exista la rama.
- `skipped`: el componente tiene peso cero y no contribuye al escalar total.

La salida mínima `TotalLossOutput` conserva:

- `loss`
- `components`
- `weights`
- `active_components`
- `inactive_components`
- `skipped_components`
- `metrics`
- `audit_flags`

## Guards incluidos

- `nt_xent_weight > 0` exige `z1` y `z2`.
- `relative_wt_weight > 0` exige `h_mut` y `h_wt`.
- `delta_weight > 0` exige `z_delta`.
- Si todos los componentes quedan inactivos o con peso cero, la pérdida total
  es cero y se registra `all_components_inactive`.
- `z_delta` no se considera entrenada cuando `L_delta` está inactiva o con peso
  cero; el ensamblador marca este caso en auditoría.
- `custom_structure_energy` sigue bloqueada como target principal salvo
  opt-in explícito mediante las protecciones ya presentes en `RelativeWTLoss`
  y `DeltaLoss`.

## Paquete `training`

Se añade un paquete inicial:

```text
src/gnn_siamese/training/
```

Incluye:

- `losses.py` con `TotalLossAssembler`, `TotalLossConfig` y `TotalLossOutput`
- `step.py` con `training_step`

`training_step` solo cubre un paso mínimo:

1. forward del modelo o callable;
2. ensamblaje de `L_total`;
3. backward opcional;
4. `optimizer.step()` solo si hay componentes activos.

## Qué queda fuera

Esta fase no implementa todavía:

- entrenamiento completo de producción;
- dataloaders nuevos;
- cambios en dataset, encoder o projection heads;
- checkpointing completo;
- scheduler complejo;
- logging avanzado;
- reconstrucción enmascarada.

La reconstrucción enmascarada sigue pendiente y continúa desactivada en la
configuración base.
