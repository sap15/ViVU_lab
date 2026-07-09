# Fase 9.6.C: `L_delta`

## Estado

La subfase 9.6.C queda implementada en:

- `src/gnn_siamese/losses/delta.py`
- `src/gnn_siamese/losses/__init__.py`
- `tests/test_delta_loss.py`

## Alcance implementado

`L_delta` se incorpora como pérdida auxiliar explícita para la ruta relacional
Mutante-WT. Esta subfase no redefine NT-Xent, no sustituye `L_relative_WT` y no
ensambla todavía `L_total`.

Modos disponibles:

- `none`
- `consistency`
- `variance`
- `covariance`
- `descriptor`

Comportamiento garantizado:

- `mode="none"` devuelve pérdida cero controlada e inactiva;
- `consistency` exige dos vistas relacionales de la misma muestra y penaliza su
  discrepancia;
- `variance` aplica una regularización de varianza mínima por dimensión con
  parámetro `gamma`;
- `covariance` penaliza covarianza fuera de la diagonal para reducir colapso y
  redundancia entre dimensiones;
- `descriptor` exige `target` y `target_name`, admite un predictor opcional y
  bloquea `custom_structure_energy` salvo `allow_energy_target=True`.

## Restricción metodológica sobre `z_delta`

`z_delta` no debe interpretarse como espacio relacional aprendido si
`MLP_delta` está desactivada o si la ruta delta no ha sido auditada. Esta
subfase añade la pérdida explícita y tests de gradiente/cambio de pesos para la
ruta delta, pero no sustituye la auditoría de entrenamiento real ni el registro
en `run_manifest.json`.

## Limitaciones actuales

- no existe todavía ensamblado real de `L_total`;
- no existe todavía paquete `training/` ni loop completo gobernado por YAML;
- no hay auditoría de entrenamiento por run ni estado final de `MLP_delta` en
  manifiesto;
- `custom_structure_energy` sigue bloqueado por defecto como target en
  `descriptor`;
- reconstrucción enmascarada sigue pendiente y fuera de esta subfase.
