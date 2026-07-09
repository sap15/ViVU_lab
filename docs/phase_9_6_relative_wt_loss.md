# Fase 9.6.B: `L_relative_WT`

## Estado

La subfase 9.6.B queda implementada en:

- `src/gnn_siamese/losses/relative_wt.py`
- `src/gnn_siamese/losses/__init__.py`
- `tests/test_relative_wt_loss.py`

## Alcance implementado

`L_relative_WT` se incorpora como pérdida auxiliar separada. WT sigue siendo una
referencia relacional y no un positivo fuerte por defecto.

Modos disponibles:

- `none`
- `margin`
- `ranking`
- `predictive`

Comportamiento garantizado:

- `mode="none"` devuelve pérdida cero controlada e inactiva;
- `margin` regulariza la distancia Mutante-WT sin colapsar ambas ramas;
- `ranking` compara distancias Mutante-WT entre muestras y exige target;
- `predictive` predice un target auxiliar explícito desde una señal relacional
  simple y bloquea `custom_structure_energy` por defecto.

## Limitaciones actuales

- no existe todavía ensamblado real de `L_total`;
- no existe todavía paquete `training/` ni loop completo gobernado por YAML;
- `L_delta` sigue pendiente y `MLP_delta` no se entrena desde esta subfase;
- reconstrucción enmascarada sigue fuera del baseline operativo;
- `strong_positive_control` no se implementa en esta subfase.

## Restricción metodológica mantenida

Esta implementación no redefine NT-Xent ni convierte Mutante-WT en un par
positivo contrastivo por defecto. Cualquier uso de WT como positivo fuerte
permanece fuera del baseline y requerirá una ablación explícita posterior.
