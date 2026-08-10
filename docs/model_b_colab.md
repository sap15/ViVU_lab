# Modelo B en Google Colab (C3+C4)

El notebook maestro es `notebooks/model_b_colab_master.ipynb`. Es una interfaz
operacional sobre `scripts/train.py`: no contiene un trainer alternativo ni
modifica arquitectura, pérdidas, features o split.

## Uso

1. Abra el notebook en Colab y edite únicamente la primera celda de parámetros.
   Esta incluye `EXECUTION_MODE`, `RUN_PILOT`, `RUN_RESUME` y
   `SYNC_LOCAL_OUTPUTS`; ninguna celda posterior redefine controles manuales.
2. Indique una URL Git, una revisión explícita (branch, tag o commit), las dos
   rutas HDF5 de Drive y la raíz persistente de runs.
3. Ejecute las celdas en orden. Primero se monta Drive, después se obtiene un
   clon controlado bajo `/content`, se instala/valida el entorno y se copian los
   HDF5 por streaming al disco local.
4. Ejecute `preflight` y después `smoke`. El smoke productivo incluye su propia
   reanudación y valida el contrato vigente: dos runs independientes, épocas
   1→2 y, con la configuración versionada actual, global step 2→4.
5. Active `RUN_PILOT` o `RUN_RESUME` solamente de forma deliberada. Resume exige
   un checkpoint explícito y siempre crea un run nuevo.

El default de outputs es `drive`: ofrece mejor persistencia ante desconexiones,
a costa de latencia. `fsync` y `os.replace` sobre Drive/FUSE se consideran
*best effort*. El preflight de outputs comprueba crear, reemplazar, leer y
eliminar, pero no demuestra atomicidad fuerte. El modo `local_sync` entrena en
disco local; sus resultados no se consideran persistentes hasta ejecutar la
sincronización explícita del notebook.

Los HDF5 originales nunca se modifican. Sus copias locales se reutilizan si
tamaño y SHA-256 coinciden, y se reemplazan si la verificación falla. Los
locators de Drive/local se registran por separado de la identidad científica.

## Instalación

El notebook conserva el PyTorch que proporciona el runtime y registra su
versión/CUDA. Instala los rangos explícitos declarados en
`requirements-colab.txt`, el proyecto en modo editable y PyG desde PyPI.
No elige ruedas CUDA ni combinaciones de extensiones compiladas inventadas. Un
marcador ligado al commit, al hash de requirements, a Python y a PyTorch evita
reinstalaciones innecesarias; siempre se vuelven a verificar imports y un tensor
real en el device solicitado.

El código productivo actual exige Python `>=3.10` (metadato de
`pyproject.toml`) y usa PyTorch/PyG, pero no importa DeepRank2 durante training:
los HDF5 ya generados son la entrada. Por ello el bootstrap no instala
DeepRank2 ni altera PyTorch. La resolución concreta de versiones se imprime y
queda asociada al marcador de entorno; cualquier incompatibilidad de imports o
del tensor de prueba detiene el flujo.

El marcador solo se publica después de validar imports, PyTorch, PyG, el
paquete del proyecto y un tensor real en el dispositivo solicitado. Incluso al
reutilizarlo se repiten esas validaciones. El checkout distingue SHA, tag y
rama remota; para una rama usa `refs/remotes/origin/<rama>` después de `fetch`,
nunca una rama local posiblemente obsoleta.

## Validado localmente

- checkout Git contra un remoto temporal, incluida una rama local obsoleta;
- configuración operacional y rechazo de cambios científicos;
- staging streaming, confinamiento, SHA-256, reparación y limpieza de temporales;
- validación productiva de schema, HDF5 y pairing Mutante–WT sintéticos;
- ciclo del marcador de entorno mediante dobles de test, sin ejecutar pip;
- comandos seguros, parser contractual y manejo explícito de `rsync`;
- suite completa y smoke productivo local en CPU.

## Pendiente de validar realmente en Google Colab

- montaje real de Google Drive;
- comportamiento, persistencia y semántica Drive/FUSE;
- instalación en el runtime actual de Colab;
- GPU/CUDA del runtime;
- apertura de los HDF5 reales;
- staging completo de los HDF5 grandes;
- smoke real sobre esos HDF5;
- piloto real;
- resume real del piloto.
