# Licencias de terceros

Política del proyecto (actualizada el 2026-10-18): licencias compatibles con uso comercial.
Permisivas (MIT, BSD, Apache 2.0, HPND, ISC, PSF, CNRI-Python…) y copyleft débil sin modificar
sus archivos (MPL-2.0; LGPL enlazada dinámicamente). Nada de GPL/AGPL en runtime ni licencias
no comerciales. Modelos OpenRAIL: solo con aviso. Comprobar y anotar aquí
cada dependencia antes de añadirla.

Versiones y licencias leídas de los metadatos del entorno `.venv` (2026-10-18).

## Runtime (se distribuyen con la app)

| Paquete | Versión | Licencia | Notas |
|---|---|---|---|
| PySide6-Essentials | 6.12.0 | LGPL-3.0 (triple licencia: LGPL-3.0 / GPL-2.0 / GPL-3.0) | Usamos la opción **LGPL-3.0**. Enlace dinámico: las DLL de Qt deben quedar como archivos separados y sustituibles en el paquete final (PyInstaller en modo carpeta, no `--onefile`). Incluir el texto de la LGPL y un aviso de cómo reemplazar Qt. Se usa Essentials, sin Addons ni WebEngine. |
| shiboken6 | 6.12.0 | LGPL-3.0 (misma triple licencia) | Dependencia de PySide6. Mismas condiciones. |
| numpy | 2.5.4 | BSD-3-Clause (incluye partes 0BSD, MIT, Zlib y CC0-1.0) | Todas permisivas. |
| Pillow | 13.0.0 | MIT-CMU (HPND) | |
| torch (PyTorch) | 2.11.0+cu128 | BSD-3-Clause | En Windows, la rueda `cu128` incluye DLL de CUDA/cuDNN de NVIDIA, sujetas a los acuerdos de licencia de NVIDIA (componentes redistribuibles). Revisar antes de distribuir. En Mac, rueda de PyPI con MPS. |
| torchvision | 0.26.0 | BSD-3-Clause | Lo pide el procesador de imágenes de Depth Pro (modelo de prueba). En Windows, rueda `cu128` como torch. |
| filelock | 4.2.2 | MIT | Dependencia de torch. |
| fsspec | 2026.9.0 | BSD-3-Clause | Dependencia de torch. |
| Jinja2 | 3.1.6 | BSD-3-Clause | Dependencia de torch. |
| MarkupSafe | 3.0.4 | BSD-3-Clause | Dependencia de Jinja2. |
| networkx | 3.7 | BSD-3-Clause | Dependencia de torch. |
| sympy | 1.14.0 | BSD-3-Clause | Dependencia de torch. |
| mpmath | 1.3.0 | BSD-3-Clause | Dependencia de sympy. |
| typing_extensions | 4.16.0 | PSF-2.0 | Dependencia de torch. Licencia permisiva. Validada (2026-10-18). |
| setuptools | 81.0.0 | MIT | Dependencia de torch. |
| diffusers | 0.41.0 | Apache-2.0 | `MarigoldDepthPipeline` oficial (fase 1). |
| accelerate | 1.15.0 | Apache-2.0 | Carga de pesos directa al dispositivo en diffusers/transformers (menos RAM al cargar). Comprobada en Mac (2026-10-31). |
| psutil | 7.2.2 | BSD-3-Clause | Dependencia de accelerate. |
| transformers | 5.20.1 | Apache-2.0 | Codificador CLIP que carga el pipeline de Marigold. |
| huggingface_hub | 1.33.0 | Apache-2.0 | Descarga única de pesos. Telemetría desactivada (`HF_HUB_DISABLE_TELEMETRY=1`). |
| hf-xet | 1.7.0 | Apache-2.0 | Transferencia de archivos grandes de Hugging Face. |
| safetensors | 0.8.0 | Apache-2.0 | Formato de pesos seguro (no ejecuta código). |
| tokenizers | 0.23.3 | Apache-2.0 | |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python | CNRI-Python: permisiva, compatible con uso comercial. Validada (2026-10-18). |
| requests | 2.34.2 | Apache-2.0 | |
| urllib3 | 2.8.0 | MIT | |
| idna | 3.20 | BSD-3-Clause | |
| charset-normalizer | 3.5.2 | MIT | |
| certifi | 2026.10.18 | MPL-2.0 | Copyleft débil por archivo: solo obliga si se modifican sus archivos (no se hace). Validada (2026-10-18). |
| httpx | 0.28.1 | BSD-3-Clause | |
| httpcore | 1.0.9 | BSD-3-Clause | |
| h11 | 0.16.0 | MIT | |
| anyio | 4.15.1 | MIT | |
| importlib_metadata | 9.0.1 | Apache-2.0 | |
| zipp | 4.1.1 | MIT | |
| PyYAML | 6.0.3 | MIT | |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | Igual que certifi: MPL-2.0 sin modificar. Validada (2026-10-18). |
| click | 8.5.0 | BSD-3-Clause | CLI de huggingface_hub (no se usa). |
| typer | 0.27.3 | MIT | CLI de huggingface_hub (no se usa). |
| rich | 15.0.0 | MIT | |
| markdown-it-py | 4.2.0 | MIT | |
| mdurl | 0.1.2 | MIT | |
| scipy | 1.18.1 | BSD-3-Clause | Necesaria para el ensemble de Marigold (calidad Máxima). |
| shellingham | 1.5.4 | ISC | ISC: equivalente a MIT. Validada (2026-10-18). |
| openexr | 3.5.2 | BSD-3-Clause | Escribir EXR de 32 bits (`deepshot/export.py`). Licencia leída del archivo `LICENSE.md` de la rueda (2026-10-19). Solo depende de numpy. |

## Solo desarrollo (no se distribuyen)

| Paquete | Versión | Licencia |
|---|---|---|
| pytest | 9.1.1 | MIT |
| pluggy | 1.7.0 | MIT |
| iniconfig | 2.3.1 | MIT |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| Pygments | 2.21.0 | BSD-2-Clause |
| colorama | 0.4.6 | BSD-3-Clause |

## Modelos

| Modelo | Licencia | Notas |
|---|---|---|
| Marigold V1.0 (prs-eth/marigold-depth-v1-0) | Apache-2.0 (model card en Hugging Face) | Modelo de profundidad usado. Pesos fp16 en safetensors (~2,6 GB), descargados la primera vez a la carpeta de datos de la app. |
| Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf) | Apache-2.0 (model card en Hugging Face) | Modelo de profundidad rápido opcional. safetensors (~99 MB), descargado a la carpeta de datos. Base y Large son CC-BY-NC: **no se usan**. |
| SAM 2.1 Hiera Small (facebook/sam2.1-hiera-small) | Apache-2.0 (model card en Hugging Face) | Lo usa el módulo de Look «Quitar fondo» (`modules_catalog/remove_background`). `Sam2Model` oficial de transformers, solo `model.safetensors` (~184 MB; el `.pt` del repositorio no se descarga). Sin dependencias nuevas. Comprobado el 2026-10-31. |
| Swin2SR x2 (caidas/swin2SR-classical-sr-x2-64) | Apache-2.0 (model card en Hugging Face) | Lo usa el módulo de Look «Superresolución» (`modules_catalog/superres`). `Swin2SRForImageSuperResolution` oficial de transformers, solo `model.safetensors` (~48 MB). Sin dependencias nuevas. Comprobado el 2026-10-31. |
| BiRefNet / RMBG-1.4 y 2.0 | MIT (BiRefNet) / no comercial (RMBG) | **No se usan**: RMBG es no comercial y BiRefNet solo se carga con `trust_remote_code` (código remoto), prohibido por la política. Mask2Former (licencia «other» en la ficha) y DETR / UperNet (solo `.bin`) tampoco. |
| Apple SHARP / Depth Pro | Licencia de modelo solo para investigación | **No se usan**: prohíben el uso comercial. |
| Marigold V1.1 (prs-eth/marigold-depth-v1-1) | OpenRAIL++ | **No se usa**: licencia con restricciones de uso, fuera de la política del proyecto. |
| Marigold V2 (huawei-bayerlab/marigold-v2-0) | Apache-2.0 (base Qwen-Image-Edit-2509: Apache-2.0) | **No se usa** por requisitos de hardware (~17 GB de VRAM, solo Linux). Ver ROADMAP.md. |

## Recursos propios

- `resources/icons/app.svg`: icono provisional, creado para este proyecto.
- Tipografía: fuentes del sistema (Segoe UI en Windows, la del sistema en macOS). No se distribuye ninguna fuente.
