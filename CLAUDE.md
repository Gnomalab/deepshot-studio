# Deepshot Studio: instrucciones para Claude Code

## Qué es
Aplicación de escritorio para Windows, con aspecto de programa comercial (no de prototipo), que corre 100% en local. Nombre: **Deepshot Studio** (lema «Deepshot Studio — the art of depthing»; paquete `deepshot`). Crea imágenes a partir de la profundidad de una foto, inspiradas en una técnica de referencia (ver "Nombre y atribución"):
1. Importar una fotografía.
2. Estimar su mapa de profundidad con Marigold V1.0 (modelo de difusión, Apache 2.0), o importar un mapa hecho fuera.
3. Dar forma al mapa: curva de tonos global y áreas locales de enfoque/luz editables.
4. Relighting (fase posterior): iluminar la escena a partir del mapa de profundidad.
5. Mezclar el resultado con la foto original.
6. Exportar a resolución completa.

Uso inicial: solo personal. Más adelante quizá distribuirla, así que el código debe ser ordenado y empaquetable.

El autor es Juanjo Fernández (Gnomalab Estudio). Es artista audiovisual, no desarrollador de oficio: prioriza código claro, comentado y listo para usar.

## Nombre y atribución
- La app se llama **Deepshot Studio** (decisión de Juanjo, 2026-10-18; ver ROADMAP, decisión 10). Paquete Python y comando: `deepshot`.
- La técnica de referencia es la de Karim Joseph Nassar. Su nombre («Vathography») NO se usa como marca, nombre de producto ni texto de la interfaz. La ÚNICA mención visible está en «Acerca de» (`deepshot/ui/about.py`: pestañas General y Créditos; un test lo vigila). Créditos lleva la cita de Zenodo (DOI 10.5281/zenodo.23103263, CC BY 4.0), los autores y artículos de Marigold y las bibliotecas. Texto de General: «Inspirado en la técnica Vathography de Karim Joseph Nassar (vathography.com). Proyecto independiente, sin afiliación con el autor.»
- En el código solo queda `LEGACY_APP_NAME` (`deepshot/__init__.py`), para migrar datos y ajustes del nombre provisional anterior.

## Cómo responder y trabajar con Juanjo
- Idioma: español, con terminología técnica sin simplificar.
- Respuestas cortas, pero explica brevemente qué hace cada paso.
- En el código, comenta las partes clave. Entrega código completo y listo para ejecutar.
- Si una solución depende de las condiciones del equipo o del lugar (GPU, VRAM, rutas, sistema), pregunta antes de suponer.
- NO relaciones este proyecto con láser, DMX ni espejos: son temas aparte. Solo entran si Juanjo lo pide.

## Restricciones
- **Todo en local.** Sin telemetría, sin llamadas a APIs externas en tiempo de uso. La única descarga permitida es la de los pesos del modelo (Hugging Face) la primera vez, y debe quedar en caché local.
- **Licencias compatibles con uso comercial** (decisión de Juanjo, 2026-10-18). Aceptadas: permisivas (MIT, BSD, Apache 2.0, HPND, ISC, PSF, CNRI-Python…) y copyleft débil sin modificar sus archivos (MPL-2.0; LGPL solo enlazada dinámicamente, p. ej. PySide6). Excluidas: GPL/AGPL en runtime (obligarían a liberar toda la app) y licencias no comerciales (CC-BY-NC…). Modelos con licencias de uso responsable (OpenRAIL): solo con aviso y confirmación. **Modelos de prueba (decidido por Juanjo, 2026-10-20):** MiDaS, ZoeDepth, Depth Pro, Depth Anything Base y Large se ofrecen solo para comparar calidad internamente (`ModelEntry.dev_only`; interruptor «Mostrar modelos de prueba» en Preferencias; `models.DEV_MODELS_DEFAULT` a False en una versión distribuida). Los de licencia no comercial llevan aviso y confirmación y nunca se descargan solos; no forman parte de la oferta final. **Excepción decidida por Juanjo (2026-10-20):** Depth Anything V2 Base y Large (CC-BY-NC 4.0, no comercial) se ofrecen en el gestor solo para probar y comparar calidad, con aviso, confirmación y sin descarga automática; hay que revisarlo antes de vender o distribuir la app. Antes de añadir una dependencia, comprueba y anota su licencia en `THIRD_PARTY_LICENSES.md`.
- **No usar PyQt** (GPL/comercial). Usar PySide6.
- **No copiar código del add-on de Blender de la técnica de referencia** (es un producto aparte). Reimplementar el concepto.
- **Modelos de profundidad:** solo implementaciones oficiales; leer el README/docs oficiales antes de escribir la llamada al modelo, no inventar la API. Modelo actual: Marigold V1.0 con `MarigoldDepthPipeline` oficial de diffusers. Marigold V2 (huawei-bayerlab/marigold-v2) se descartó en local por VRAM (ver ROADMAP.md, decisión 7). Solo pesos `.safetensors`. Si hay ComfyUI, solo su plantilla oficial: el autor advierte de nodos de terceros con código malicioso.
- **Windows.** Rutas con espacios: usar `pathlib`.
- **GPU: PENDIENTE** de confirmar (modelo y VRAM). Hasta entonces, el backend de inferencia debe ser configurable (CUDA o CPU).

## Referencias (carpetas del proyecto)
- `Capure references/`: 4 capturas de la web y del add-on de Blender de la técnica de referencia. Úsalas para el diseño visual y el flujo, no para copiar nada.
- `Link References/`: enlaces guardados. Marigold V2 en ComfyUI (docs.comfy.org/tutorials/utility/marigold-v2) y la página del add-on de Blender (vathography.gumroad.com/l/addon).

Lo que muestran las capturas, y lo que implica para la app:
1. **Flujo en 4 pasos** (idea → fotografía → enfoque y expresión → resultado). Usarlo como estructura de la interfaz: importar, dar forma, resultado.
2. **Mapa de profundidad como lienzo**: la app muestra el mapa en gris (cerca claro, lejos oscuro, o al revés según la convención). El resultado es una imagen en blanco y negro.
3. **Controles del add-on** (referencia de funcionalidad):
   - Carga del mapa de profundidad y recarga.
   - Relieve: escala de profundidad (p. ej. 4.00).
   - Rampa artística: paradas con posición (0.734 en la captura), interpolación Linear/RGB, cargar presets, y "Fit ramp to relief" (ajustar la rampa al rango real de profundidad).
   - Bandas (Focus & Shape Band): se aplican sobre la rampa, sobre negro o sobre blanco.
   - Áreas (Focus & Shape Area): forma elipsoide o caja, escala X/Y/Z, rotación X/Y/Z, fuerza (valores negativos incluidos), suavidad del borde (edge softness) y caída (falloff). Cada área también se aplica sobre rampa, negro o blanco.
   - Exportación: PNG 8 bits, PNG 16 bits, EXR 32 bits y carpeta de salida.
4. **Estilos de la expresión** de las capturas: haze atmosférico (gris, niebla), contraste fuerte (negro y blanco) y color (mapa de profundidad coloreado en rojo y azul). El modo color necesita una rampa multicolor.

Decisión para la app (cerrada): áreas como **elipsoides 3D completos**, igual que el add-on. Ver "Decisiones tomadas" en ROADMAP.md.

## Git y GitHub
- El código vive en un repositorio **privado** de GitHub (cuenta Gnomalab, `Gnomalab/deepshot-studio`) para seguirlo y probarlo en el PC y en el Mac M1. Se trabaja en la carpeta local; GitHub es copia y punto de encuentro. Versión actual: 0.1.0, fase alfa; la etiqueta `v0.1.0-alpha` se pone cuando Juanjo lo pida (hito: probado en Mac).
- **Avisar a Juanjo** cuando haya cambios importantes (función terminada con las pruebas en verde, o final de una tanda larga) de que toca commit y push, proponiendo el mensaje. **Nunca hacer commit ni push sin que él lo diga.**
- Normalmente trabaja en el PC. En el Mac se hace `git pull` solo cuando él lo pida («actualiza todo en el Mac»). Al cambiar de máquina: push en la que deja y pull en la que empieza; se edita en una sola a la vez.
- **Pendiente al llegar al PC (nota del Mac, 2026-10-31):** tras el `git pull`, recordar a Juanjo que reinstale el entorno (`uv pip install --python .venv -e ".[dev]"`) para que entre `accelerate` (nueva dependencia). Borrar esta nota cuando esté hecho.
- **No van a Git:** `models/` (se bajan desde el gestor en cada máquina; en el Mac empezar por Marigold V1.0 y Depth Anything V2 Small), `Fotos test/`, `Capure references/`, `.venv/` y cachés.
- En el disco externo `E:` Git da «dubious ownership»: usar `git -c safe.directory=E:/deepshot-studio ...` (o las variables `GIT_CONFIG_COUNT`, `GIT_CONFIG_KEY_0=safe.directory`, `GIT_CONFIG_VALUE_0=E:/deepshot-studio`) sin tocar la configuración global.
- Los commits terminan con la línea de atribución que indique el entorno. Los saltos de línea los normaliza `.gitattributes`.

## Equipos objetivo
- **PC (principal):** ASUS TUF, Windows, 16 GB de RAM. GPU NVIDIA GeForce RTX 4060 (portátil). VRAM: 8 GB según la configuración habitual de esta GPU, PENDIENTE de confirmar en Windows (Administrador de tareas > Rendimiento > GPU). Es la máquina de referencia para CUDA.
- **Mac:** M1 (Apple Silicon), 16 GB de memoria unificada (CPU y GPU comparten la misma memoria). Inferencia con PyTorch MPS.
- La app debe funcionar en ambos. Ninguna ruta, fuente ni dependencia puede ser específica de un SO sin una alternativa.
  - Rutas con `pathlib`. Fuentes y tema sin rutas absolutas.
  - Backend de inferencia: CUDA en Windows si hay GPU NVIDIA, MPS en Mac, CPU como respaldo.
  - En MPS hace falta `PYTORCH_ENABLE_MPS_FALLBACK=1` para las operaciones sin soporte. Comprobar cada modelo.
  - Empaquetado: PyInstaller no hace compilación cruzada. El .exe se genera en Windows y el .app en macOS.

## Stack
- Python 3.11+
- PySide6 (interfaz, LGPL-3.0)
- PyTorch con CUDA si hay GPU (BSD-3)
- numpy (BSD-3), Pillow (HPND)
- Opcional: OpenGL para la vista previa en tiempo real si el procesado en CPU no llega
- Tests: pytest (MIT)

## Diseño de la interfaz
- Aspecto de software profesional: tema oscuro por defecto, paleta sobria, tipografía limpia, iconos vectoriales coherentes, estados de carga visibles, atajos de teclado y deshacer/rehacer.
- Diseño en paneles acoplables o al menos con zonas fijas: biblioteca de fotos, lienzo central, panel de ajustes a la derecha.
- Nada de ventanas de depuración ni texto técnico visible al usuario salvo en un panel de registro.
- **Idiomas:** inglés (por defecto, desde 2026-10-20) y español (opcional), elegibles en Preferencias; se aplica al reiniciar. El inglés es la prioridad: si cambian textos, se traduce el inglés en el mismo cambio. La guía rápida se actualiza al final, no en cada cambio. Los textos del código están en español. Todo texto nuevo visible al usuario se marca como traducible: `self.tr()` en widgets, `QCoreApplication.translate("Contexto", ...)` fuera de ellos, `QT_TRANSLATE_NOOP("Contexto", ...)` en constantes de módulo (y se traduce al mostrarlo con el mismo contexto). **Siempre la llamada completa** (no un alias: lupdate no lo entiende) y con `.format(...)` en vez de f-strings. Después: `python tools/update_translations.py --extract`, traducir lo «unfinished» de `resources/translations/deepshot_en.ts` y `python tools/update_translations.py --compile`; `tests/test_i18n.py` falla si falta algo. Los comentarios del código siguen en español.
- **Estética de referencia: Blender** (preferencia de Juanjo). Inspiración visual, sin copiar recursos de Blender (iconos, temas o código).
  - Paleta de grises neutros (fondos ~#1d1d1d / #303030, paneles #3d3d3d, botones #545454) con acento azul (#4772b3) para selección, estados activos y rellenos de deslizadores. Texto #e5e5e5.
  - Botones al estilo Blender: compactos, esquinas poco redondeadas, grupos segmentados (p. ej. "sobre Rampa | Negro | Blanco"), campos numéricos que se arrastran con la barra de valor rellena en azul (como "Edge softness"), y secciones plegables con cabecera y flecha.

## Estructura objetivo
```
deepshot-studio/
├── CLAUDE.md
├── ROADMAP.md
├── THIRD_PARTY_LICENSES.md
├── pyproject.toml
├── deepshot/
│   ├── __init__.py
│   ├── depth.py        # Marigold V1.0: foto -> profundidad (PNG 16 bits, caché)
│   ├── models.py       # gestor de modelos: catálogo, descarga, instalar, borrar
│   ├── refine.py       # refinado del mapa guiado por la foto (guided filter)
│   ├── tonemap.py      # color_ramp, modos banda / degradado / multibanda
│   ├── areas.py        # áreas locales (elipse, borde suave, ganancia)
│   ├── area_edit.py    # de movimientos del ratón a valores del área (sin interfaz)
│   ├── paint.py        # máscara pintada con pincel (sin interfaz)
│   ├── presets.py      # presets de rampa y de estilo, de fábrica y del usuario
│   ├── look.py         # paso Look: pila de efectos al estilo Photoshop (niveles, contraste, saturación, grano, nitidez, viñeta + módulos), sin interfaz
│   ├── modules.py      # módulos descargables con modelo o librerías (module.json + module.py), en <modelos>/looks: catálogo local, instalar, aprobar, cargar (sin interfaz)
│   ├── export.py       # PNG 8/16, EXR 32; efecto o mapa de profundidad
│   ├── errors.py       # errores inesperados: registro y aviso
│   ├── dof.py          # desenfoque de lente por profundidad (apertura, diafragma, bokeh; sin interfaz)
│   ├── depth_control.py # control de profundidad: mueve las zonas cerca / medio / lejos del mapa (sin interfaz)
│   ├── timing.py       # tiempo transcurrido y estimado de las tareas largas (sin interfaz)
│   ├── memory.py       # medir la memoria del proceso y devolverla (sin interfaz)
│   ├── i18n.py         # idioma de la interfaz: es / en, traducciones de Qt (resources/translations)
│   ├── themes.py       # temas de color y aspecto de la interfaz: paletas, plantilla QSS, temas del usuario (sin interfaz)
│   ├── depth_runs.py   # cálculos de profundidad de una foto con sus tiempos, para compararlos (sin interfaz)
│   ├── depth_view.py   # el mapa de profundidad en gris, en color o con sus zonas marcadas (sin interfaz)
│   ├── view3d.py       # vista 3D: cámara orbital y dibujo de la nube de puntos (numpy, sin interfaz)
│   ├── area3d.py       # áreas en 3D: contornos, tiradores y gestos de ratón a valores (sin interfaz)
│   ├── project_file.py # archivo de proyecto .deepshot (ZIP: ajustes, mapas, máscaras; fotos por ruta)
│   ├── relight.py      # (fase 6) normales desde profundidad + sombreado
│   ├── compose.py      # mezcla con la foto original, modos de fusión
│   ├── project.py      # ajustes en JSON (reproducibilidad y presets)
│   ├── ui/             # widgets, tema QSS, ventanas
│   └── app.py          # punto de entrada
├── resources/          # tema, iconos
├── tests/
└── samples/            # fotos de prueba
```

## Estado actual
- **Fase 0 completada (2026-10-18).** Entorno `.venv` con Python 3.12 (uv), PyTorch 2.11 + CUDA 12.8 (RTX 4060 detectada, 8 GB), PySide6-Essentials 6.12. Ventana base con tema tipo Blender, barra de pasos, biblioteca, lienzo con zoom, panel de ajustes por pasos (marcadores) y registro oculto. `pytest` en verde.
- **Fase 1 completada (2026-10-18).** `depth.py`: Marigold V1.0 (diffusers), descarga única con progreso a `<datos>/models`, caché de mapas en PNG 16 bits (`<datos>/depth_cache`, clave = contenido de la foto + ajustes), registro `MODELS` para el futuro gestor de modelos. En la RTX 4060: 11 s de carga, ~1,7 s por foto, 3,6 GB de VRAM. UI: calcular en segundo plano con progreso, conmutador Foto/Profundidad (V), importar mapa externo PNG/TIFF. Carpeta de datos: `%LOCALAPPDATA%\Gnomalab Estudio\Deepshot Studio` (o `DEEPSHOT_DATA_DIR`).
- **Gestor de modelos hecho (2026-10-18):** `models.py` (catálogo, añadir repo de Hugging Face, instalar desde carpeta solo con archivos seguros, borrar, tamaño) y diálogo `ui/model_manager.py`. Modelo activo en QSettings (`depth/active_model`).
- **Calidad del mapa hecha (2026-10-18):** presets Rápida/Alta/Máxima y refinado guiado (`refine.py`). Visor comparativo A/B con divisor (`ui/canvas.py`).
- **Cabecera, pie, ayuda y atajos hechos (2026-10-18).** Web del autor: https://www.gnomalab.es.
- **Fase 2 hecha (2026-10-18):** `tonemap.py`, `areas.py`, `compose.py`, `project.py` (ver ROADMAP). Convención: rampa posición 0 = cerca. Escena 3D: x ∈ ±aspecto, y ∈ ±1 (arriba +), z = profundidad × relieve.
- **Fase 3, núcleo hecho (2026-10-18):** edición en vivo con `ui/editor.py` (sesión: proyecto por foto, deshacer), `ui/render_controller.py` (vista previa en 2 hilos; llamar a `shutdown()` al cerrar), `ui/editor_panels.py`, `ui/ramp_editor.py`, `ui/value_slider.py`. Los hilos que emiten señales deben protegerse con try/except RuntimeError (la ventana puede haberse cerrado). En tests, borrar los widgets sueltos con deleteLater (ver `tests/test_editor.py`).
- **Mejoras tras probar la app hechas (2026-10-19):** edición de áreas sobre la imagen (`area_edit.py`, `ui/canvas.py`: arrastrar, tiradores, G/S/R, rueda con modificadores, dibujar y colocar), colores y nombres de área, rampa con color en vivo, presets de rampa y de estilo (`presets.py`, datos en `<datos>/presets/`), máscara pintada con pincel (`paint.py`, `<datos>/masks/`), exportación PNG 8/16 y EXR 32 de efecto o mapa (`export.py`, `ui/export_panel.py`), iconos propios (`ui/icons.py`), biblioteca con quitar/doble clic, registro con herramientas y archivo (`<datos>/logs/`), ayuda al pasar el ratón en el pie, errores inesperados visibles (`errors.py`). Vista «Resultado» renombrada **Efecto**; Relieve está en el paso 3. Detalle en ROADMAP.md.
- Reglas que salieron de ahí: todo control nuevo lleva `setToolTip` (hay un test que lo exige); los diálogos modales en tests se parchean (`QMessageBox.exec`) o el test se cuelga; los hilos que importan torch hay que esperarlos antes de cerrar la ventana en tests (`window._probe.wait()`).
- **Vista 3D hecha (2026-10-19):** modo «3D» de la vista Efecto, con conmutador 2D | 3D (`view3d.py`, `area3d.py`, `ui/view3d_widget.py`): relieve del mapa como superficie que se orbita con el ratón, con textura foto / profundidad / efecto, y las áreas dibujadas con tiradores (flechas, cuadrados, anillos) que se editan con un paso de deshacer por gesto (`EditSession.gesture3d`). Calidad 3D elegible (baja…máxima y automática, `view3d.QUALITY_POINTS`). Pendiente en 3D: G / S / R con ejes, dibujar y colocar sobre la superficie (ver ROADMAP).
- **Desenfoque de lente (DOF) hecho (2026-10-19):** `dof.py` + panel «Desenfoque de lente» en el paso 3 (`LensPanel`). Foco y zona nítida en profundidad, apertura en f/, diafragma redondo o de 3-12 palas con giro, altas luces y borde del disco, desenfoque delante y detrás del foco por separado; sobre la imagen final o solo sobre la foto. Va en el proyecto (`Project.lens`, JSON) y en los estilos. Capas de radio creciente por FFT, sin halos del primer plano nítido y con el primer plano desenfocado translúcido sobre lo que tiene detrás. ~0,1-0,25 s a 1000 px y ~14 s a 25 Mpx. Cuentagotas del foco sobre la imagen (`EditSession.pick_focus_gesture`, herramienta «pick_focus» del lienzo) y capa verde «Ver zona en foco» (`dof.focus_map`).
- **Control de profundidad hecho (2026-10-19):** `depth_control.py` + panel «Control de profundidad» (`DepthControlPanel`, paso 3, bajo Relieve). Tres zonas (cerca / medio / lejos) con centro, destino, suavizado y límites; `Project.render` mueve la profundidad antes de todo, así que rampa, bandas, desenfoque, áreas y vista 3D la leen ya movida. En el editor, la profundidad con el control aplicado es `EditSession.warped(depth)` (con caché): todo lo nuevo que lea el mapa para algo del efecto debe usarla (o `project.depth_control.apply`). No va en los estilos. Reimplementa una función Pro del add-on de referencia (ver ROADMAP).
- **Pendiente anotado por Juanjo:** el modo 3D debe ser un conmutador 2D / 3D dentro de la pestaña Efecto y no una vista aparte del conmutador (ROADMAP, «Por revisar»).
- **Apariencia hecha (2026-10-19):** temas de color (oscuro, claro, medianoche y los del usuario en `<datos>/themes/`), densidad y redondeo de los controles, plantillas de disposición de los paneles y ventana de Preferencias (Ctrl+,); pantalla de inicio con proyectos recientes. Regla nueva: **nada de colores escritos a mano en widgets**: usa `deepshot.ui.theme.color("nombre")` al pintar, propiedades del QSS (`dim`, `tone`) en etiquetas y `icons.set_icon(boton, "nombre")` para iconos (se repintan al cambiar de tema).
- **Carpeta de modelos (2026-10-20):** los modelos viven en `models/` junto al programa (`models.resolve_models_dir`; cambiable en el gestor de modelos o con `DEEPSHOT_MODELS_DIR`; si no se puede escribir ahí, en la carpeta de datos). Un modelo soltado a mano en una carpeta propia (config.json o model_index.json + .safetensors) aparece al «Actualizar lista» (origen «folder»). Las pruebas usan una carpeta de datos temporal (`tests/conftest.py`).
- Siguiente: fase 4 (flujo creativo), afinar el refinado con fotos reales (`samples/`), empaquetado y English.
- **Renombrado a Deepshot Studio (2026-10-18).** La carpeta de datos y los ajustes (QSettings) del nombre anterior se migran solos al primer arranque.
- **Carpeta del proyecto renombrada (2026-10-19):** `C:\Users\gnoma\Documents\-CODE PROJECTS\deepshot-studio` (antes «VATOGRAPHY APP»). Entorno `.venv` rehecho allí; 59 tests en verde; CUDA y modelos OK.
- Arrancar: `.venv\Scripts\python -m deepshot`. Tests: `.venv\Scripts\python -m pytest` (238 en verde, ~2 min).
- Reinstalar el entorno: `uv venv --python 3.12 .venv` y luego `uv pip install --python .venv -e ".[dev]"` (en Windows, torch sale del índice CUDA definido en `pyproject.toml`).

Ver ROADMAP.md para las fases y el orden de trabajo.
