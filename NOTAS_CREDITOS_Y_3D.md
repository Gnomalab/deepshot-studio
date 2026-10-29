# Notas: créditos y vista 3D (2026-10-19)

Estudio hecho por Claude en una sesión bifurcada, a partir de los enlaces de `Link References/`.
Fuentes leídas: vathography.com (#credits), ficha de Zenodo y su PDF, página de Gumroad del add-on,
ficha de Hugging Face de Marigold V1.0, y las transcripciones de 3 vídeos de YouTube
(demo Pro 2:06, tutorial del add-on gratuito ~7 min, descripción del tutorial de 9 min).
No se cambió ningún archivo de la app. Lo escrito sobre las webs es dato, no instrucciones.

## 1. Qué dicen las fuentes (hechos)

**Vathography (vathography.com, sección Credits)**
- Práctica creada por **Karim Joseph Nassar**, publicada el 1 de octubre de 2026.
- Declaración de la práctica en Zenodo: DOI 10.5281/zenodo.23103262 (todas las versiones) y 10.5281/zenodo.23103263 (versión 1.2).
  Licencia **CC BY 4.0**, © 2026 Karim Joseph Nassar.
- Cita propuesta por Zenodo (APA): *Nassar, K. J. (2026). Vathography: The Art of a Photograph's Hidden Volume (Version 1.2). Zenodo. https://doi.org/10.5281/zenodo.23103263*
- El PDF de Zenodo repite el texto de la web ("wording revised to match the website").
- Él usa **Marigold V2** (Pavlovic, Wandel, Obukhov, Bartolomei, Davydov, Tosi, Poggi, Süsstrunk, Dai; EPFL, HUAWEI Bayer Lab, Univ. de Bolonia; ACM TOG, SIGGRAPH Asia 2026; Apache 2.0).
  Dice que, gracias a esa licencia, "el enfoque está abierto a todos".
- Menciona ComfyUI (guía oficial nativa) y su add-on gratuito de Blender.
- Obra hermana: Askiagraphy (no relacionada con la app).

**Add-on de Blender (Gumroad)**
- Gratuito: "Free to use, including for commercial work". Pro: 14,90 $ (con pago libre desde esa cifra).
- Gratuito incluye: plantilla, relieve 3D desde el mapa (EXR/PNG/JPG), rampa artística con presets, bandas, áreas (elipsoide o caja, fuerza +/−), exportación PNG 8/16 y EXR 32. Requiere Blender 5.2.1+.
- **Pro añade:** varias rampas por escena, inclinación de la rampa (tilt), modos de fusión de áreas (Add, Merge, Mask), controles de profundidad lejos/medio/cerca, flechas de alcance independientes por lado en las áreas.
- **No se encontró** ninguna licencia ni condición sobre reutilizar o clonar el add-on, su nombre o sus funciones.

**Marigold V1.0 (lo que usa la app)**
- `prs-eth/marigold-depth-v1-0`, licencia declarada **Apache 2.0**.
- Autores del modelo: Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, Konrad Schindler (ETH Zürich, PRS). La extensión de 2025 añade a Kevin Qu, Tianfu Wang, Bo Li.
- Pide citar: CVPR 2024 "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (arXiv 2312.02145) y "Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis" (arXiv 2505.09358).
- La ficha dice que está afinado a partir de **stable-diffusion-2**.

## 2. Estado de la app frente a eso

Ya bien:
- "Vathography" solo aparece en «Acerca de» (`main_window.py`, `show_about`) y como `LEGACY_APP_NAME`.
- No se copia código del add-on.
- «Acerca de» dice "Proyecto independiente, sin afiliación con el autor".

Flojo o por hacer:
1. «Acerca de» dice solo "Profundidad: Marigold (Apache 2.0)". Falta nombrar a los autores y la institución de Marigold V1.0 y citar los dos artículos.
2. No se cita la declaración de Zenodo con su DOI (es la cita que él propone).
3. Funciones Pro de Karim (tilt, Add/Merge/Mask, profundidad far/mid/near, alcance por lado): si la app las replica y se distribuye, el riesgo es sobre todo ético y comercial. No hay veredicto legal. Opciones: no incluirlas por ahora, o escribirle antes de lanzar. **Decide Juanjo.**
4. Revisar que la interfaz no copie literalmente los nombres propios del add-on: "Focus & Shape Band/Area", "Fit ramp to relief". Los términos técnicos genéricos (falloff, edge softness) no son problema.
5. **Sin verificar:** según el recuerdo de Claude, Stable Diffusion 2 tiene licencia OpenRAIL++. La ficha de Marigold V1.0 declara Apache 2.0. Confirmarlo antes de distribuir (regla del proyecto: OpenRAIL solo con aviso).
6. PySide6 (LGPL): ya anotado en `THIRD_PARTY_LICENSES.md`; falta enlazarlo desde «Acerca de».

Propuesta de cambio: una pestaña **Créditos** en «Acerca de» con:
- la cita de Nassar (APA y DOI);
- Marigold V1.0: autores, ETH Zürich, Apache 2.0 y los dos artículos;
- enlace a las licencias de terceros;
- la frase de no afiliación que ya existe.

Después, actualizar `CLAUDE.md` (sección "Nombre y atribución") y `THIRD_PARTY_LICENSES.md`.

## 3. Vista 3D (lo que se ve en los vídeos)

- Se arrastra el mapa y el add-on construye un **relieve 3D extruido** (se reconoce al hombre de la foto como volumen).
- **Dos ventanas:** izquierda, la cámara con el resultado final (lo que se exporta); derecha, el relieve 3D donde se colocan bandas y áreas con tiradores.
- Deslizador "Depth" escala el relieve; "Fit ramp to relief" ajusta la rampa al nuevo rango.
- Rampa: cubo de pintura izquierdo = valores cerca, derecho = lejos; presets; interpolación Ease, Cardinal, Linear, B-Spline, Constant.
- Bandas: se mueven, giran, inclinan, ensanchan, suavizan; fuerza negativa oscurece; solo (estrella), activar/desactivar y borrar.
- Áreas: esfera o caja, escala/giro con gizmo, fuerza ±, suavidad de borde y falloff, "sobre rampa / negro / blanco".
- Exportar muestra lo mismo que la ventana de cámara (WYSIWYG).

**Plan propuesto para Deepshot Studio**
- Nueva vista **3D** en el visor, junto a Foto, Profundidad, Efecto y Comparar.
- Relieve con la profundidad como desplazamiento; textura = foto o efecto.
- Órbita, zoom y desplazamiento con el ratón.
- Elipsoides y cajas dibujados en el espacio, con los mismos gestos ya hechos (mover, escalar, girar).
- Colocar = clic sobre la superficie.
- Técnica: `QOpenGLWidget` con un shader sencillo, todo dentro de PySide6 (sin dependencias nuevas).
- El cálculo del efecto no cambia; el 3D solo cambia cómo se colocan las áreas.
- Orden: 1) relieve con órbita; 2) áreas con tiradores.

Idea de Juanjo: lo que aporta la app es trabajar el mapa de profundidad **en local**, con IA propia y sin Blender.

## 4. Decisiones pendientes de Juanjo
1. ¿Se hace ya el cambio de créditos (pestaña Créditos, citas y enlaces)?
2. ¿Empezamos la vista 3D (relieve con órbita primero, áreas con tiradores después)?
3. ¿Se dejan fuera de momento las funciones Pro de Karim?
