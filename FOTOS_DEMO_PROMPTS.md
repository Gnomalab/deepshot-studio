# Fotos de prueba para la demo: prompts para generarlas con IA

Lista de fotos que cubren los casos que necesitamos probar (calidad del mapa de profundidad, casos difíciles, áreas y
bandas, y la propia app). Cada prompt está en inglés porque los generadores de imagen lo entienden mejor.

**Sufijo común** (añádelo al final de cada prompt): `photorealistic photograph, natural light, sharp focus, high detail,
no text, no watermark, no illustration`

**Consejos al generar**
- Pide la resolución más alta que dé el generador y, si puedes, reescala después. Para pruebas de rendimiento hacen falta
  fotos de 24 Mpx: eso se consigue reescalando una buena, no generándola.
- Genera varias variantes y quédate con la que tenga la profundidad más clara.
- Personas: que sean generadas por IA (no retratos de gente real), por si la demo se publica.
- Guarda por subcarpetas: `interior`, `paisaje`, `retrato`, `detalle`, `dificil`, `formatos`, `areas`.

---

## 1. Calidad del mapa de profundidad

### 1.1 Interior con perspectiva marcada (`interior/`)
`A long narrow hallway in an old building, strong one-point perspective, floor tiles leading to a door far away,
doors on both sides, warm light from the end`

### 1.2 Calle con personajes y fondo (`interior/`)
`A busy European street seen from the middle of the road, people walking at three clearly different distances (one very
close in the foreground, a few in the middle, others far away), parked cars, buildings receding into the distance, late
afternoon light`

### 1.3 Paisaje con capas (`paisaje/`)
`Mountain landscape with clear depth layers: wildflowers in the foreground, a lake in the middle ground, forested hills,
and far snowy mountains under a sky with clouds`

### 1.4 Retrato con fondo desenfocado (`retrato/`)
`Close portrait of a woman with long wavy hair, shallow depth of field, creamy blurred city background with bokeh,
hair strands clearly visible against the background`

### 1.5 Retrato a cuerpo entero, sujeto a distancia (`retrato/`)
`Full-body photo of a man standing alone about 15 metres away in a large empty plaza, small in the frame, long shadows,
wide angle`

### 1.6 Bodegón o macro (`detalle/`)
`Macro still life on a wooden table: a bowl of fruit, a glass of water and a few coins very close to the lens, strong
texture, the table extending away to a blurred wall`

### 1.7 Objetos finos (`detalle/`)
`Tangled bare tree branches and a wire fence against a bright overcast sky, thin wires, cables, leaves and twigs
crossing in front of a distant building`

---

## 2. Casos difíciles (`dificil/`)

### 2.1 Cristales y reflejos
`A shop window at night with reflections of the street over the displayed mannequins, glass with reflections and
transparency, a person standing behind the glass`

### 2.2 Espejo y agua
`A large mirror in a living room reflecting the opposite wall, and next to it a calm pond in a garden reflecting trees`

### 2.3 Cielo liso o pared lisa
`A plain white wall with a single small window and a clear flat blue sky above it, almost no texture or detail`

### 2.4 Niebla
`A foggy forest road at dawn, trees fading into thick fog, a person walking far away, strong atmospheric haze`

### 2.5 Contraluz
`A person standing in front of a low sun, strong backlight, rim light around the hair, lens flare, dark silhouette
against a bright sky`

### 2.6 Escena nocturna
`A city street at night with low light, neon signs, wet asphalt reflections, a few people and cars, deep shadows`

### 2.7 Primer plano muy cerca y fondo muy lejos
`A hand holding a flower very close to the lens, sharp in the foreground, with a huge mountain valley far in the
background, extreme depth range`

### 2.8 Plano casi sin profundidad
`A flat frontal photograph of a brick wall with a poster on it, shot perfectly straight on, no perspective`

---

## 3. Áreas y bandas (`areas/`)

### 3.1 Varios sujetos a distancias distintas en fila
`Five people standing in a row at clearly increasing distances from the camera, along a long straight path in a park,
all fully visible, evenly spaced in depth, same clothing style`

### 3.2 Sujeto muy pequeño en un fondo grande (para el cuentagotas del foco)
`A tiny red-jacketed hiker in a vast empty desert landscape with big dunes, the person occupies less than 2 percent of
the image`

---

## 4. Para probar la app, no solo el modelo (`formatos/`)

Estas **no** se piden a la IA; se hacen a partir de una foto buena con un editor o un script (te puedo preparar el
script):

- Fotos **grandes (24 Mpx)**: reescalar una buena, p. ej. a 6048×8064.
- Formas: **vertical**, **cuadrada** y **panorámica muy ancha** (21:9 o más).
- Formatos: JPG con perfil de color, PNG, TIFF de 16 bits, WebP y HEIC.
- Una en **blanco y negro**, otra en **CMYK** y otra **PNG con transparencia** (canal alfa).
- **Nombres con tildes, espacios o símbolos** y una **ruta larga**.
- Una **muy pequeña** (menos de 500 px).
- Una con **orientación EXIF girada** (comprueba que no sale de lado).

## 5. Mapas de profundidad ya hechos, para importar (`formatos/`)

- Un mapa de **8 bits**, otro de **16 bits** y uno **invertido** (cerca = claro) que haga falta invertir.
- Si puedes, uno de otra herramienta (p. ej. ComfyUI con Marigold V2) para comparar.
