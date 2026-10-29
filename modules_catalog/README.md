# Catálogo local de módulos

Aquí van los **módulos** que el gestor de modelos y módulos (Ctrl+M) ofrece para instalar. Un módulo es para cosas
complejas que necesitan un modelo o librerías (relighting, embellecer…): los efectos básicos (niveles, contraste,
saturación, grano, nitidez, viñeta) ya vienen de serie en el paso Look.

Cada módulo es una carpeta con el nombre de su id:

```
modules_catalog/
  mi_modulo/
    module.json
    module.py
```

`module.json`:

```json
{
  "id": "mi_modulo",
  "name": "Mi módulo",
  "version": "1.0",
  "license": "MIT",
  "author": "Quien sea",
  "kind": "look",
  "homepage": "https://ejemplo.com/mi_modulo",
  "description": "Qué hace.",
  "requires_models": ["id-de-un-modelo-del-gestor"],
  "params": [
    {"name": "amount", "label": "Cantidad", "default": 0.0, "min": 0.0, "max": 1.0, "decimals": 2, "step": 0.01,
     "tooltip": "Ayuda del control"}
  ]
}
```

`module.py`:

```python
def process(image, depth, params):
    # image: float32 (alto, ancho, 3) en 0..1; depth: float32 (alto, ancho) en 0..1 (0 = cerca)
    # params: {"amount": ...}. Devuelve la imagen procesada, del mismo tamaño.
    return image
```

Al instalarlo se copia a `<carpeta de modelos>/looks/<id>/`. Su código se ejecuta con tus permisos: la app pide
confirmación al instalarlo y solo lo carga si lo has aprobado. `homepage` (http o https) añade el enlace a su página en
el gestor.
