"""Ayuda: guía rápida de uso y lista de atajos de teclado.

La lista de atajos se genera leyendo los menús de la ventana: así siempre está completa
y al día, sin mantener una tabla aparte. Los gestos de ratón del lienzo se añaden a mano.
"""

from __future__ import annotations

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QMainWindow,
    QMenu,
    QPushButton,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepshot import APP_NAME


def guide_html() -> str:
    """Guía rápida en HTML sencillo (lo renderiza QTextBrowser)."""
    return QCoreApplication.translate("Help", """
<h2>Guía rápida de {app}</h2>
<p>{app} crea imágenes a partir de la <b>profundidad</b> de una fotografía:
lo cercano y lo lejano se convierten en luz, sombra o color. El trabajo sigue cinco pasos,
visibles en la barra superior (<b>Alt+1…5</b>).</p>

<h3>1 · Importar</h3>
<p>Arrastra fotos a la ventana o usa <b>Archivo › Importar fotografía</b> (Ctrl+O).
Aparecen en la <b>Biblioteca</b> de la izquierda; haz clic en una para trabajar con ella.</p>
<p>A la derecha ves la <b>información de la foto</b> (cámara, objetivo, exposición…; la ubicación GPS está oculta
hasta que la muestres). <b>Reencuadrar…</b> te deja trabajar solo con una zona de la foto, sin tocar el archivo original:
así no se calcula entera una foto grande. <i>Foto entera</i> quita el encuadre.</p>

<h3>2 · Profundidad</h3>
<ul>
<li><b>Calcular profundidad</b> (Ctrl+D) estima el mapa con el modelo elegido.
La primera vez se descarga el modelo (unos 2,6 GB); después funciona sin conexión.</li>
<li><b>Calidad</b>: <i>Rápida</i> para probar, <i>Alta</i> para trabajar,
<i>Máxima</i> para el resultado final. Cada mapa se guarda en caché: volver a una foto es instantáneo.</li>
<li><b>Refinar bordes con la foto</b> ajusta los contornos del mapa a los de la foto.</li>
<li><b>Cálculos de esta foto</b>: cada vez que calculas (o recuperas de la caché) un mapa queda en una lista con lo que
<b>tardó de verdad</b> ese modelo y esa calidad. Selecciona uno para verlo, selecciona dos para compararlos con el
divisor, y <i>Usar este mapa</i> para seguir con él; <i>Quedarme solo con este</i> quita los demás. Los mapas se calculan a
un tamaño fijo (el lado mayor de la calidad elegida) y se amplían al de la foto.</li>
<li>Si ya tienes un mapa hecho en otro programa, usa <b>Cargar mapa de profundidad</b>.</li>
<li>Convención: en pantalla, <b>cerca = claro</b> y <b>lejos = oscuro</b>.</li>
</ul>

<h3>3 · Enfoque y expresión</h3>
<p>Aquí se da forma al efecto. La vista <b>Efecto</b> se actualiza al momento: si tocas un ajuste
estando en otra vista, se pasa a ella sola.</p>
<ul>
<li><b>Estilo</b>: aspectos completos de partida (neblina, silueta fotográfica, foco de luz…). Al <b>pasar el ratón</b>
por la lista ves el resultado en la vista previa sin aplicarlo; las <b>áreas</b> que trae un estilo se quitan al
elegir otro si no las has editado, y las que dibujas tú o editaste se conservan; al elegir uno, su nombre se queda en el desplegable mientras
no toques nada (lo mismo en los presets de rampa y de bokeh).
Con el icono del disquete guardas el tuyo.</li>
<li><b>Relieve</b>: escala de profundidad y perspectiva; cambia la forma de las áreas.</li>
<li><b>Control de profundidad</b>: mueve por separado lo <b>cercano</b>, lo <b>medio</b> y lo <b>lejano</b> del mapa.
Cada zona tiene un centro y un destino: sube el destino para alejarla y bájalo para acercarla.
<i>Centros desde el mapa</i> coloca los centros donde hay profundidad en tu foto. Lo notan la rampa, las bandas,
el desenfoque y las áreas, y se ve en las vistas Efecto y 3D. Con <i>Añadir zona</i> puedes tener hasta tres más. Para ver qué parte del mapa es de cada zona, en la vista Profundidad elige <i>Mapa: zonas de profundidad</i>; ahí mismo el mapa también se puede ver en color (se puede exportar así, en Exportar › Mapa de profundidad).</li>
<li><b>Rampa de color</b>: colores de cerca (izquierda) a lejos (derecha). Arrastra las paradas;
<b>doble clic en una parada</b> elige su color viendo el resultado en vivo; doble clic en la barra añade una;
Supr borra. Puedes guardar tus rampas como presets. Si la base es Negro o Blanco, la rampa no se pinta: al elegir un preset de rampa
(o al pasar el ratón por la lista) la base pasa a Rampa para que se vea.</li>
<li><b>Bandas</b>: franjas de profundidad que se iluminan (o se tiñen) sobre la rampa, negro o blanco. Cada banda tiene posición,
anchura, suavidad, intensidad y los giros <b>Pan</b> y <b>Tilt</b> (hasta ±89°), que inclinan la franja como un plano.
En la vista 3D, la banda elegida lleva <b>tiradores</b>: la flecha la mueve en profundidad, los círculos giran Pan y Tilt
y el cuadrado cambia su grosor.</li>
<li><b>Listas de capas</b> (bandas, áreas y efectos de Look): cada fila tiene el <b>ojo</b> (aplica o salta ese elemento),
la <b>S</b> (<i>solo</i>: en la vista previa solo se ve ese elemento, uno a la vez; no cambia lo que se exporta), el nombre
(doble clic o F2) y el <b>cuadro de color</b> (clic para cambiarlo). Se reordenan con las <b>flechas</b> o <b>arrastrando</b>
la fila con la mano.</li>
<li><b>Áreas</b>: elipsoides o cajas en 3D que aclaran (fuerza +) u oscurecen (fuerza −). Se editan
<b>sobre la imagen</b>:
  <ul>
  <li><b>Dibujar</b> (Mayús+A): pulsa en el centro y arrastra para darle tamaño.</li>
  <li><b>Colocar</b> (P): pulsa y arrastra; el área sigue al puntero, pegada a la superficie, hasta soltar.</li>
  <li>Arrastra un área para moverla; los <b>cuadrados</b> escalan, el <b>círculo</b> superior gira.</li>
  <li>Como en Blender, con el ratón sobre la imagen: <b>G</b> mover, <b>S</b> escalar, <b>R</b> girar;
  <b>X / Y / Z</b> fijan el eje; clic o Intro aceptan, Esc cancela.</li>
  <li>Rueda con <b>Ctrl</b>: tamaño · <b>Mayús</b>: profundidad · <b>Alt</b>: suavidad del borde.</li>
  <li>Cada área tiene su color y su nombre (doble clic en la lista); el anillo punteado interior
  muestra dónde empieza a difuminarse.</li>
  </ul></li>
<li><b>Desenfoque de lente</b>: desenfoca lo que queda lejos de la profundidad en foco, como una cámara real. Elige el <b>foco</b> (0 = lo más cercano, 1 = lo más lejano; mira el mapa en la vista Profundidad), la <b>apertura</b> (f/1.4 desenfoca mucho, f/16 casi nada) y el <b>diafragma</b>: redondo o de 3 a 12 palas, que da forma a los puntos de luz. Con el <b>cuentagotas</b> eliges el foco pulsando sobre la imagen en lo que
quieres nítido, y «Ver zona en foco» lo tiñe de verde. Puedes desenfocar la imagen final o solo la foto antes de
mezclarla. «Cargar bokeh…» trae presets (retrato, macro, cine…) que solo cambian el carácter del desenfoque, no el foco de tu foto, y <b>Ojo de gato</b> recorta en almendra los puntos de luz de los bordes de la imagen.</li>
<li><b>Mezcla y máscaras</b>: añade la foto con un modo de fusión, limitada si quieres a las áreas,
a un rango de profundidad o a una <b>máscara pintada con pincel</b> (B pincel, E borrador,
[ y ] tamaño; el cursor muestra el tamaño y la dureza).</li>
<li>Los campos numéricos se <b>arrastran</b> (Mayús = fino) o se escriben con un clic.</li>
<li>Todo se puede <b>deshacer</b> (Ctrl+Z).</li>
<li><b>Proyecto</b> (Archivo › Guardar proyecto, Ctrl+S; Abrir proyecto, Ctrl+Alt+O): guarda toda la
biblioteca (fotos, mapas, ajustes y máscaras) en un archivo .deepshot para retomarla otro día. Las fotos no se
copian: si cambian de sitio, la aplicación te pide localizarlas. Al abrir la aplicación sale una <b>pantalla de
inicio</b> con tus proyectos recientes (también en Archivo › Proyectos recientes y Archivo › Pantalla de inicio…);
<i>Proyecto nuevo</i> (Ctrl+N) empieza en blanco, y si hay cambios sin guardar te pregunta antes de salir o abrir otro.</li>
<li><b>Preferencias</b> (Archivo › Preferencias…, Ctrl+,): el <b>tema de color</b> (oscuro, claro, medianoche o los tuyos:
cada uno es un archivo .json con colores en la carpeta de temas), la <b>densidad</b> y el <b>redondeo</b> de los controles,
el <b>idioma</b> y la <b>disposición de los paneles</b>: plantillas de fábrica (edición, revisión, pantalla pequeña, con registro), las tuyas
y «Restablecer disposición». Todo se aplica al momento, sin reiniciar (salvo el idioma). En Ver › Tema de color y Ver › Disposición de los
paneles tienes el mismo cambio rápido.</li>
<li><b>Estilo</b> (Archivo › Guardar estilo / Cargar estilo, Ctrl+L): el aspecto artístico como archivo suelto,
para aplicarlo a otras fotos.</li>
<li>Pasa el ratón por cualquier control: su explicación aparece abajo, en el pie de la ventana.</li>
</ul>

<h3>4 · Look</h3>
<p>Acabado opcional: una pila de <b>efectos</b> que se aplican de abajo arriba, como en Photoshop (niveles, contraste,
saturación, grano, nitidez, viñeta, claridad, resplandor, aberración cromática, niebla por profundidad y temperatura).
Cada efecto se puede limitar a un rango de profundidad, mezclar con una opacidad y un modo de fusión, renombrar y
reordenar. <i>Cargar look…</i> trae looks de fábrica (hechos solo con efectos de serie) y puedes guardar los tuyos.
Los <b>módulos</b> (Profundidad › Gestor de modelos y módulos, filtro <i>Look modules</i>) añaden efectos más complejos:
relighting por profundidad, quitar fondo y superresolución.</p>

<h3>5 · Exportar</h3>
<p>Elige <b>Efecto</b>, <b>Mapa de profundidad</b> o <b>Imagen final</b> (el efecto con el Look aplicado, en JPG, TIFF, PNG o
WebP, con tamaño, resolución y datos EXIF a elegir), el formato (PNG de 8 o 16 bits, EXR de 32 bits),
la carpeta y el nombre. Se guarda a <b>resolución completa</b> y nunca se sobrescribe un archivo existente.
El efecto puede guardar también sus ajustes en un .json.</p>

<h3>Ver y comparar</h3>
<ul>
<li>Arriba a la derecha: <b>Foto</b>, <b>Profundidad</b>, <b>Efecto</b> o <b>Comparar</b>
(V cambia de vista, C compara con la foto).</li>
<li>En la vista <b>Efecto</b>, el conmutador <b>2D | 3D</b> (tecla 3) muestra el relieve del mapa en 3D: arrastra para girar, Mayús+arrastrar (o botón central) para mover,
rueda para el zoom, doble clic para restablecer; teclas 1, 2 y 7: frente, lateral y arriba. Elige qué imagen
cubre el relieve (foto, profundidad o efecto).</li>
<li>Las <b>áreas</b> se ven y se editan también en 3D: clic en un contorno la selecciona; en la seleccionada, las
<b>flechas</b> la mueven por eje, los <b>cuadrados</b> escalan cada semieje (Mayús: todos), los <b>anillos</b> la giran
por eje (Ctrl: de 15° en 15°) y el <b>punto central</b> la arrastra en el plano de la pantalla. Esc cancela.
Así se ve y se ajusta la profundidad (Z), que en 2D es difícil de apreciar.</li>
<li>En <b>Comparar</b>, arrastra el tirador azul; el botón ⇆ / ⇅ (Mayús+C) gira el divisor.</li>
<li>Rueda del ratón: zoom. Arrastrar o botón central: desplazar. Ctrl+0: ajustar a la ventana. Ctrl+1: 100 %.</li>
</ul>

<h3>Modelos</h3>
<p><b>Profundidad › Gestor de modelos</b> (Ctrl+M) descarga, instala y borra modelos.
Solo se aceptan pesos <i>.safetensors</i>, por seguridad.</p>

<h3>Biblioteca</h3>
<p>Doble clic en el espacio vacío para importar. <b>Supr</b> (o el icono de la papelera) quita las fotos
seleccionadas de la lista; los archivos no se borran.</p>

<h3>Si algo falla</h3>
<p><b>Ver › Registro</b> muestra el detalle técnico de cada operación; con <i>Copiar</i> lo llevas al portapapeles
y <i>Abrir carpeta</i> muestra el archivo de registro, que se guarda siempre.</p>
""").format(app=APP_NAME)


# Gestos de ratón y teclas que no son acciones de menú.
def _extra_shortcuts() -> list[tuple[str, str, str]]:
    return [
        (QCoreApplication.translate("Help", "Lienzo"), QCoreApplication.translate("Help", "Rueda del ratón"), QCoreApplication.translate("Help", "Acercar / alejar")),
        (QCoreApplication.translate("Help", "Lienzo"), QCoreApplication.translate("Help", "Arrastrar con el botón izquierdo"), QCoreApplication.translate("Help", "Desplazar la imagen")),
        (QCoreApplication.translate("Help", "Lienzo"), QCoreApplication.translate("Help", "Arrastrar el tirador azul"), QCoreApplication.translate("Help", "Mover el divisor al comparar")),
        (QCoreApplication.translate("Help", "Lienzo"), QCoreApplication.translate("Help", "Botón central"), QCoreApplication.translate("Help", "Desplazar la imagen")),
        (QCoreApplication.translate("Help", "Áreas"), QCoreApplication.translate("Help", "Arrastrar un área"), QCoreApplication.translate("Help", "Moverla (Mayús: pegada a la superficie)")),
        (QCoreApplication.translate("Help", "Áreas"), QCoreApplication.translate("Help", "Arrastrar un cuadrado / el círculo"), QCoreApplication.translate("Help", "Escalar un semieje / girar el área")),
        (QCoreApplication.translate("Help", "Áreas"), QCoreApplication.translate("Help", "G · S · R con el ratón sobre la imagen"), QCoreApplication.translate("Help", "Mover · escalar · girar (X / Y / Z: eje)")),
        (QCoreApplication.translate("Help", "Áreas"), QCoreApplication.translate("Help", "Rueda con Ctrl / Mayús / Alt"), QCoreApplication.translate("Help", "Tamaño / profundidad / suavidad del área seleccionada")),
        (QCoreApplication.translate("Help", "Áreas"), QCoreApplication.translate("Help", "Dibujar (Mayús+A) y Colocar (P)"), QCoreApplication.translate("Help", "Crear o mover un área arrastrando")),
        (QCoreApplication.translate("Help", "Lienzo"), QCoreApplication.translate("Help", "Esc"), QCoreApplication.translate("Help", "Salir de la herramienta activa o cancelar el gesto")),
        (QCoreApplication.translate("Help", "Máscara"), QCoreApplication.translate("Help", "Pincel (B), borrador (E), [ y ]"), QCoreApplication.translate("Help", "Pintar la máscara y cambiar el tamaño")),
        (QCoreApplication.translate("Help", "Campos numéricos"), QCoreApplication.translate("Help", "Arrastrar / Mayús+arrastrar"), QCoreApplication.translate("Help", "Cambiar el valor / ajuste fino")),
        (QCoreApplication.translate("Help", "Campos numéricos"), QCoreApplication.translate("Help", "Clic"), QCoreApplication.translate("Help", "Escribir el valor")),
        (QCoreApplication.translate("Help", "Rampa"), QCoreApplication.translate("Help", "Doble clic en una parada"), QCoreApplication.translate("Help", "Elegir su color con vista previa en vivo")),
        (QCoreApplication.translate("Help", "Rampa"), QCoreApplication.translate("Help", "Doble clic en la barra"), QCoreApplication.translate("Help", "Añadir parada")),
        (QCoreApplication.translate("Help", "Rampa"), QCoreApplication.translate("Help", "Supr"), QCoreApplication.translate("Help", "Borrar la parada seleccionada")),
        (QCoreApplication.translate("Help", "Biblioteca"), QCoreApplication.translate("Help", "Arrastrar archivos a la ventana"), QCoreApplication.translate("Help", "Importar fotografías")),
        (QCoreApplication.translate("Help", "Biblioteca"), QCoreApplication.translate("Help", "Doble clic en el espacio vacío"), QCoreApplication.translate("Help", "Importar fotografías")),
        (QCoreApplication.translate("Help", "Biblioteca"), QCoreApplication.translate("Help", "Supr"), QCoreApplication.translate("Help", "Quitar las fotos seleccionadas de la lista")),
        (QCoreApplication.translate("Help", "Gestor de modelos"), QCoreApplication.translate("Help", "Doble clic en un modelo"), QCoreApplication.translate("Help", "Usarlo como activo")),
    ]


class GuideDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(QCoreApplication.translate("Help", "Guía rápida"))
        self.resize(640, 640)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(guide_html())
        close = QPushButton(QCoreApplication.translate("Help", "Cerrar"))
        close.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(browser)
        layout.addWidget(close, alignment=Qt.AlignRight)


class ShortcutsDialog(QDialog):
    """Tabla de atajos: se lee de los menús de la ventana, más los gestos del lienzo."""

    def __init__(self, window: QMainWindow) -> None:
        super().__init__(window)
        self.setWindowTitle(QCoreApplication.translate("Help", "Atajos de teclado"))
        self.resize(560, 600)
        tree = QTreeWidget()
        tree.setHeaderLabels([QCoreApplication.translate("Help", "Acción"), QCoreApplication.translate("Help", "Atajo")])
        tree.setRootIsDecorated(True)
        tree.header().setSectionResizeMode(0, QHeaderView.Stretch)

        for menu_action in window.menuBar().actions():
            menu = menu_action.menu()
            if not isinstance(menu, QMenu):
                continue
            rows = self._collect(menu)
            if not rows:
                continue
            group = QTreeWidgetItem([menu.title().replace("&", ""), ""])
            group.setFirstColumnSpanned(True)
            for text, keys in rows:
                QTreeWidgetItem(group, [text, keys])
            tree.addTopLevelItem(group)

        groups: dict[str, QTreeWidgetItem] = {}
        for area, keys, text in _extra_shortcuts():
            if area not in groups:
                groups[area] = QTreeWidgetItem([area, ""])
                tree.addTopLevelItem(groups[area])
            QTreeWidgetItem(groups[area], [text, keys])

        tree.expandAll()
        tree.resizeColumnToContents(1)
        close = QPushButton(QCoreApplication.translate("Help", "Cerrar"))
        close.clicked.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(tree)
        layout.addWidget(close, alignment=Qt.AlignRight)

    @staticmethod
    def _collect(menu: QMenu) -> list[tuple[str, str]]:
        """Acciones con atajo de un menú (y sus submenús)."""
        rows = []
        for action in menu.actions():
            if action.menu():
                rows += ShortcutsDialog._collect(action.menu())
            elif not action.isSeparator():
                keys = ", ".join(k.toString(QKeySequence.NativeText) for k in action.shortcuts())
                if keys:
                    rows.append((action.text().replace("&", "").rstrip("…").strip(), keys))
        return rows
