# Plan de Reestructuracion UI - Mapa de Pesca

**Fecha:** 2026-02-22
**Estado:** Pendiente
**Archivo objetivo:** `views/map_view.py` (800+ lineas)

---

## Problemas Identificados

| # | Problema | Impacto |
|---|----------|---------|
| 1 | Archivo monolitico (800+ lineas) | Dificil de mantener |
| 2 | Mezcla de logica y presentacion | Acoplamiento alto |
| 3 | Panel lateral ocupa espacio valioso | Mala UX movil |
| 4 | Controles dispersos | Confusion del usuario |
| 5 | No hay patron bottom-sheet | No se siente nativo en movil |
| 6 | Dependencias hardcodeadas | Dificil testing |

---

## Arquitectura Propuesta

### Estructura de Archivos

```
views/
├── map_view.py              # Coordinador principal (150 lineas max)
├── components/
│   ├── __init__.py
│   ├── map_component.py     # Mapa Folium + marcadores
│   ├── sidebar.py           # Panel lateral (desktop)
│   ├── bottom_sheet.py      # Bottom sheet (mobile)
│   ├── zone_selector.py     # Selector de zonas
│   ├── conditions_panel.py  # Panel de condiciones
│   └── legend.py            # Leyenda de colores
├── layouts/
│   ├── __init__.py
│   ├── desktop_layout.py    # Layout escritorio
│   └── mobile_layout.py     # Layout movil
└── styles/
    ├── __init__.py
    └── map_styles.py        # CSS centralizado
```

### Patron de Diseno

```
┌─────────────────────────────────────────────────────────┐
│                    MapView (Coordinador)                │
│  - Detecta dispositivo (desktop/mobile)                 │
│  - Selecciona layout apropiado                          │
│  - Coordina comunicacion entre componentes              │
└─────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │  Layout   │   │   Data    │   │  Styles   │
    │  Manager  │   │  Manager  │   │  Manager  │
    └───────────┘   └───────────┘   └───────────┘
```

---

## Fases de Implementacion

### FASE 1: Preparacion (30 min)
- [ ] Crear estructura de directorios `views/components/`, `views/layouts/`, `views/styles/`
- [ ] Crear archivos `__init__.py` en cada directorio
- [ ] Hacer backup de `map_view.py` actual

**Comandos:**
```bash
mkdir -p views/components views/layouts views/styles
touch views/components/__init__.py views/layouts/__init__.py views/styles/__init__.py
cp views/map_view.py views/map_view_backup.py
```

---

### FASE 2: Extraer Estilos (45 min)
- [ ] Crear `views/styles/map_styles.py`
- [ ] Mover todo CSS inline a archivo centralizado
- [ ] Crear funciones `get_desktop_css()` y `get_mobile_css()`
- [ ] Definir variables de colores y dimensiones

**Archivo: `views/styles/map_styles.py`**
```python
# Paleta de colores
COLORS = {
    'primary': '#2196F3',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'danger': '#f44336',
    'background': '#f5f5f5',
    'card': '#ffffff',
    'text': '#333333',
    'text_secondary': '#666666',
}

# Dimensiones
DIMENSIONS = {
    'sidebar_width': '320px',
    'bottom_sheet_height': '40vh',
    'border_radius': '12px',
    'spacing': '16px',
}

def get_base_css() -> str:
    """CSS base compartido"""
    pass

def get_desktop_css() -> str:
    """CSS especifico para desktop"""
    pass

def get_mobile_css() -> str:
    """CSS especifico para mobile"""
    pass
```

---

### FASE 3: Extraer Componentes (2 horas)

#### 3.1 MapComponent (30 min)
- [ ] Crear `views/components/map_component.py`
- [ ] Extraer logica de creacion de mapa Folium
- [ ] Extraer logica de marcadores y popups
- [ ] Extraer logica de colores segun score

**Interfaz:**
```python
class MapComponent:
    def __init__(self, center: tuple, zoom: int):
        pass

    def add_fishing_spots(self, spots: List[FishingSpot]) -> None:
        pass

    def add_legend(self) -> None:
        pass

    def render(self) -> str:
        """Retorna HTML del mapa"""
        pass
```

#### 3.2 ZoneSelector (20 min)
- [ ] Crear `views/components/zone_selector.py`
- [ ] Extraer dropdown/lista de zonas
- [ ] Manejar evento de seleccion

**Interfaz:**
```python
class ZoneSelector:
    def __init__(self, zones: List[str], selected: str = None):
        pass

    def on_select(self, callback: Callable) -> None:
        pass

    def render(self) -> str:
        pass
```

#### 3.3 ConditionsPanel (30 min)
- [ ] Crear `views/components/conditions_panel.py`
- [ ] Extraer visualizacion de SST, olas, viento, etc.
- [ ] Crear mini-graficos/indicadores

**Interfaz:**
```python
class ConditionsPanel:
    def __init__(self, conditions: OceanConditions):
        pass

    def render_compact(self) -> str:
        """Version compacta para bottom sheet"""
        pass

    def render_expanded(self) -> str:
        """Version expandida para sidebar"""
        pass
```

#### 3.4 BottomSheet (30 min)
- [ ] Crear `views/components/bottom_sheet.py`
- [ ] Implementar estados: collapsed, half, expanded
- [ ] Agregar handle para drag
- [ ] Animaciones CSS suaves

**Interfaz:**
```python
class BottomSheet:
    COLLAPSED = 'collapsed'  # Solo handle visible
    HALF = 'half'            # 40% de pantalla
    EXPANDED = 'expanded'    # 85% de pantalla

    def __init__(self, content: str, initial_state: str = HALF):
        pass

    def render(self) -> str:
        pass
```

---

### FASE 4: Crear Layouts (1 hora)

#### 4.1 DesktopLayout
- [ ] Crear `views/layouts/desktop_layout.py`
- [ ] Sidebar fijo a la izquierda (320px)
- [ ] Mapa ocupa resto del espacio
- [ ] Panel inferior para detalles

**Estructura visual:**
```
┌──────────────┬────────────────────────────────┐
│              │                                │
│   SIDEBAR    │           MAPA                 │
│   (320px)    │        (flexible)              │
│              │                                │
│  - Zonas     │                                │
│  - Filtros   │                                │
│  - Info      │                                │
│              ├────────────────────────────────┤
│              │    PANEL INFERIOR (detalles)   │
└──────────────┴────────────────────────────────┘
```

#### 4.2 MobileLayout
- [ ] Crear `views/layouts/mobile_layout.py`
- [ ] Mapa ocupa 100% pantalla
- [ ] Bottom sheet superpuesto
- [ ] Boton flotante para acciones rapidas

**Estructura visual:**
```
┌────────────────────────┐
│                        │
│         MAPA           │
│      (100% width)      │
│                        │
│    [FAB]               │  <- Floating Action Button
├────────────────────────┤
│ ══════════════════════ │  <- Handle para drag
│                        │
│     BOTTOM SHEET       │
│   (zonas, condiciones) │
│                        │
└────────────────────────┘
```

---

### FASE 5: Refactorizar map_view.py (1 hora)
- [ ] Reducir a coordinador de ~150 lineas
- [ ] Detectar tipo de dispositivo
- [ ] Instanciar layout correcto
- [ ] Conectar eventos entre componentes

**Nueva estructura de `map_view.py`:**
```python
import streamlit as st
from views.layouts.desktop_layout import DesktopLayout
from views.layouts.mobile_layout import MobileLayout
from views.components.map_component import MapComponent

class MapView:
    def __init__(self, predictor, data_service):
        self.predictor = predictor
        self.data_service = data_service
        self.is_mobile = self._detect_mobile()

    def _detect_mobile(self) -> bool:
        """Detecta si es dispositivo movil"""
        # Usar streamlit-javascript o user-agent
        pass

    def render(self):
        """Punto de entrada principal"""
        if self.is_mobile:
            layout = MobileLayout(self.predictor, self.data_service)
        else:
            layout = DesktopLayout(self.predictor, self.data_service)

        layout.render()
```

---

### FASE 6: Testing y Polish (1 hora)
- [ ] Probar en desktop (Chrome, Firefox)
- [ ] Probar en mobile (Android Chrome, Safari iOS)
- [ ] Ajustar breakpoints si es necesario
- [ ] Optimizar rendimiento (lazy loading)
- [ ] Documentar componentes

---

## Comparacion Antes/Despues

### ANTES (Actual)
```
views/
└── map_view.py  (800+ lineas, todo mezclado)
```

**Problemas:**
- Un solo archivo hace todo
- CSS inline repetido
- Dificil de testear
- No responsive real

### DESPUES (Propuesto)
```
views/
├── map_view.py              (150 lineas - coordinador)
├── components/
│   ├── map_component.py     (200 lineas)
│   ├── bottom_sheet.py      (100 lineas)
│   ├── zone_selector.py     (80 lineas)
│   └── conditions_panel.py  (120 lineas)
├── layouts/
│   ├── desktop_layout.py    (100 lineas)
│   └── mobile_layout.py     (100 lineas)
└── styles/
    └── map_styles.py        (150 lineas)
```

**Beneficios:**
- Separacion de responsabilidades
- Facil de testear cada componente
- Responsive real (desktop vs mobile)
- Mantenible a largo plazo

---

## Estimacion de Tiempo Total

| Fase | Duracion | Acumulado |
|------|----------|-----------|
| 1. Preparacion | 30 min | 30 min |
| 2. Estilos | 45 min | 1h 15min |
| 3. Componentes | 2h | 3h 15min |
| 4. Layouts | 1h | 4h 15min |
| 5. Refactorizar | 1h | 5h 15min |
| 6. Testing | 1h | 6h 15min |

**Total estimado: ~6-7 horas de desarrollo**

---

## Notas de Implementacion

### Deteccion de Dispositivo Movil
```python
# Opcion 1: streamlit-javascript
from streamlit_javascript import st_javascript
width = st_javascript("window.innerWidth")
is_mobile = width < 768

# Opcion 2: Query params
is_mobile = st.query_params.get("mobile", "false") == "true"
```

### Bottom Sheet CSS
```css
.bottom-sheet {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: white;
    border-radius: 16px 16px 0 0;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.15);
    transition: transform 0.3s ease;
    z-index: 1000;
}

.bottom-sheet.collapsed { transform: translateY(calc(100% - 48px)); }
.bottom-sheet.half { transform: translateY(60%); }
.bottom-sheet.expanded { transform: translateY(15%); }

.bottom-sheet-handle {
    width: 40px;
    height: 4px;
    background: #ccc;
    border-radius: 2px;
    margin: 12px auto;
}
```

---

## Checklist de Completado

- [ ] Fase 1 completada
- [ ] Fase 2 completada
- [ ] Fase 3 completada
- [ ] Fase 4 completada
- [ ] Fase 5 completada
- [ ] Fase 6 completada
- [ ] Commit final realizado
- [ ] Documentacion actualizada

---

*Plan creado: 2026-02-22*
*Ultima actualizacion: 2026-02-22*
