"""
Desktop layout for map view.
Sidebar on left, map fills rest, bottom panel for details.
"""

from views.styles.map_styles import get_base_css, DIMENSIONS


class DesktopLayout:
    """
    Desktop layout manager.

    Structure:
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
    """

    def __init__(self):
        self.sidebar_width = DIMENSIONS['sidebar_width']

    def get_css(self) -> str:
        """Get desktop-specific CSS."""
        return f'''
        {get_base_css()}
        <style>
            .desktop-layout {{
                display: flex;
                height: 100vh;
            }}

            .desktop-sidebar {{
                width: {self.sidebar_width};
                min-width: {self.sidebar_width};
                background: white;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
                overflow-y: auto;
                z-index: 100;
            }}

            .desktop-main {{
                flex: 1;
                display: flex;
                flex-direction: column;
            }}

            .desktop-map-container {{
                flex: 1;
                position: relative;
            }}

            .desktop-bottom-panel {{
                height: auto;
                max-height: 200px;
                background: white;
                border-top: 1px solid #ddd;
                overflow-y: auto;
            }}

            /* Panels positioned relative to main area */
            .fishing-panel {{
                position: absolute;
            }}

            #timeline-panel {{
                top: 10px;
                right: 10px;
            }}

            #hourly-panel {{
                bottom: 10px;
                right: 10px;
            }}

            /* Legend in bottom left */
            .legend-container {{
                position: absolute;
                bottom: 30px;
                left: 30px;
            }}
        </style>
        '''

    def wrap_content(self, map_html: str, sidebar_content: str = '') -> str:
        """Wrap map and sidebar in desktop layout."""
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Predictor de Pesca - Desktop</title>
            {self.get_css()}
        </head>
        <body style="margin:0;padding:0;">
            <div class="desktop-layout">
                <aside class="desktop-sidebar">
                    {sidebar_content}
                </aside>
                <main class="desktop-main">
                    <div class="desktop-map-container">
                        {map_html}
                    </div>
                </main>
            </div>
        </body>
        </html>
        '''
