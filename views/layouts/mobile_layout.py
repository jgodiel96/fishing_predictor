"""
Mobile layout for map view.
Full-screen map with bottom sheet overlay.
"""

from views.styles.map_styles import get_base_css, get_mobile_css, COLORS


class MobileLayout:
    """
    Mobile layout manager with bottom sheet pattern.

    Structure:
    ┌────────────────────────┐
    │                        │
    │         MAPA           │
    │      (100% width)      │
    │                        │
    │    [FAB]               │
    ├────────────────────────┤
    │ ══════════════════════ │  <- Handle
    │                        │
    │     BOTTOM SHEET       │
    │   (zonas, condiciones) │
    │                        │
    └────────────────────────┘
    """

    def __init__(self):
        self.sheet_states = ['collapsed', 'half', 'expanded']

    def get_css(self) -> str:
        """Get mobile-specific CSS with bottom sheet."""
        return f'''
        {get_base_css()}
        {get_mobile_css()}
        <style>
            .mobile-layout {{
                position: relative;
                height: 100vh;
                width: 100vw;
                overflow: hidden;
            }}

            .mobile-map-container {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
            }}

            .bottom-sheet {{
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: white;
                border-radius: 16px 16px 0 0;
                box-shadow: 0 -4px 20px rgba(0,0,0,0.15);
                transition: transform 0.3s ease;
                z-index: 1000;
                max-height: 85vh;
                overflow-y: auto;
            }}

            .bottom-sheet.collapsed {{
                transform: translateY(calc(100% - 60px));
            }}

            .bottom-sheet.half {{
                transform: translateY(50%);
            }}

            .bottom-sheet.expanded {{
                transform: translateY(0);
            }}

            .bottom-sheet-handle {{
                width: 40px;
                height: 4px;
                background: #ccc;
                border-radius: 2px;
                margin: 12px auto;
                cursor: grab;
            }}

            .bottom-sheet-content {{
                padding: 0 16px 16px 16px;
            }}

            .fab {{
                position: fixed;
                bottom: 80px;
                right: 16px;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                background: {COLORS['ui']['primary']};
                color: white;
                border: none;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                font-size: 24px;
                cursor: pointer;
                z-index: 999;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .fab:active {{
                transform: scale(0.95);
            }}

            /* Hide desktop panels on mobile */
            @media (max-width: 768px) {{
                #timeline-panel,
                #hourly-panel,
                #timeline-btn,
                #hourly-btn {{
                    display: none !important;
                }}
            }}
        </style>
        '''

    def get_bottom_sheet_js(self) -> str:
        """JavaScript for bottom sheet interaction."""
        return '''
        <script>
            let sheetState = 'half';
            const sheet = document.getElementById('bottom-sheet');
            const handle = document.querySelector('.bottom-sheet-handle');

            if (handle && sheet) {
                handle.addEventListener('click', function() {
                    if (sheetState === 'half') {
                        sheet.className = 'bottom-sheet expanded';
                        sheetState = 'expanded';
                    } else if (sheetState === 'expanded') {
                        sheet.className = 'bottom-sheet collapsed';
                        sheetState = 'collapsed';
                    } else {
                        sheet.className = 'bottom-sheet half';
                        sheetState = 'half';
                    }
                });

                // Touch drag support
                let startY = 0;
                let startTransform = 0;

                handle.addEventListener('touchstart', function(e) {
                    startY = e.touches[0].clientY;
                    sheet.style.transition = 'none';
                });

                handle.addEventListener('touchmove', function(e) {
                    const deltaY = e.touches[0].clientY - startY;
                    const currentTransform = parseInt(sheet.style.transform.replace(/[^0-9-]/g, '')) || 0;
                    sheet.style.transform = `translateY(${Math.max(0, currentTransform + deltaY)}px)`;
                });

                handle.addEventListener('touchend', function(e) {
                    sheet.style.transition = 'transform 0.3s ease';
                    const rect = sheet.getBoundingClientRect();
                    const screenHeight = window.innerHeight;
                    const sheetTop = rect.top;

                    if (sheetTop > screenHeight * 0.7) {
                        sheet.className = 'bottom-sheet collapsed';
                        sheetState = 'collapsed';
                    } else if (sheetTop > screenHeight * 0.3) {
                        sheet.className = 'bottom-sheet half';
                        sheetState = 'half';
                    } else {
                        sheet.className = 'bottom-sheet expanded';
                        sheetState = 'expanded';
                    }
                    sheet.style.transform = '';
                });
            }

            // FAB action
            function onFabClick() {
                const sheet = document.getElementById('bottom-sheet');
                if (sheet) {
                    sheet.className = 'bottom-sheet expanded';
                    sheetState = 'expanded';
                }
            }
        </script>
        '''

    def wrap_content(self, map_html: str, sheet_content: str = '') -> str:
        """Wrap map with bottom sheet in mobile layout."""
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <title>Predictor de Pesca</title>
            {self.get_css()}
        </head>
        <body style="margin:0;padding:0;">
            <div class="mobile-layout">
                <div class="mobile-map-container">
                    {map_html}
                </div>

                <button class="fab" onclick="onFabClick()">
                    ⚓
                </button>

                <div id="bottom-sheet" class="bottom-sheet half">
                    <div class="bottom-sheet-handle"></div>
                    <div class="bottom-sheet-content">
                        {sheet_content}
                    </div>
                </div>
            </div>
            {self.get_bottom_sheet_js()}
        </body>
        </html>
        '''

    def build_sheet_content(self, spots_summary: dict = None) -> str:
        """Build content for the bottom sheet."""
        if not spots_summary:
            return '<p>Cargando datos...</p>'

        return f'''
        <h3 style="margin:0 0 12px 0;color:{COLORS['ui']['primary']};">Mejores Spots</h3>

        <div style="display:flex;gap:12px;margin-bottom:16px;">
            <div style="flex:1;background:#e3f2fd;padding:12px;border-radius:8px;text-align:center;">
                <div style="font-size:24px;font-weight:bold;color:{COLORS['ui']['primary']};">
                    {spots_summary.get('best_score', '--')}
                </div>
                <small>Mejor Score</small>
            </div>
            <div style="flex:1;background:#e8f5e9;padding:12px;border-radius:8px;text-align:center;">
                <div style="font-size:24px;font-weight:bold;color:{COLORS['ui']['success']};">
                    {spots_summary.get('good_spots', '--')}
                </div>
                <small>Spots Buenos</small>
            </div>
        </div>

        <div style="border-top:1px solid #eee;padding-top:12px;">
            <small style="color:#666;">Desliza hacia arriba para ver mas detalles</small>
        </div>
        '''
