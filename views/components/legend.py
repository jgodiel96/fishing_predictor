"""
Map legend component.
"""

import folium
from typing import Tuple

from views.styles.map_styles import COLORS


class Legend:
    """
    Legend component for the fishing map.
    """

    def __init__(self, map_obj: folium.Map):
        self.map = map_obj

    def render(self, score_range: Tuple[float, float] = (0, 100)):
        """Add legend to map."""
        if not self.map:
            return

        # Always use fixed 0-100 range for a useful legend
        min_score, max_score = 0, 100
        html = self._build_html(min_score, max_score)
        self.map.get_root().html.add_child(folium.Element(html))

    def _build_html(self, min_score: float, max_score: float) -> str:
        return f'''
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:rgba(255,255,255,0.95);padding:12px;border-radius:8px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.4);font-size:10px;
                    font-family:Arial;max-height:90vh;overflow-y:auto;width:180px;">
            <b style="font-size:13px;">Predictor de Pesca</b><br>
            <small style="color:#666;">ML + Mareas + Hora</small>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b>Score (Heatmap):</b><br>
            <div style="width:100%;height:15px;border-radius:3px;margin:5px 0;
                        background:linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff8000, #ff0000);">
            </div>
            <div style="display:flex;justify-content:space-between;font-size:9px;">
                <span>{min_score:.0f}</span>
                <span>Score</span>
                <span>{max_score:.0f}</span>
            </div>
            <div style="margin-top:3px;">
                <span style="color:#FF0000;">●</span> #1 Mejor spot
            </div>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b style="color:#FF4500;">Anchoveta:</b><br>
            <span style="color:#FF4500;">◯</span> Alta probabilidad<br>
            <span style="color:#FF8C00;">◯</span> Media probabilidad<br>
            🐟 Centro cardumen<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b>SST:</b><br>
            <span style="color:#1E90FF;">◆</span> Frio |
            <span style="color:#00FF00;">◆</span> Optimo |
            <span style="color:#FF4500;">◆</span> Caliente<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <b>Corrientes:</b><br>
            <span style="color:{COLORS['flow']['slow']};">→</span> Lento |
            <span style="color:{COLORS['flow']['very_fast']};">→</span> Rapido<br>

            <hr style="margin:5px 0;border-color:#ddd;">
            <span style="color:{COLORS['zones']['high']};">◯</span> Zona peces
            <span style="color:{COLORS['movement']};">→</span> Movimiento
        </div>
        '''
