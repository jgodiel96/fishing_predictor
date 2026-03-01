"""
Centralized styles for map visualization.
All colors, dimensions, and CSS are defined here.
"""

# Color palettes (exclusive per layer)
COLORS = {
    'spots': {
        'best': '#FF0000',
        'excellent': '#228B22',
        'good': '#32CD32',
        'regular': '#FFD700',
        'poor': '#DC143C'
    },
    'sst': {
        'cold': '#0000CD',
        'cool': '#1E90FF',
        'fresh': '#00BFFF',
        'optimal_low': '#00FA9A',
        'optimal': '#00FF00',
        'optimal_high': '#ADFF2F',
        'warm': '#FFD700',
        'hot': '#FF8C00',
        'very_hot': '#FF4500'
    },
    'flow': {
        'slow': '#B0C4DE',
        'moderate': '#7B9FCC',
        'fast': '#5078AA',
        'very_fast': '#2E5984'
    },
    'zones': {
        'high': '#00CED1',
        'medium': '#20B2AA',
        'low': '#48D1CC',
        'fill_high': '#00FFFF',
        'fill_medium': '#40E0D0',
        'fill_low': '#7FFFD4'
    },
    'anchovy': {
        'high': '#FF4500',
        'high_fill': '#FF6347',
        'medium': '#FF8C00',
        'medium_fill': '#FFA500',
        'low': '#FFD700',
        'low_fill': '#FFEC8B'
    },
    'ui': {
        'primary': '#0d47a1',
        'secondary': '#1a5f7a',
        'success': '#4caf50',
        'warning': '#ff9800',
        'danger': '#f44336',
        'background': '#f5f5f5',
        'card': '#ffffff',
        'text': '#333333',
        'text_secondary': '#666666',
        'border': '#ddd'
    },
    'movement': '#FFD700',
    'coast': '#CCCC00'
}

# Dimensions
DIMENSIONS = {
    'sidebar_width': '320px',
    'panel_width': '380px',
    'timeline_width': '320px',
    'bottom_sheet_height': '40vh',
    'border_radius': '10px',
    'spacing': '15px',
    'mobile_breakpoint': 768
}


def get_base_css() -> str:
    """Base CSS shared across all components."""
    return '''
    <style>
        .fishing-panel {
            position: fixed;
            z-index: 1000;
            background: rgba(255,255,255,0.97);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            max-height: 90vh;
            overflow-y: auto;
        }

        .fishing-panel::-webkit-scrollbar {
            width: 6px;
        }

        .fishing-panel::-webkit-scrollbar-thumb {
            background: #ccc;
            border-radius: 3px;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .panel-header h3 {
            margin: 0;
        }

        .panel-minimize-btn {
            border: none;
            background: none;
            cursor: pointer;
            font-size: 18px;
        }

        .info-box {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 12px;
        }

        .info-box b {
            color: #1565c0;
        }

        .tab-container {
            display: flex;
            margin-bottom: 10px;
            border-bottom: 2px solid #eee;
        }

        .tab-btn {
            flex: 1;
            padding: 8px;
            border: none;
            background: none;
            cursor: pointer;
        }

        .tab-btn.active {
            font-weight: bold;
            border-bottom: 2px solid;
        }

        .minimized-btn {
            display: none;
            position: fixed;
            z-index: 1000;
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        /* Score colors */
        .score-excellent { color: #4caf50; }
        .score-good { color: #ff9800; }
        .score-poor { color: #f44336; }
    </style>
    '''


def get_panel_css(panel_type: str = 'timeline') -> str:
    """Get CSS for specific panel type."""
    if panel_type == 'timeline':
        return f'''
        <style>
            #timeline-panel {{
                top: 10px;
                right: 10px;
                width: {DIMENSIONS['timeline_width']};
            }}

            #timeline-panel .tab-btn.active {{
                color: {COLORS['ui']['secondary']};
                border-bottom-color: {COLORS['ui']['secondary']};
            }}

            #timeline-btn {{
                top: 10px;
                right: 10px;
                background: {COLORS['ui']['secondary']};
            }}
        </style>
        '''
    elif panel_type == 'hourly':
        return f'''
        <style>
            #hourly-panel {{
                bottom: 10px;
                right: 10px;
                width: {DIMENSIONS['panel_width']};
            }}

            #hourly-panel .tab-btn.active {{
                color: {COLORS['ui']['primary']};
                border-bottom-color: {COLORS['ui']['primary']};
            }}

            #hourly-btn {{
                bottom: 10px;
                right: 10px;
                background: {COLORS['ui']['primary']};
            }}
        </style>
        '''
    return ''


def get_mobile_css() -> str:
    """CSS for mobile bottom sheet pattern."""
    return f'''
    <style>
        @media (max-width: {DIMENSIONS['mobile_breakpoint']}px) {{
            .fishing-panel {{
                position: fixed !important;
                bottom: 0 !important;
                left: 0 !important;
                right: 0 !important;
                top: auto !important;
                width: 100% !important;
                max-width: 100% !important;
                border-radius: 16px 16px 0 0 !important;
                max-height: 70vh !important;
            }}

            .bottom-sheet-handle {{
                display: block;
                width: 40px;
                height: 4px;
                background: #ccc;
                border-radius: 2px;
                margin: 8px auto 12px auto;
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
        }}

        @media (min-width: {DIMENSIONS['mobile_breakpoint'] + 1}px) {{
            .bottom-sheet-handle {{
                display: none;
            }}
        }}
    </style>
    '''


def get_chart_js_cdn() -> str:
    """Return Chart.js CDN script tag."""
    return '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'


def get_score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 80:
        return COLORS['spots']['excellent']
    elif score >= 60:
        return COLORS['spots']['good']
    elif score >= 40:
        return COLORS['spots']['regular']
    return COLORS['spots']['poor']


def get_sst_color(sst: float) -> str:
    """Get color based on SST value."""
    if sst <= 14:
        return COLORS['sst']['cold']
    elif sst <= 16:
        return COLORS['sst']['cool']
    elif sst <= 17:
        return COLORS['sst']['fresh']
    elif sst <= 18:
        return COLORS['sst']['optimal_low']
    elif sst <= 19:
        return COLORS['sst']['optimal']
    elif sst <= 20:
        return COLORS['sst']['optimal_high']
    elif sst <= 21:
        return COLORS['sst']['warm']
    elif sst <= 22:
        return COLORS['sst']['hot']
    return COLORS['sst']['very_hot']


def get_flow_color(speed: float) -> str:
    """Get color based on current flow speed."""
    if speed > 0.3:
        return COLORS['flow']['very_fast']
    elif speed > 0.2:
        return COLORS['flow']['fast']
    elif speed > 0.1:
        return COLORS['flow']['moderate']
    return COLORS['flow']['slow']


def get_zone_colors(intensity: float) -> dict:
    """Get border and fill colors for fish zones."""
    if intensity >= 0.8:
        return {'border': COLORS['zones']['high'], 'fill': COLORS['zones']['fill_high']}
    elif intensity >= 0.6:
        return {'border': COLORS['zones']['medium'], 'fill': COLORS['zones']['fill_medium']}
    return {'border': COLORS['zones']['low'], 'fill': COLORS['zones']['fill_low']}


def get_anchovy_colors(intensity: float) -> dict:
    """Get colors for anchovy zones."""
    if intensity >= 0.7:
        return {'border': COLORS['anchovy']['high'], 'fill': COLORS['anchovy']['high_fill']}
    elif intensity >= 0.4:
        return {'border': COLORS['anchovy']['medium'], 'fill': COLORS['anchovy']['medium_fill']}
    return {'border': COLORS['anchovy']['low'], 'fill': COLORS['anchovy']['low_fill']}


def get_heatmap_color(score: float, min_score: float = 0, max_score: float = 100) -> str:
    """Get heatmap color for spot based on score (blue -> cyan -> green -> yellow -> orange -> red)."""
    # Normalize score to 0-1 range
    if max_score == min_score:
        normalized = 0.5
    else:
        normalized = (score - min_score) / (max_score - min_score)

    # Clamp to 0-1
    normalized = max(0, min(1, normalized))

    # Heatmap gradient
    if normalized < 0.2:
        r, g, b = 0, int(255 * normalized * 5), 255
    elif normalized < 0.4:
        t = (normalized - 0.2) * 5
        r, g, b = 0, 255, int(255 * (1 - t))
    elif normalized < 0.6:
        t = (normalized - 0.4) * 5
        r, g, b = int(255 * t), 255, 0
    elif normalized < 0.8:
        t = (normalized - 0.6) * 5
        r, g, b = 255, int(255 * (1 - t * 0.5)), 0
    else:
        t = (normalized - 0.8) * 5
        r, g, b = 255, int(128 * (1 - t)), 0

    return f'#{r:02x}{g:02x}{b:02x}'
