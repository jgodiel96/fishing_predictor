# Investigacion Completa: Estado del Arte en Vision Computacional para Deteccion Costera

## Resumen Ejecutivo

Este documento compila una investigacion exhaustiva de papers academicos, tesis, conferencias y recursos tecnicos sobre deteccion de linea costera, segmentacion de agua, clasificacion de sustrato y batimetria derivada de satelite.

---

## HALLAZGO CRITICO: Incompatibilidad de Datos

Durante la investigacion se identifico un problema fundamental:

```
┌─────────────────────────────────────────────────────────────────┐
│  PROBLEMA: Los modelos pre-entrenados (DeepWaterMap, WatNet)   │
│  requieren 6 bandas espectrales (RGB + NIR + SWIR).            │
│                                                                 │
│  Los tiles ESRI World Imagery son solo RGB (3 bandas).         │
│                                                                 │
│  CONCLUSION: No es posible usar estos modelos directamente     │
│  con las imagenes que estamos descargando.                     │
└─────────────────────────────────────────────────────────────────┘
```

### Opciones Viables Identificadas:

1. **Usar datos OSM + GEBCO** (recomendado): Linea costera ya procesada + batimetria real
2. **Descargar Sentinel-2 real**: Permite usar modelos pre-entrenados correctamente
3. **Usar SAM/SamGeo**: Funciona con RGB pero requiere GPU
4. **Entrenar modelo propio**: Requiere crear dataset local

Ver: `PLAN_V8_CV_ACTUALIZADO.md` para el nuevo enfoque de implementacion.

---

---

## 1. ARQUITECTURAS DE SEGMENTACION SEMANTICA

### 1.1 U-Net y Variantes

**Paper Fundamental:**
- Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"

**Variantes para Remote Sensing:**

| Modelo | Paper/Fuente | Precision Reportada | Caracteristicas |
|--------|--------------|---------------------|-----------------|
| **U-Net Original** | MICCAI 2015 | ~95% IoU | Encoder-decoder con skip connections |
| **EU-Net** | [Scientific Reports 2024](https://www.nature.com/articles/s41598-024-67113-7) | 97.31% accuracy, 93.51% IoU | Multi-scale information fusion |
| **AER U-Net** | [Scientific Reports 2025](https://www.nature.com/articles/s41598-025-99322-z) | >99% accuracy | Attention-enhanced residual blocks |
| **NDR U-Net** | [MDPI 2024](https://iieta.org/journals/ria/paper/10.18280/ria.380324) | 95.36% IoU, 96.99% accuracy | Nested Dense Residual |
| **BDCN_UNet** | [Springer 2025](https://link.springer.com/article/10.1007/s12145-024-01693-w) | Estado del arte | Integra segmentacion + deteccion de bordes |
| **HED-UNet** | Heidler et al. | Alta para SAR | Holistically Nested Edge Detection + UNet |

**Hallazgo Clave:**
> "U-Net usando input-image de 512x512 proporciona el rendimiento mas alto de 98% con loss function de 0.16" - [ScienceDirect Vietnam Study](https://www.sciencedirect.com/science/article/abs/pii/S0301479722013056)

### 1.2 DeepLabV3+ y PSPNet

**DeepLabV3+:**
- Arquitectura encoder-decoder con Atrous Spatial Pyramid Pooling (ASPP)
- Usa Xception como backbone
- [Paper Review](https://sh-tsang.medium.com/review-deeplabv3-atrous-separable-convolution-semantic-segmentation-a625f6e83b90)

**Comparacion:**
> "DeepLabV3+ supera a PSPNet porque no solo expande el campo receptivo mediante convolucion atrous sino que tambien fusiona informacion contextual mediante skip connections" - [Scientific Reports](https://www.nature.com/articles/s41598-024-60375-1)

**Rendimiento en LoveDA dataset:**
- DeepLabV3+: MIOU base
- Metodo mejorado: +9.76% sobre DeepLabv3Plus, +6.65% sobre PSPNet

### 1.3 Vision Transformers (ViT) para Segmentacion

**Modelos Recientes (2024-2025):**

| Modelo | Fuente | Caracteristicas |
|--------|--------|-----------------|
| **SatViT-Seg** | [MDPI Dec 2025](https://www.mdpi.com/2072-4292/18/1/1) | Backbone puro ViT, ligero, +1.81% mIoU |
| **VistaFormer** | [arXiv Sep 2024](https://arxiv.org/abs/2409.08461) | Multi-escala, position-free self-attention |
| **MeViT** | [Panboonyuen](https://kaopanboonyuen.github.io/publication/mevit-a-medium-resolution-vision-transformer/) | 92.22% precision, 83.63% mIoU |
| **SegFormer** | Multiple papers | Mejor para transfer learning |

**Hallazgo Importante:**
> "Pre-training con datos RS ofrece mejor punto de partida que ImageNet, y transformers exhiben rendimiento superior" - [arXiv 2025](https://arxiv.org/html/2502.10669v1)

---

## 2. DETECCION DE LINEA COSTERA

### 2.1 Metodos Deep Learning Especificos

**CCESAR (2025)** - [arXiv](https://arxiv.org/html/2501.12384v1):
- Two-stage: CNN classification + U-Net segmentation
- Supera single U-Net para diferentes tipos de costa
- Aplicado a imagenes Sentinel-1 SAR

**Coast Train Dataset + CNN:**
- 85% accuracy, 80% IoU en segmentacion
- Input resolution: 512x512 pixels
- [ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0098300424001870)

**Modified U-Net para Coastline:**
- Fusion RGB + NIR
- +1.68% y +8.95% mejora IoU en China y Mar Caspio
- [ScienceDirect 2022](https://www.sciencedirect.com/science/article/pii/S0303243422001118)

### 2.2 Herramientas de Codigo Abierto

| Herramienta | GitHub | Descripcion |
|-------------|--------|-------------|
| **CoastSat** | [kvos/CoastSat](https://github.com/kvos/CoastSat) | 40 anos de shoreline, Landsat + Sentinel-2 |
| **CoastSeg** | [SatelliteShorelines/CoastSeg](https://github.com/SatelliteShorelines/CoastSeg) | Extension con modelos de segmentacion DL |
| **DeepWaterMap** | [isikdogan/deepwatermap](https://github.com/isikdogan/deepwatermap) | CNN pre-entrenado, >95% accuracy |
| **WatNet** | [xinluo2018/WatNet](https://github.com/xinluo2018/WatNet) | Sentinel-2, >95% accuracy |
| **SamGeo** | [opengeos/segment-geospatial](https://github.com/opengeos/segment-geospatial) | SAM para datos geoespaciales |

### 2.3 Indices Espectrales Tradicionales

**NDWI (McFeeters, 1996):**
```
NDWI = (Green - NIR) / (Green + NIR)
```
- Valores > 0.5 = agua
- Sensible a estructuras urbanas

**MNDWI (Xu, 2006):**
```
MNDWI = (Green - SWIR) / (Green + SWIR)
```
- Mejor supresion de ruido urbano
- [Paper original](https://www.tandfonline.com/doi/full/10.1080/01431160600589179)

---

## 3. SEGMENTACION AGUA/TIERRA

### 3.1 Datasets y Benchmarks

| Dataset | Tamano | Sensor | Fuente |
|---------|--------|--------|--------|
| **SNOWED** | 4,334 samples | Sentinel-2 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11397981/) |
| **S1S2-Water** | 65 escenas globales | Sentinel-1/2 | IEEE JSTARS 2024 |
| **Chengdu Dataset** | 400 tiles | Sentinel-2 | Yuan et al. |
| **SWED** | 49 locations | Sentinel-2 | Diverso para coastline |

### 3.2 Modelos Pre-entrenados

**DeepWaterMap v2.0:**
- Entrenado en Landsat-8
- Soporta Landsat-5,7,8,9 y Sentinel-2 sin retraining
- Robusto contra nubes y ruido
- [Checkpoint disponible](https://github.com/isikdogan/deepwatermap)

**WatNet:**
- Usa bandas Sentinel-2: B2, B3, B4, B8, B11, B12
- >95% accuracy en todas las regiones de prueba
- [GitHub](https://github.com/xinluo2018/WatNet)

### 3.3 SAR para Segmentacion de Agua

**Ventajas:**
- All-weather, all-day operation
- No afectado por nubes

**Modelos:**
- **SARSNet** (2024): CNN especializado para SAR water segmentation
- **DBU-Net**: Dual-branch encoder para clasificacion de hielo/agua
- [Review Paper Springer 2023](https://link.springer.com/article/10.1007/s11831-023-10000-7)

---

## 4. CLASIFICACION DE SUSTRATO BENTONICO

### 4.1 Deep Learning para Sustrato

**Paper Destacado - Journal of Applied Ecology 2023:**
- **98.19% validation accuracy**
- CNN entrenado en ~70,000 imagenes clasificadas
- 6 clases de sustrato bentonico
- "Mas consistente que anotadores humanos"
- [Paper](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/1365-2664.14408)

**Comparacion de Metodos (Taylor & Francis 2023):**

| Metodo | Overall Accuracy |
|--------|------------------|
| CNN | 89.80% |
| SVM | 84.19% |
| Random Forest | 80.74% |

### 4.2 Transfer Learning con VGG16

**Enfoque Accesible (ScienceDirect 2024):**
- VGG16 pre-entrenado como extractor de features
- SVM como clasificador final
- Test accuracy: 0.87-0.95 (promedio 0.9)
- No requiere GPU potente
- [Paper](https://www.sciencedirect.com/science/article/pii/S1574954124001614)

### 4.3 Clases de Sustrato Comunes

**Clasificacion amplia:**
- Soft Substrate (arenas, lodos)
- Hard Substrate (gravas, rocas, cantos)
- Reef (coral, estructuras biologicas)

**Clasificacion major level:**
- Coral reef
- Seagrass
- Macroalgae
- Bare substratum

---

## 5. BATIMETRIA DERIVADA DE SATELITE (SDB)

### 5.1 Algoritmos Fundamentales

**Lyzenga (1978, 1985):**
- Modelo lineal multi-banda
- Efectivo hasta ~15m
- Sensible a variabilidad de sustrato

**Stumpf Ratio Transform (2003):**
```
depth = m0 + m1 * ln(n * Blue / Green)
```
- Independiente del albedo del fondo
- Efectivo hasta ~25m en agua clara
- [Paper original](https://www.sciencedirect.com/science/article/pii/S0034425722001584)

**Comparacion:**
> "El ratio transform puede recuperar profundidades hasta 25m en agua clara, mientras Lyzenga no distingue >15m" - [MDPI 2022](https://www.mdpi.com/2072-4292/14/3/772)

### 5.2 Machine Learning para SDB

**Sentinel-2 + ML (Taylor & Francis 2023):**
- XAI (Explainable AI) para estimacion de profundidad
- Mejora sobre metodos empiricos tradicionales
- [Paper](https://www.tandfonline.com/doi/full/10.1080/19475705.2023.2225691)

### 5.3 Limitaciones

- Sobreestimacion en aguas poco profundas
- Subestimacion en zonas profundas
- Afectado por: turbidez, clorofila, tipo de fondo

---

## 6. SEGMENT ANYTHING MODEL (SAM) PARA GEOESPACIAL

### 6.1 Implementaciones Disponibles

**segment-geospatial (SamGeo):**
- [GitHub](https://github.com/opengeos/segment-geospatial)
- [Documentacion](https://samgeo.gishub.org/)
- Publicado en Journal of Open Source Software

**Instalacion:**
```bash
# Conda
conda create -n geo python
conda activate geo
conda install -c conda-forge segment-geospatial

# Con GPU
conda install -c conda-forge segment-geospatial "pytorch=*=cuda*"

# Pip con opciones
pip install "segment-geospatial[samgeo]"  # SAM basico
pip install "segment-geospatial[samgeo2]"  # SAM 2
pip install "segment-geospatial[samgeo3]"  # SAM 3
pip install "segment-geospatial[text]"     # Con prompts de texto
```

**Requisitos:**
- GPU con minimo 8GB VRAM recomendado
- Disponible Google Colab gratuito

### 6.2 Geo-SAM (Plugin QGIS)

- [GitHub](https://github.com/coolzhao/Geo-SAM)
- Soporta imagenes de 1-2 bandas
- Puede usar NDWI, NDVI como input
- Consultas en tiempo real (milisegundos)

### 6.3 CSW-SAM para Water Bodies

- Basado en SAM2
- Cross-scale learning: 10m -> 0.3m resolution
- [ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S0924271625002709)

---

## 7. FUNCIONES DE PERDIDA PARA CLASES DESBALANCEADAS

### 7.1 Comparacion de Loss Functions

| Loss Function | Fortaleza | Debilidad |
|---------------|-----------|-----------|
| **Cross Entropy** | Simple, estable | Falla con desbalance |
| **Focal Loss** | Enfoca en samples dificiles | Requiere tuning de gamma |
| **Dice Loss** | Bueno para desbalance | Gradientes que desaparecen |
| **Tversky Loss** | Control precision/recall | Mas parametros |
| **Unified Focal** | Generaliza Dice + CE | Mas complejo |

### 7.2 Unified Focal Loss (2021)

- Generaliza Dice y Cross Entropy
- Robusto para class imbalance
- [Paper](https://www.sciencedirect.com/science/article/pii/S0895611121001750)
- [GitHub](https://github.com/mlyg/unified-focal-loss)

### 7.3 Recomendacion

> "Para segmentacion de imagenes con clases desbalanceadas, la combinacion de Dice loss con variaciones de Cross Entropy esta entre las mejores" - [arXiv Survey](https://arxiv.org/html/2312.05391v1)

---

## 8. DETECCION DE BORDES

### 8.1 HED (Holistically-Nested Edge Detection)

**Paper:** Xie & Tu, ICCV 2015
- [arXiv](https://arxiv.org/abs/1504.06375)
- [GitHub](https://github.com/s9xie/hed)

**Ventajas sobre Canny:**
- End-to-end learning
- No requiere tuning manual de umbrales
- Mejor preservacion de boundaries
- ODS F-score: 0.790 en BSDS500

### 8.2 Aplicacion a Coastline

> "HED muestra ventaja clara en consistencia sobre Canny" - [PyImageSearch Tutorial](https://pyimagesearch.com/2019/03/04/holistically-nested-edge-detection-with-opencv-and-deep-learning/)

---

## 9. METODOS DE AUGMENTACION DE DATOS

### 9.1 Transformaciones Geometricas

- **Rotacion**: 90, 180, 270 grados
- **Flip**: Horizontal, vertical
- **Scaling**: Magnificacion controlada
- **Cropping**: Recortes aleatorios

### 9.2 Augmentaciones Especificas para Satelite

- **Gaussian Noise**: Simula perturbaciones atmosfericas
- **Channel Averaging**: Fusion de canales
- **Spectral Augmentation**: Variacion de bandas

**Hallazgo:**
> "Transformaciones geometricas preservan la categoria semantica y reducen sensibilidad a posicion y angulo" - [MDPI Review 2023](https://www.mdpi.com/2072-4292/15/3/827)

---

## 10. APRENDIZAJE DEBILMENTE/SEMI-SUPERVISADO

### 10.1 SAM + Weak Supervision

**Metodo (ScienceDirect 2025):**
1. AESAM: EfficientViT-SAM adaptado
2. Pseudo-labels basados en incertidumbre
3. Consistency constraint

**Resultados:**
- +5.89% a +10.56% mejora en F1
- +5.95% a +11.13% mejora en IoU
- [Paper](https://www.sciencedirect.com/science/article/pii/S1569843225000871)

### 10.2 Pseudo-Label Generation

**Clase-aware Cross Entropy (CCE):**
- Distingue solo clases concurrentes
- Simplifica target de pseudo-label
- [Pattern Recognition 2022](https://www.sciencedirect.com/science/article/abs/pii/S003132032200406X)

---

## 11. ESTIMACION DE INCERTIDUMBRE

### 11.1 Bayesian U-Net para Earth Observation

- Monte-Carlo Dropout en inferencia
- Mapas de incertidumbre pixel-wise
- [MDPI 2021](https://www.mdpi.com/2072-4292/13/19/3836)

### 11.2 Metricas de Incertidumbre

- Confidence maps
- Entropy
- Mutual information
- Expected KL divergence

### 11.3 Aplicacion Practica

> "Cuando no hay ground truth, se pueden mantener solo pixeles con predicciones de alta confianza" - [arXiv 2024](https://arxiv.org/html/2510.19586v1)

---

## 12. SUPERPIXELS PARA SEGMENTACION

### 12.1 SLIC para Water Extraction

**EDC-SLIC + MFW-Otsu (IEEE JSTARS 2024):**
- SLIC modificado con convolucion 8-direcciones
- Extraccion fina de cuerpos de agua en SAR
- [ResearchGate](https://www.researchgate.net/publication/396042511)

### 12.2 Ventajas

- Reduce redundancia de informacion
- Preserva boundaries con over-segmentation
- Mas eficiente que pixel-wise processing

---

## 13. FUSION MULTI-MODAL Y ENSEMBLE

### 13.1 Late Fusion Deep Learning

**FLAIR Competition Winner:**
- Combina VHR imagery + SITS
- 64.52% mIoU
- [arXiv 2024](https://arxiv.org/html/2410.00469v1)

### 13.2 Ensemble Methods

**Tecnicas:**
- Majority voting
- Weighted averaging
- Stacking

> "Ensembles llevan a predicciones mas precisas y robustas, menos propensas a overfitting" - [ScienceDirect Chapter](https://www.sciencedirect.com/science/article/abs/pii/B9780443190773000158)

---

## 14. TESIS DOCTORALES RELEVANTES

### 14.1 University of Edinburgh (2024)
- **Tema:** ML para extraccion de shoreline con Sentinel-2
- **Metodos:** ANN, CART para wet/dry boundary
- **Precision:** ±2.15m a ±10.67m
- [Tesis](https://era.ed.ac.uk/handle/1842/42028)

### 14.2 Stanford University
- **Tema:** ML para satellite imagery con labels escasos
- **Enfoque:** Transfer learning, domain adaptation
- [PDF](https://stacks.stanford.edu/file/druid:zw179sw4070/Stanford_University_PhD_Dissertation-augmented.pdf)

---

## 15. RECOMENDACIONES DE IMPLEMENTACION

### 15.1 Arquitectura Recomendada

```
┌─────────────────────────────────────────────────────────────┐
│           SISTEMA HIBRIDO RECOMENDADO                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NIVEL 1: DATOS BASE                                        │
│  ├── Linea costera: OpenStreetMap (gratuito, preciso)      │
│  ├── Batimetria: GEBCO + SDB para shallow                   │
│  └── Imagenes: Sentinel-2 (10m) o ESRI tiles               │
│                                                             │
│  NIVEL 2: SEGMENTACION                                      │
│  ├── Opcion A: DeepWaterMap/WatNet (pre-entrenado)         │
│  ├── Opcion B: SamGeo con prompts (mas flexible)           │
│  └── Opcion C: U-Net fine-tuned en SNOWED dataset          │
│                                                             │
│  NIVEL 3: CLASIFICACION SUSTRATO                           │
│  ├── Transfer learning: VGG16 + SVM                         │
│  └── O: CNN fine-tuned si hay datos locales                │
│                                                             │
│  NIVEL 4: POST-PROCESAMIENTO                                │
│  ├── Active contours para refinamiento de bordes           │
│  ├── Superpixels SLIC para homogeneidad                    │
│  └── Uncertainty filtering                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 15.2 Configuracion de Entrenamiento

**Loss Function:**
```python
# Combinacion recomendada para agua/tierra
loss = 0.5 * DiceLoss() + 0.5 * FocalLoss(gamma=2.0)
```

**Data Augmentation:**
```python
transforms = [
    RandomRotation([0, 90, 180, 270]),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    GaussianNoise(std=0.01),
]
```

### 15.3 Metricas de Evaluacion

| Metrica | Uso |
|---------|-----|
| IoU (Jaccard) | Overlap general |
| Dice (F1) | Balance precision/recall |
| Boundary IoU | Precision de bordes |
| FOM | Mejor para coastline |

---

## 16. CONCLUSIONES

### Hallazgos Principales:

1. **U-Net y variantes** siguen siendo el estado del arte para segmentacion, con precisiones >95%

2. **Vision Transformers** estan emergiendo con ventajas en captura de contexto global

3. **Transfer learning** desde ImageNet funciona, pero pre-training en datos RS es mejor

4. **SAM/SamGeo** ofrece flexibilidad sin precedentes pero requiere GPU

5. **Modelos pre-entrenados** (DeepWaterMap, WatNet) ofrecen solucion practica inmediata

6. **Datos OSM** para coastline son precisos y evitan necesidad de deteccion

7. **Clasificacion de sustrato** con CNN alcanza 98% accuracy con suficientes datos

### Proximos Pasos Sugeridos:

1. Implementar con **OSM coastline + GEBCO bathymetry** como baseline
2. Agregar **DeepWaterMap** para refinamiento de mascara de agua
3. Evaluar **SamGeo** si se dispone de GPU
4. Fine-tune **U-Net en SNOWED** para caso especifico de Peru

---

## REFERENCIAS COMPLETAS

### Papers Fundamentales
1. Ronneberger et al. (2015) - U-Net
2. Chen et al. (2018) - DeepLabV3+
3. Stumpf et al. (2003) - SDB Ratio Transform
4. Xie & Tu (2015) - HED
5. Kirillov et al. (2023) - Segment Anything

### Reviews y Surveys
- [Developments in deep learning for coastline extraction](https://link.springer.com/article/10.1007/s12145-025-01805-0)
- [Loss Functions Survey](https://arxiv.org/html/2312.05391v1)
- [Transformers for Remote Sensing](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175147/)

### Datasets
- SNOWED, S1S2-Water, BigEarthNet, LoveDA

### Herramientas
- CoastSat, CoastSeg, SamGeo, DeepWaterMap, WatNet

---

*Documento generado: Febrero 2026*
*Investigacion basada en papers 2020-2025*
