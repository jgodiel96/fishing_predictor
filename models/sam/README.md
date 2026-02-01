# SAM (Segment Anything Model) Checkpoint

Este directorio contiene el checkpoint del modelo SAM de Meta AI.

## Descarga del Modelo

El modelo SAM ViT-H (el mas preciso) requiere ~2.4GB de espacio.

### Opcion 1: Descarga directa (recomendado)

```bash
# Desde la raiz del proyecto
cd models/sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Opcion 2: curl

```bash
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -o models/sam/sam_vit_h_4b8939.pth
```

## Verificacion

Despues de descargar, verifica que el archivo existe:

```bash
ls -lh models/sam/sam_vit_h_4b8939.pth
# Deberia mostrar ~2.4GB
```

## Modelos Disponibles

| Modelo | Tamano | Precision | Uso |
|--------|--------|-----------|-----|
| vit_h | 2.4GB | Maxima | Recomendado para produccion |
| vit_l | 1.2GB | Alta | Buen balance |
| vit_b | 375MB | Buena | Desarrollo rapido |

Este proyecto usa `vit_h` por defecto para maxima precision en deteccion de costa.

## Dependencias

```bash
pip install segment-anything torch torchvision
```

Para aceleracion en Mac M1/M2/M3:
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

## Links

- [Repositorio SAM](https://github.com/facebookresearch/segment-anything)
- [Paper SAM](https://arxiv.org/abs/2304.02643)
- [Demo Online](https://segment-anything.com/)
