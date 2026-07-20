"""
Interpolação Espacial de Nitratos - IDW e Kriging com Deriva de Estufas
v2.0 - com transformação logarítmica e deriva externa (estufas)
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

# Leitura directamente da GDB (sem necessidade de exportar)
CAMINHO_GDB  = r"path/to/water_points.gdb"        # monitoring points with nitrate field
CAMADA_PONTOS = "pontos_agua_merge"

# Alternativa: usar CSV exportado (descomenta se preferires)
# CAMINHO_CSV = r"path/to/water_points.csv"       # optional CSV fallback (x, y, nitratos)

CAMINHO_ESTUFAS = r"path/to/detected_greenhouses.shp"   # external drift: U-Net detections
PASTA_OUTPUT    = r"path/to/output"
CAMPO_NITRATOS  = "nitratos"
LIMITE_EU       = 50.0   # mg/L
RESOLUCAO_M     = 100    # metros

os.makedirs(PASTA_OUTPUT, exist_ok=True)

# =============================================================================
# 1. CARREGAR PONTOS DE ÁGUA
# =============================================================================

print("A carregar dados da GDB...")
try:
    # Leitura directa da File Geodatabase
    gdf = gpd.read_file(CAMINHO_GDB, layer=CAMADA_PONTOS)
    print(f"  Lido da GDB: {len(gdf)} registos")
except Exception as e:
    print(f"  Erro ao ler GDB: {e}")
    print("  A tentar via CSV...")
    df = pd.read_csv(CAMINHO_CSV)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs="EPSG:3763")

# Limpar dados
gdf = gdf.dropna(subset=[CAMPO_NITRATOS, 'geometry'])
gdf = gdf[gdf[CAMPO_NITRATOS] > 0]
gdf = gdf[np.isfinite(gdf[CAMPO_NITRATOS])]

print(f"  Pontos válidos: {len(gdf)}")
print(f"  Nitratos — Min: {gdf[CAMPO_NITRATOS].min():.1f} | "
      f"Max: {gdf[CAMPO_NITRATOS].max():.1f} | "
      f"Média: {gdf[CAMPO_NITRATOS].mean():.1f} | "
      f"Mediana: {gdf[CAMPO_NITRATOS].median():.1f} mg/L")
print(f"  Acima de 50 mg/L: {(gdf[CAMPO_NITRATOS] > LIMITE_EU).sum()} "
      f"({(gdf[CAMPO_NITRATOS] > LIMITE_EU).mean()*100:.1f}%)")

x = gdf.geometry.x.values
y = gdf.geometry.y.values
z = gdf[CAMPO_NITRATOS].values

# =============================================================================
# 2. CARREGAR ESTUFAS E CALCULAR PROXIMIDADE
# =============================================================================

print("\nA carregar estufas...")
try:
    estufas = gpd.read_file(CAMINHO_ESTUFAS)
    # Garantir mesmo CRS
    if estufas.crs != gdf.crs:
        estufas = estufas.to_crs(gdf.crs)

    # Usar centróides das estufas (ou vértices se forem polígonos)
    if estufas.geometry.geom_type[0] == 'Polygon':
        coords_estufas = np.column_stack([
            estufas.geometry.centroid.x,
            estufas.geometry.centroid.y
        ])
    else:
        coords_estufas = np.column_stack([
            estufas.geometry.x,
            estufas.geometry.y
        ])

    tree_estufas = cKDTree(coords_estufas)
    print(f"  {len(estufas)} estufas carregadas")
    TEM_ESTUFAS = True

except Exception as e:
    print(f"  AVISO: não foi possível carregar estufas — {e}")
    print("  A continuar sem deriva de estufas...")
    TEM_ESTUFAS = False

# Função para calcular proximidade a estufas (usada nos pontos e no grid)
def calc_prox_estufas(px, py, tree, raio_influencia=3000):
    """
    Proximidade inversa à estufa mais próxima.
    raio_influencia: distância (m) a partir da qual a estufa já não influencia
    """
    coords = np.column_stack([px, py])
    dist, _ = tree.query(coords)
    # Normalizar: 1 = em cima da estufa, 0 = muito longe
    prox = np.maximum(0, 1 - dist / raio_influencia)
    return prox

if TEM_ESTUFAS:
    prox_pontos = calc_prox_estufas(x, y, tree_estufas)
    print(f"  Proximidade a estufas nos pontos — "
          f"Min: {prox_pontos.min():.3f} | Max: {prox_pontos.max():.3f} | "
          f"Média: {prox_pontos.mean():.3f}")

# =============================================================================
# 3. TRANSFORMAÇÃO LOGARÍTMICA
# =============================================================================

# log1p(x) = log(1+x) — evita log(0) e é invertível com expm1
z_log = np.log1p(z)
print(f"\nTransformação log — z_log range: {z_log.min():.2f} — {z_log.max():.2f}")

# =============================================================================
# 4. GRID DE INTERPOLAÇÃO
# =============================================================================

margem = 500
x_min, y_min, x_max, y_max = gdf.total_bounds
xi = np.arange(x_min - margem, x_max + margem, RESOLUCAO_M)
yi = np.arange(y_min - margem, y_max + margem, RESOLUCAO_M)
xi_grid, yi_grid = np.meshgrid(xi, yi)
print(f"\nGrid: {len(xi)} x {len(yi)} células ({RESOLUCAO_M}m)")

# Proximidade de estufas no grid completo
if TEM_ESTUFAS:
    prox_grid = calc_prox_estufas(
        xi_grid.flatten(), yi_grid.flatten(), tree_estufas
    ).reshape(xi_grid.shape)

# =============================================================================
# 5. IDW (com dados originais, não log — IDW é robusto a outliers)
# =============================================================================

print("\nA calcular IDW (p=2)...")

def idw(x, y, z, xi, yi, power=2):
    xi_flat = xi.flatten()
    yi_flat = yi.flatten()
    zi = np.zeros(len(xi_flat))
    for i, (xp, yp) in enumerate(zip(xi_flat, yi_flat)):
        dist = np.sqrt((x - xp)**2 + (y - yp)**2)
        if np.min(dist) == 0:
            zi[i] = z[np.argmin(dist)]
        else:
            w = 1.0 / dist**power
            zi[i] = np.sum(w * z) / np.sum(w)
    return zi.reshape(xi.shape)

zi_idw = idw(x, y, z, xi_grid, yi_grid, power=2)

def guardar_tif(array, path, xi, yi, crs_epsg=3763):
    transform = from_bounds(xi.min(), yi.min(), xi.max(), yi.max(), len(xi), len(yi))
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=array.shape[0], width=array.shape[1],
        count=1, dtype=np.float64,
        crs=CRS.from_epsg(crs_epsg),
        transform=transform
    ) as dst:
        dst.write(np.array(array, dtype=np.float64), 1)

guardar_tif(zi_idw, os.path.join(PASTA_OUTPUT, "idw_p2.tif"), xi, yi)
print("  IDW guardado.")

# =============================================================================
# 6. KRIGING ORDINÁRIO COM TRANSFORMAÇÃO LOG
# =============================================================================

print("\nA calcular Kriging Ordinário (log-transformado)...")

resultados_kriging = {}
for modelo in ['spherical', 'exponential', 'gaussian']:
    print(f"  Modelo: {modelo}...")
    try:
        ok = OrdinaryKriging(
            x, y, z_log,
            variogram_model=modelo,
            verbose=False,
            enable_plotting=False,
            nlags=15
        )
        zi_log, ss = ok.execute('grid', xi, yi)

        # Reverter transformação log
        zi = np.expm1(np.array(zi_log))
        zi = np.maximum(zi, 0)  # evitar valores negativos residuais

        resultados_kriging[modelo] = {
            'grid': zi,
            'variance': np.array(ss),
            'params': ok.variogram_model_parameters
        }

        guardar_tif(zi, os.path.join(PASTA_OUTPUT, f"kriging_{modelo}_log.tif"), xi, yi)
        guardar_tif(np.array(ss), os.path.join(PASTA_OUTPUT, f"kriging_{modelo}_variancia.tif"), xi, yi)
        print(f"    Parâmetros variograma: {ok.variogram_model_parameters}")

    except Exception as e:
        print(f"    ERRO: {e}")

# =============================================================================
# 7. KRIGING COM DERIVA EXTERNA (estufas) — Universal Kriging
# =============================================================================

if TEM_ESTUFAS:
    print("\nA calcular Universal Kriging com deriva de estufas...")
    try:
        uk = UniversalKriging(
            x, y, z_log,
            variogram_model='exponential',
            drift_terms=['external_Z'],
            external_drift=prox_pontos,
            external_drift_x=x,
            external_drift_y=y,
            verbose=False
        )

        zi_uk_log, ss_uk = uk.execute(
            'grid', xi, yi,
            external_drift_grid=prox_grid
        )

        zi_uk = np.expm1(np.array(zi_uk_log))
        zi_uk = np.maximum(zi_uk, 0)

        resultados_kriging['uk_estufas'] = {
            'grid': zi_uk,
            'variance': np.array(ss_uk)
        }

        guardar_tif(zi_uk, os.path.join(PASTA_OUTPUT, "kriging_uk_estufas.tif"), xi, yi)
        guardar_tif(np.array(ss_uk), os.path.join(PASTA_OUTPUT, "kriging_uk_estufas_variancia.tif"), xi, yi)
        print("  Universal Kriging guardado.")

    except Exception as e:
        print(f"  ERRO no Universal Kriging: {e}")

# =============================================================================
# 8. VALIDAÇÃO CRUZADA LEAVE-ONE-OUT
# =============================================================================

print("\nA realizar validação cruzada...")

# IDW
erros_idw = []
for i in range(len(z)):
    xt, yt, zt = np.delete(x,i), np.delete(y,i), np.delete(z,i)
    dist = np.sqrt((xt - x[i])**2 + (yt - y[i])**2)
    w = 1.0 / dist**2
    erros_idw.append(z[i] - np.sum(w*zt)/np.sum(w))
erros_idw = np.array(erros_idw)
rmse_idw = np.sqrt(np.mean(erros_idw**2))
mae_idw  = np.mean(np.abs(erros_idw))
print(f"  IDW (p=2):              RMSE={rmse_idw:.2f} | MAE={mae_idw:.2f} mg/L")

# Kriging log
for modelo in ['exponential', 'spherical']:
    if modelo not in resultados_kriging:
        continue
    erros = []
    for i in range(len(z)):
        xt, yt, zt = np.delete(x,i), np.delete(y,i), np.delete(z_log,i)
        try:
            ok_loo = OrdinaryKriging(xt, yt, zt, variogram_model=modelo, verbose=False)
            zp, _ = ok_loo.execute('points', np.array([x[i]]), np.array([y[i]]))
            erros.append(z[i] - np.expm1(float(zp[0])))
        except:
            erros.append(np.nan)
    erros = np.array(erros)
    erros = erros[~np.isnan(erros)]
    rmse = np.sqrt(np.mean(erros**2))
    mae  = np.mean(np.abs(erros))
    resultados_kriging[modelo]['rmse'] = rmse
    resultados_kriging[modelo]['mae']  = mae
    print(f"  Kriging log ({modelo:<12}): RMSE={rmse:.2f} | MAE={mae:.2f} mg/L")

# =============================================================================
# 9. VISUALIZAÇÃO
# =============================================================================

print("\nA gerar figuras...")

cores = ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c']
cmap  = mcolors.LinearSegmentedColormap.from_list("nitratos", cores)
vmin, vmax = 0, min(200, np.percentile(z, 95))  # limitar outliers na visualização

# Escolher grids para comparação
grids_plot = [
    ("IDW (p=2)", zi_idw),
    ("Kriging Exp. (log)", resultados_kriging.get('exponential', {}).get('grid', zi_idw)),
]
if TEM_ESTUFAS and 'uk_estufas' in resultados_kriging:
    grids_plot.append(("UK c/ Estufas (log)", resultados_kriging['uk_estufas']['grid']))

fig, axes = plt.subplots(1, len(grids_plot), figsize=(7*len(grids_plot), 8))
if len(grids_plot) == 1:
    axes = [axes]
fig.suptitle("Interpolação Espacial de Nitratos — ZV Esposende/Vila do Conde\n"
             "(Kriging com transformação logarítmica)", fontsize=13, fontweight='bold')

for ax, (titulo, grid) in zip(axes, grids_plot):
    im = ax.pcolormesh(xi_grid, yi_grid, grid, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    ax.scatter(x, y, c=z, cmap=cmap, vmin=vmin, vmax=vmax,
               edgecolors='black', linewidths=0.4, s=20, zorder=5)
    # Isoline 50 mg/L
    try:
        cs = ax.contour(xi_grid, yi_grid, grid, levels=[LIMITE_EU],
                        colors='red', linewidths=1.5, linestyles='--')
        ax.clabel(cs, fmt=f'{LIMITE_EU:.0f} mg/L', fontsize=8)
    except:
        pass
    # Estufas
    if TEM_ESTUFAS:
        estufas.plot(ax=ax, color='yellow', edgecolor='orange',
                     linewidth=0.5, alpha=0.6, zorder=6, markersize=3)
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label="NO₃ (mg/L)", shrink=0.8)

plt.tight_layout()
fig_path = os.path.join(PASTA_OUTPUT, "interpolacao_v2_comparacao.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Figura guardada: {fig_path}")

# =============================================================================
# 10. RESUMO
# =============================================================================

print("\n" + "="*65)
print("RESUMO — INTERPOLAÇÃO v2 (log-transform + estufas)")
print("="*65)
print(f"Pontos de amostragem : {len(z)}")
print(f"Nitratos             : {z.min():.1f} — {z.max():.1f} mg/L "
      f"(média={z.mean():.1f}, mediana={np.median(z):.1f})")
print(f"Estufas incluídas    : {'Sim' if TEM_ESTUFAS else 'Não'}")
print(f"\nMÉTRICAS DE VALIDAÇÃO (com transformação log):")
print(f"  IDW (p=2)              : RMSE={rmse_idw:.2f} | MAE={mae_idw:.2f} mg/L")
for modelo in ['exponential', 'spherical']:
    if modelo in resultados_kriging and 'rmse' in resultados_kriging[modelo]:
        r = resultados_kriging[modelo]
        print(f"  Kriging log ({modelo:<10}): RMSE={r['rmse']:.2f} | MAE={r['mae']:.2f} mg/L")
print(f"\nFicheiros em: {PASTA_OUTPUT}")
print("="*65)