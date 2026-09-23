# Bitácora de experimentos

Calibración de los hiperparámetros y experimentos sobre MovieLens, en el
orden en que se hicieron. Cada experimento registra la pregunta, el método
y el criterio antes de correr, y después el resultado.

## Calibración de defaults sobre MovieLens real

Dos corridas de calibración hechas con scripts temporales, fuera de `src/`
y `tests/` (sin commitear), sobre `data/ml-100k/` real. Semilla fija = 42,
escala de inicialización = 1.0 en todas, usuario 1 (id crudo) para las
recomendaciones.

### Corrida "k=10" (defaults originales de la demo + exploración de k/eta)

**Resumen del filtro (k=10, que en ese momento era también el umbral):**

```
WARNING:src.datos:filtro por k=10: se eliminaron 0 usuarios, 530 películas y 2047 calificaciones
MovieLens filtrado (k=10): 943 usuarios, 1152 películas, 97953 calificaciones.
```

**ALS (epsilon=1e-2, max_iter=1000) — primeras 5 y últimas 5 iteraciones:**

```
ALS iteración 1: f=70372.7          ALS iteración 996: f=46410.7
ALS iteración 2: f=60252.4          ALS iteración 997: f=46410.6
ALS iteración 3: f=56433.4          ALS iteración 998: f=46410.5
ALS iteración 4: f=54231.9          ALS iteración 999: f=46410.4
ALS iteración 5: f=52810            ALS iteración 1000: f=46410.4
WARNING:src.als: se alcanzó max_iter=1000 sin cortar por tolerancia
```

**GD (eta=1e-4, epsilon=1e-2, max_iter=1000) — primeras 5 y últimas 5 iteraciones:**

```
GD iteración 1: f=228991            GD iteración 996: f=59871.7
GD iteración 2: f=195636            GD iteración 997: f=59858.4
GD iteración 3: f=172057            GD iteración 998: f=59845.2
GD iteración 4: f=155357            GD iteración 999: f=59831.9
GD iteración 5: f=143347            GD iteración 1000: f=59818.7
WARNING:src.gradiente: se alcanzó max_iter=1000 sin cortar por tolerancia
```

**Tabla de comparación:**

| método | iteraciones | motivo de corte | tiempo (s) | SCE final |
|---|---:|---|---:|---:|
| ALS | 1000 | max_iter | 48.2639 | 46410.3539 |
| GD | 1000 | max_iter | 31.7897 | 59818.7316 |

**Top-10 ALS y GD lado a lado, usuario 1:**

| ALS | r̂ | GD | r̂ |
|---|---:|---|---:|
| That Old Feeling (1997) | 15785.72 | Secrets & Lies (1996) | 5.60 |
| I'll Do Anything (1994) | 10546.00 | Ran (1985) | 5.48 |
| Shooting Fish (1997) | 9114.86 | Harold and Maude (1971) | 5.24 |
| Little Odessa (1994) | 5766.62 | Lawrence of Arabia (1962) | 5.23 |
| Sum of Us, The (1994) | 3914.38 | Magnificent Seven, The (1954) | 5.16 |
| Cronos (1992) | 3183.21 | 8 1/2 (1963) | 5.15 |
| For Love or Money (1993) | 2795.84 | Down by Law (1986) | 5.11 |
| Rich Man's Wife, The (1996) | 2468.23 | Duck Soup (1933) | 5.10 |
| Wild Bill (1995) | 1974.99 | Hard Eight (1996) | 5.08 |
| Addiction, The (1995) | 1965.20 | Annie Hall (1977) | 5.06 |

**Extensión: max\|r̂\| dentro/fuera de Ω, k=10, resultado de ALS:**

| max\|r̂\| dentro de Ω | max\|r̂\| fuera de Ω |
|---|---|
| 6.4463 | 170967321.8578 |

Top-10 ALS usuario 1 (misma corrida) con cantidad de calificaciones de esa película:

| película | r̂ | n_calificaciones |
|---|---:|---:|
| That Old Feeling (1997) | 15785.72 | 11 |
| I'll Do Anything (1994) | 10546.00 | 10 |
| Shooting Fish (1997) | 9114.86 | 10 |
| Little Odessa (1994) | 5766.62 | 10 |
| Sum of Us, The (1994) | 3914.38 | 11 |
| Cronos (1992) | 3183.21 | 10 |
| For Love or Money (1993) | 2795.84 | 12 |
| Rich Man's Wife, The (1996) | 2468.23 | 15 |
| Wild Bill (1995) | 1974.99 | 12 |
| Addiction, The (1995) | 1965.20 | 11 |

10 filas de V (ALS) con mayor norma vs. cantidad de calificaciones de esa película:

| película | \|\|V_fila\|\| | n_calificaciones |
|---|---:|---:|
| That Old Feeling (1997) | 73588.28 | 11 |
| Fresh (1994) | 64077.32 | 10 |
| Clean Slate (1994) | 61664.03 | 10 |
| Ponette (1996) | 44083.15 | 10 |
| Wild Things (1998) | 35788.49 | 11 |
| Sum of Us, The (1994) | 28641.27 | 11 |
| Unhook the Stars (1996) | 25024.27 | 10 |
| I'll Do Anything (1994) | 23335.42 | 10 |
| White Man's Burden (1995) | 22381.27 | 10 |
| Bloodsport 2 (1995) | 20180.87 | 10 |

correlación(norma de fila de V, n_calificaciones) = **-0.1112** (casi nula:
la magnitud de la fila de V no depende de cuántas calificaciones tiene esa
película — el problema es otro: filas/columnas cerca del mínimo de
observaciones quedan sub-determinadas).

**ALS con k=2, 3, 5 (epsilon=1, max_iter=1000, mismo umbral=k=10 original
para el filtro, solo cambiando k):**

| k | usuarios | películas | iteraciones | motivo de corte | SCE final | max\|r̂\| fuera Ω |
|---:|---:|---:|---:|---|---:|---:|
| 2 | 943 | 1541 | 27 | tolerancia | 75127.8977 | 19971.2215 |
| 3 | 943 | 1473 | 89 | tolerancia | 69754.9064 | 107047.4924 |
| 5 | 943 | 1349 | — | **SistemaSingularError** (fila 1344, 5 observaciones) | — | — |

Top-10 usuario 1, k=2:

| película | r̂ |
|---|---:|
| Country Life (1994) | 1187.64 |
| Men With Guns (1997) | 894.09 |
| Cérémonie, La (1995) | 253.76 |
| Foreign Student (1994) | 212.79 |
| S.F.W. (1994) | 186.10 |
| Men of Means (1998) | 144.29 |
| Slingshot, The (1993) | 111.07 |
| Visitors, The (Visiteurs, Les) (1993) | 90.81 |
| Love and Death on Long Island (1997) | 82.05 |
| Joy Luck Club, The (1993) | 75.69 |

Top-10 usuario 1, k=3:

| película | r̂ |
|---|---:|
| Fausto (1993) | 4655.08 |
| Designated Mourner, The (1997) | 661.60 |
| Savage Nights (Nuits fauves, Les) (1992) | 490.46 |
| It Takes Two (1995) | 311.62 |
| Prefontaine (1997) | 248.29 |
| Best Men (1997) | 210.54 |
| Rendezvous in Paris (Rendez-vous de Paris, Les) (1995) | 200.24 |
| Underneath, The (1995) | 124.46 |
| Boys Life (1995) | 82.88 |
| Grateful Dead (1995) | 60.71 |

**GD con k=2, eta=2e-4/5e-4/1e-3 (epsilon=1, max_iter=3000):**

| eta | iteraciones | motivo de corte | SCE final | iteraciones con f creciente | resultado |
|---:|---:|---|---:|---:|---|
| 2e-4 | 1002 | tolerancia | 75251.9766 | 0 | — |
| 5e-4 | 3000 | max_iter | 93681.1269 | 1491 | — |
| 1e-3 | — | — | — | — | **DivergenciaError** en la iteración 14 |

### Corrida "experimento de umbral" (umbral independiente de k)

ALS: epsilon=1, max_iter=1000. GD: eta=2e-4, epsilon=1, max_iter=3000.

| umbral | k | usuarios | películas | calificaciones | ALS iter | ALS motivo | ALS SCE | ALS max\|r̂\| fuera Ω | GD iter | GD motivo | GD SCE | GD max\|r̂\| fuera Ω |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---:|---:|
| 20 | 2 | 917 | 937 | 94443 | 13 | tolerancia | 70970.26 | 8.8834 | 956 | tolerancia | 71247.59 | 6.7561 |
| 20 | 3 | 917 | 937 | 94443 | 27 | tolerancia | 66362.70 | 10.3223 | 1261 | tolerancia | 67086.13 | 6.5461 |
| 20 | 5 | 917 | 937 | 94443 | 70 | tolerancia | 59013.85 | 203.8911 | 1789 | tolerancia | 60491.63 | 9.6413 |
| 50 | 2 | 513 | 560 | 69222 | 11 | tolerancia | 50893.56 | 5.8295 | 711 | tolerancia | 51015.37 | 5.8523 |
| 50 | 3 | 513 | 560 | 69222 | 25 | tolerancia | 47940.72 | 6.4760 | 1143 | tolerancia | 48408.32 | 6.1028 |
| 50 | 5 | 513 | 560 | 69222 | 56 | tolerancia | 43561.83 | 11.9232 | 1439 | tolerancia | 44447.87 | 6.9047 |
| 100 | 2 | — | — | — | — | **ErrorDatosInsuficientes** | — | — | — | **ErrorDatosInsuficientes** | — | — |
| 100 | 3 | — | — | — | — | **ErrorDatosInsuficientes** | — | — | — | **ErrorDatosInsuficientes** | — | — |
| 100 | 5 | — | — | — | — | **ErrorDatosInsuficientes** | — | — | — | **ErrorDatosInsuficientes** | — | — |

(umbral=100 vacía toda la matriz en la cascada del filtro: no queda ningún
usuario ni película con al menos 100 calificaciones.)

Combinación elegida (menor umbral, luego menor k, con max\|r̂\| fuera de Ω
< 7 para ALS y GD a la vez): **umbral=50, k=2**.

Top-10 ALS, usuario 1 (umbral=50, k=2):

| película | r̂ |
|---|---:|
| Close Shave, A (1995) | 4.90 |
| Casablanca (1942) | 4.86 |
| Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) | 4.85 |
| Rear Window (1954) | 4.77 |
| Chinatown (1974) | 4.76 |
| Secrets & Lies (1996) | 4.76 |
| Third Man, The (1949) | 4.75 |
| One Flew Over the Cuckoo's Nest (1975) | 4.71 |
| Ran (1985) | 4.71 |
| Lawrence of Arabia (1962) | 4.69 |

Top-10 GD, usuario 1 (umbral=50, k=2):

| película | r̂ |
|---|---:|
| Close Shave, A (1995) | 4.89 |
| Casablanca (1942) | 4.87 |
| Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963) | 4.85 |
| Rear Window (1954) | 4.77 |
| Third Man, The (1949) | 4.74 |
| Chinatown (1974) | 4.72 |
| Secrets & Lies (1996) | 4.71 |
| One Flew Over the Cuckoo's Nest (1975) | 4.69 |
| Lawrence of Arabia (1962) | 4.69 |
| Manchurian Candidate, The (1962) | 4.68 |

### Conclusión

- **Tener al menos k observaciones por fila es necesario, pero no
  suficiente.** Con el diseño original (`umbral == k`), incluso con k=10
  (bien por encima del mínimo teórico) las predicciones de ALS fuera de Ω
  explotan (~1.7×10⁸ en el peor caso, top-10 del usuario 1 en el orden de
  los miles); con k=5 aparece directamente `SistemaSingularError`. La
  correlación entre la norma de una fila de V y la cantidad de
  calificaciones de esa película es prácticamente nula (-0.11): no es que
  "más calificaciones" arregle la norma, es que estar cerca del mínimo dejaba
  el sistema mal condicionado.
- **Separar `umbral` de `k` y subir el umbral bien por encima de k lo
  arregla.** Con `umbral=50` y `k=2` (bastante margen sobre el mínimo),
  max\|r̂\| fuera de Ω baja de miles/millones a ~5.8 para ALS y GD, en el
  rango real de las calificaciones (1 a 5). `umbral=20` todavía da >7 para
  algún k; `umbral=100` es demasiado (vacía la matriz).
- **`eta=2e-4` es el mejor de los tres probados para GD con k=2:** `5e-4`
  no converge en 3000 iteraciones (f sube 1491 veces) y `1e-3` diverge
  (`DivergenciaError`) en la iteración 14.
- **Valores definitivos** (`src/config.py`): `UMBRAL_DEFECTO=50`,
  `K_DEFECTO=2`, `ETA_DEFECTO=2e-4`, `EPSILON_DEFECTO=1.0`,
  `MAX_ITER_DEFECTO=3000`, `SEMILLA_DEFECTO=42`,
  `ESCALA_INICIALIZACION_DEFECTO=1.0`.

