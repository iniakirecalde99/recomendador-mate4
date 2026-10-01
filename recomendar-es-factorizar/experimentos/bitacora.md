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

## Recalibración sobre MovieLens latest-small

Mismo experimento que la calibración sobre 100K, repetido tras el cambio de
dataset. Scripts temporales fuera de `src/` y `tests/` (sin
commitear), sobre `data/ml-latest-small/` real. Semilla 42, escala de
inicialización 1.0. ALS: epsilon=1, max_iter=1000. GD: eta=2e-4, epsilon=1,
max_iter=3000. Criterio, igual que antes: menor umbral y después menor k con
max|r̂| fuera de Ω < 7 para ALS y GD a la vez. Como latest-small es mucho más
disperso (9.724 películas para 610 usuarios), se probaron umbrales más bajos
que en 100K, y después se afinó entre 30 y 50.

Nota: la primera corrida se cortó con `_ArrayMemoryError` (2,7 MiB) porque
otro proceso de la máquina tenía tomada casi toda la memoria virtual de
Windows; no es un problema del código. Se repitió por partes.

| umbral | k | usuarios | películas | calificaciones | ALS iter | ALS SCE | ALS max\|r̂\| fuera Ω | GD iter | GD SCE | GD max\|r̂\| fuera Ω | año mediano, % ≥ 2000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 2 | 609 | 2269 | 81109 | 27 | 46992.48 | 37.6676 | 1416 | 47539.44 | 7.1587 | 1997, 42 % |
| 10 | 3 | 609 | 2269 | 81109 | 78 | 42382.89 | 5006.0691 | 1926 | 43642.62 | 14.2253 | 1997, 42 % |
| 10 | 5 | 609 | 2269 | 81109 | 136 | 35272.49 | 268700.7511 | 2284 | 37293.07 | 11.1774 | 1997, 42 % |
| 20 | 2 | 566 | 1286 | 67020 | 21 | 39522.42 | 32.0125 | 1491 | 39957.39 | 6.3205 | 1997, 40 % |
| 20 | 3 | 566 | 1286 | 67020 | 38 | 35974.15 | 367.9705 | 1644 | 37045.56 | 10.4327 | 1997, 40 % |
| 20 | 5 | 566 | 1286 | 67020 | 96 | 30622.31 | 989.5806 | 1836 | 32181.10 | 9.3268 | 1997, 40 % |
| 30 | 2 | 435 | 827 | 52971 | 14 | 30992.45 | 8.5325 | 1416 | 31786.46 | 8.9711 | 1997, 40 % |
| 30 | 3 | 435 | 827 | 52971 | 47 | 28467.82 | 58.2797 | 1300 | 28996.37 | 7.1169 | 1997, 40 % |
| 30 | 5 | 435 | 827 | 52971 | 52 | 24651.24 | 138.7910 | 1482 | 25764.74 | 9.2326 | 1997, 40 % |
| 35 | 2 | 382 | 693 | 47024 | 13 | 27601.18 | 7.1834 | 1027 | 27953.26 | 6.4272 | 1997, 37 % |
| **40** | **2** | **321** | **534** | **38891** | **10** | **22937.39** | **6.3881** | **922** | **23191.62** | **5.8621** | 1997, 36 % |
| 50 | 2 | 207 | 302 | 23434 | 9 | 13367.34 | 5.4628 | 717 | 13639.46 | 5.4622 | 1997, 37 % |
| 50 | 3 | 207 | 302 | 23434 | 17 | 12293.65 | 17.5185 | 746 | 12965.39 | 7.3155 | 1997, 37 % |
| 50 | 5 | 207 | 302 | 23434 | 51 | 10627.39 | 22.8481 | 931 | 11407.25 | 6.5442 | 1997, 37 % |

Todas las corridas cortaron por tolerancia. Todas las combinaciones con k=3
o k=5 fallan el criterio (ALS amplifica |r̂| fuera de Ω); con k=2, umbral=40
es el menor que cumple.

**GD con umbral=40, k=2, eta=2e-4/5e-4/1e-3 (epsilon=1, max_iter=3000):**

| eta | iteraciones | motivo de corte | SCE final | iteraciones con f creciente | max\|r̂\| fuera Ω |
|---:|---:|---|---:|---:|---:|
| 2e-4 | 922 | tolerancia | 23191.62 | 0 | 5.8621 |
| 5e-4 | 462 | tolerancia | 23045.88 | 0 | 6.1228 |
| 1e-3 | — | — | — | — | **DivergenciaError** en la iteración 35 |

**Robustez de eta=5e-4 frente a la semilla (umbral=40, k=2):**

| semilla | GD iteraciones | f creciente | GD max\|r̂\| fuera Ω | ALS max\|r̂\| fuera Ω |
|---:|---:|---:|---:|---:|
| 1 | 511 | 0 | 7.24 | 6.46 |
| 2 | 755 | 0 | 6.57 | 6.15 |
| 3 | 554 | 0 | 5.65 | 6.23 |
| 7 | 432 | 0 | 5.92 | 6.16 |
| 123 | 430 | 0 | 6.07 | 6.15 |

(Con la semilla 1, GD queda apenas por encima de 7; el criterio se evalúa
con la semilla por defecto, 42.)

**Demo con los defaults nuevos (umbral=40, k=2, eta=5e-4):**

```
MovieLens filtrado (umbral=40, k=2): 321 usuarios, 534 películas, 38891 calificaciones.
ALS                 10        tolerancia      0.1198    22937.3886
GD                 462        tolerancia      2.0637    23045.8796
```

Top-10 ALS del usuario 1: Shawshank Redemption (5.25), Casablanca, Boondock
Saints (2000), Wallace & Gromit: The Wrong Trousers, Godfather, Dark Knight
(2008), Unforgiven, Amelie (2001), Life Is Beautiful, Lord of the Rings:
The Return of the King (2003). Sin regularización ni recorte, r̂ puede pasar
de 5, que es la calificación máxima.

### Conclusión

- **Valores definitivos** (`src/config.py`): `UMBRAL_DEFECTO=40`,
  `K_DEFECTO=2`, `ETA_DEFECTO=5e-4`; sin cambios `EPSILON_DEFECTO=1.0`,
  `MAX_ITER_DEFECTO=3000`, `SEMILLA_DEFECTO=42`,
  `ESCALA_INICIALIZACION_DEFECTO=1.0`. `tests/test_calibracion.py` verifica
  el criterio con esos defaults sobre el dataset real.
- **La misma lección que con 100K:** k=2 y un umbral bien por encima de k.
  Con más factores o menos observaciones, ALS amplifica r̂ fuera de Ω.
- **Películas nuevas:** el filtro se queda con las más calificadas, que en
  este dataset son mayormente de los 90 (año mediano 1997 en todos los
  umbrales; entre 36 % y 42 % del 2000 en adelante). Igual aparecen en las
  recomendaciones películas de 2000 a 2008, que con 100K (hasta 1998) no
  existían.

## Experimento de concentración de las recomendaciones con k = 2 (centrado descartado)

### Objetivo

Medir cuánto se concentran las recomendaciones con k = 2 (con el dataset
anterior se notaba que la página recomendaba Titanic casi siempre, aunque
se cambiaran las calificaciones) y probar si centrar las calificaciones
restando el promedio global μ lo corrige.

Criterio de adopción, fijado antes de correr: adoptar el centrado solo si la
cantidad de películas que aparecen en algún top-10 sube claramente respecto
de 65 y la más frecuente baja claramente del 46 %. Si la mejora es marginal,
se deja como limitación de k = 2 en la sección 8 del informe.

### Método

- Script: `experimentos/concentracion.py` (`python -m experimentos.concentracion`;
  parámetros como argumentos, con defaults de `config.py` y constantes del
  propio script). El centrado vive solo en ese script: `src/` no centra.
- Datos y modelo: MovieLens latest-small con los defaults de la recalibración (umbral=40,
  k=2, epsilon=1, semilla 42): 321 usuarios × 534 películas. ALS sobre R
  (sin centrado) o sobre R − μ (con centrado, μ = promedio de las
  calificaciones observadas en Ω = 3,679).
- Usuarios simulados: 3.000 (semilla de simulación 0). Cada uno califica
  entre 5 y 10 de las 30 películas más calificadas (las mismas que ofrece la
  página). Su vector u se resuelve con V fija (`resolver_factor`, informe 5),
  con sus notas − μ cuando hay centrado; la predicción es μ + Vⱼ·u. Se cuenta
  en qué top-10 (sin las ya calificadas) aparece cada película.
- Dos distribuciones de notas: uniforme (enteros 1 a 5 equiprobables,
  promedio 3) y real (notas de `ratings.csv` redondeadas hacia arriba a
  enteros, promedio 3,65). La variante real se agregó porque con notas
  uniformes todos los usuarios simulados quedan por debajo de μ y el
  centrado los trata como "exigentes", lo que sesga la comparación.

### Resultados

| notas | centrado | SCE ALS (iter) | max\|r̂\| fuera Ω | distintas en algún top-10 | más frecuente | frec. | películas en > 20 % de los top-10 | ángulo de u p5 / p50 / p95 |
|---|---|---:|---:|---:|---|---:|---:|---|
| uniforme | no | 22.937 (10) | 6,39 | 65 de 534 | Harry Potter and the Order of the Phoenix (2007) | 45,5 % | 19 | 6° / 44° / 89° |
| uniforme | sí (μ = 3,679) | 23.573 (8) | 6,03 | 77 de 534 | Batman & Robin (1997) | 52,4 % | 18 | −167° / −96° / 56° |
| real | no | 22.937 (10) | 6,39 | 65 de 534 | Harry Potter and the Order of the Phoenix (2007) | 44,5 % | 23 | 18° / 44° / 75° |
| real | sí (μ = 3,679) | 23.573 (8) | 6,03 | 80 de 534 | Nutty Professor, The (1996) | 38,4 % | 21 | −172° / 1° / 171° |

Top-5 más frecuentes de cada corrida:

- uniforme, sin centrado: Harry Potter and the Order of the Phoenix 46 %; 10 Things I Hate About You 43 %; The Patriot 42 %; The Chronicles of Narnia: The Lion, the Witch and the Wardrobe 38 %; Chinatown 33 %.
- uniforme, con centrado: Batman & Robin 52 %; The Flintstones 51 %; Wild Wild West 50 %; Fantasia 47 %; Mortal Kombat 47 %.
- real, sin centrado: Harry Potter and the Order of the Phoenix 45 %; 10 Things I Hate About You 37 %; The Patriot 34 %; How to Train Your Dragon 34 %; The Boondock Saints 33 %.
- real, con centrado: The Nutty Professor 38 %; Coneheads 38 %; The Maltese Falcon 35 %; Beverly Hills Cop III 35 %; Chinatown 35 %.

Referencia con el dataset anterior (MovieLens 100K, umbral=50, k=2, notas
uniformes, misma simulación): 63 de 560 películas en algún top-10; Titanic
en el 45,7 % de los top-10, sin estar entre las 30 para calificar (no se la
podía excluir calificándola).

### Decisión

**El centrado no se adopta.** Con la simulación pedida (notas uniformes), la
cantidad de películas distintas sube poco (65 → 77) y la más frecuente
empeora (45,5 % → 52,4 %). Con notas reales mejora algo (65 → 80; 44,5 % →
38,4 %), pero sigue siendo marginal: 80 de 534 películas es un 15 %, y 21
películas siguen apareciendo en el top-10 de más del 20 % de los usuarios.
Además, centrar empeora el ajuste: la SCE de ALS pasa de 22.937 a 23.573.

**La limitación es k = 2.** Con dos factores, cada película es un punto Vⱼ
del plano y la predicción r̂ⱼ = Vⱼ · u es su proyección sobre la dirección
de u, así que para cualquier u ganan las películas del borde exterior de la
nube de puntos. El centrado arregla otra causa, que u quede siempre en un
abanico angosto porque todas las notas son positivas (sin centrar, de 6° a
89°; centrado, casi toda la vuelta). Pero no cambia esa geometría: las
favoritas cambian, pero siguen siendo pocas. Subir k sin regularización no
es una salida (ver la recalibración: k = 3 y k = 5 hacen explotar r̂
fuera de Ω). Queda como limitación de k = 2 en la sección 8 del informe.
`src/` sigue sin centrado.

## Experimento 3: concentración con los usuarios reales (sobre main)

**Registrado antes de correr.** Usa el modelo de main sin cambios (no toca
`src/`).

### Pregunta

Con usuarios reales, ¿el modelo de main concentra las recomendaciones como
con usuarios simulados al azar?

### Método

- Modelo de main: ALS con k = 2 y los defaults de `config.py` (umbral=40,
  epsilon=1, max_iter=3000, semilla 42), entrenado sobre todo Ω (321 usuarios
  × 534 películas, 38.891 calificaciones).
- Para cada uno de los 321 usuarios, top-10 entre las películas que no
  calificó, con su fila de U entrenada: r̂ᵢⱼ = (U·Vᵀ)ᵢⱼ, de mayor a menor,
  empates por índice (como `recomendaciones.recomendar_top_n`).
- Se reportan: películas distintas en algún top-10, la más frecuente y el
  porcentaje de usuarios en cuyo top-10 aparece, el top-5 de las más
  frecuentes y cuántas aparecen en el top-10 de más del 20 % de los usuarios.
- Script: `experimentos/usuarios_reales.py` (con test en
  `tests/test_experimento_usuarios_reales.py`). Una sola corrida:
  `python -m experimentos.usuarios_reales`, con los defaults.

### Referencia

Experimento de concentración (usuarios simulados con notas reales, mismo
modelo): 65 de 534 películas en algún top-10; la más frecuente, Harry Potter
and the Order of the Phoenix (2007), en el 44,5 %.

No hay un criterio de decisión: es una medición descriptiva para comparar con
esa referencia. El resultado se registra tal como salga.

### Resultado

Corrida única de `python -m experimentos.usuarios_reales`. El modelo es el
de main: ALS en 10 iteraciones (tolerancia), SCE 22.937,3886.

| | usuarios simulados (referencia) | usuarios reales |
|---|---:|---:|
| películas en algún top-10 | 65 de 534 | **127 de 534** |
| la más frecuente | Harry Potter and the Order of the Phoenix (2007), 44,5 % | **Casablanca (1942), 43,9 %** |
| películas en el top-10 de más del 20 % de los usuarios | 23 | 20 |

Top-5 más frecuentes con usuarios reales: Casablanca (1942) 43,9 %; Boondock
Saints, The (2000) 43,3 %; Wallace & Gromit: The Wrong Trousers (1993)
37,4 %; Princess Bride, The (1987) 34,6 %; 12 Angry Men (1957) 32,4 %.

Lectura: con usuarios reales aparecen casi el doble de películas distintas
(127 contra 65), pero la concentración en la parte de arriba es la misma: la
más frecuente sigue en el top-10 de alrededor del 44 % de los usuarios, y unas
20 películas aparecen en el top-10 de más de uno de cada cinco. Una parte de
la diferencia en películas distintas puede venir de cómo se arman los dos
grupos: cada usuario real calificó al menos 40 películas, que quedan
excluidas de su top-10, mientras que los simulados califican solo entre 5 y
10 de las 30 más calificadas. Con usuarios reales, la concentración de main
se confirma: no es un artefacto de la simulación.

## Experimento 4: ordenar por la estimación relativa (sobre main)

**Registrado antes de correr.** Usa el modelo de main sin cambios (no toca
`src/`).

### Pregunta

¿Ordenar por la estimación relativa reduce la concentración con usuarios
reales sin recomendar películas de estimación baja?

### Método

- Modelo: el mismo del experimento 3 (ALS de main, k = 2, defaults de
  `config.py`, entrenado sobre todo Ω: 321 × 534).
- ū = promedio de las filas de U entrenada.
- Para cada usuario real, top-10 entre las películas que no calificó, con dos
  ordenamientos:
  - absoluto: r̂ᵢⱼ = Vⱼ · Uᵢ (el del experimento 3);
  - relativo: Vⱼ · (Uᵢ − ū), que es r̂ᵢⱼ menos el promedio de r̂ de la
    película j sobre todos los usuarios (lo verifica
    `test_puntaje_relativo_es_r_hat_menos_el_promedio_de_la_pelicula_sobre_los_usuarios`).
  En los dos, orden de mayor a menor con empates por índice.
- Para cada ordenamiento se reporta: películas distintas en algún top-10, la
  más frecuente y su porcentaje, películas en el top-10 de más del 20 % de
  los usuarios y el promedio de r̂ᵢⱼ absoluto de las películas recomendadas.
- Referencia, fuera del criterio: los mismos números con usuarios simulados
  con notas reales (3.000, el mismo sorteo que `experimentos/concentracion.py`),
  que es el caso del front. Para ellos, el relativo es V · (u − ū), con el ū
  de los usuarios reales.
- Script: `experimentos/ordenamiento_relativo.py` (con tests en
  `tests/test_experimento_ordenamiento_relativo.py`; uno verifica que la
  simulación con el ordenamiento absoluto da lo mismo que
  `concentracion.simular`). Una sola corrida:
  `python -m experimentos.ordenamiento_relativo`, con los defaults.

### Criterio (fijado antes de correr; se evalúa sobre usuarios reales)

Se cumple si se cumplen los tres, para el ordenamiento relativo:

1. La más frecuente aparece en a lo sumo el 25 % de los top-10.
2. Películas distintas en algún top-10: al menos 127 (no empeora respecto del
   absoluto del experimento 3).
3. El promedio de r̂ᵢⱼ absoluto de lo recomendado baja a lo sumo 0,5 respecto
   del ordenamiento absoluto.

### Resultado

(Pendiente: se completa con la salida de la corrida única.)
