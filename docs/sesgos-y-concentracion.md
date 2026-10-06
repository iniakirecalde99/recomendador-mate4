# Por qué el modelo con sesgos reduce la concentración de las recomendaciones

Resumen de los experimentos registrados en `experimentos/bitacora.md`, sobre MovieLens
latest-small filtrado (321 usuarios × 534 películas). La concentración se mide
con el top-10 de cada usuario: cuántas películas distintas aparecen en algún
top-10 y en qué porcentaje de los top-10 aparece la más frecuente.

**En una frase:** la concentración bajó cuando el modelo separó *lo que le
gusta a cada persona* de *qué tan buena es una película en general*. Antes las
dos cosas estaban mezcladas en el mismo término, y lo "bueno en general"
ganaba siempre.

## Por qué se repetía antes (`main`: k = 2, sin sesgos)

El modelo de `main` predice con un solo término:

    r̂ᵢⱼ = Uᵢ · Vⱼ

Ese producto tiene que explicar todo a la vez:

- que el promedio de las notas ronda 3,7;
- que hay usuarios generosos y usuarios exigentes;
- que hay películas que casi todos califican alto (Casablanca, Shawshank);
- y recién al final, los gustos personales.

Con k = 2 hay solo dos números por película y dos por usuario para todo eso.
Lo que más pesa en el error es lo común (las notas son altas y algunas
películas son mejores para casi todos), así que los vectores terminan usando
casi toda su capacidad en ese patrón general. Queda muy poco espacio para el
gusto individual.

### La explicación geométrica

- Con k = 2, cada película es un punto en un plano y cada usuario es una
  dirección. El top-10 de un usuario son las películas más alejadas en su
  dirección: las del borde de la nube de puntos.
- Como todas las notas son positivas, las direcciones de los usuarios son casi
  iguales: con usuarios simulados, entre 6° y 89°.
- Entonces casi todos miran hacia el mismo borde, donde están las películas
  "buenas para todos", y ganan siempre las mismas. Con los usuarios reales,
  Casablanca aparecía en el **43,9 %** de los top-10 y solo **127** películas
  aparecían alguna vez.

### Lo que se probó antes y por qué no alcanzó

| Intento | Qué cambió | Por qué no alcanzó |
|---|---|---|
| Centrar por μ global | Restó el promedio de todas las notas | Sacó el nivel general, pero no la calidad de cada película |
| Regularización con k > 2 | Frenó los valores extremos | La calidad general seguía dentro de V: Shawshank en el 69–73 % de los top-10 |
| Ordenamiento relativo | Restó el promedio de cada película | Con k = 2 cambió las favoritas, no la concentración (41,1 %) |

## Por qué los sesgos lo resolvieron

El modelo con sesgos le da a cada efecto su propio lugar:

    r̂ᵢⱼ = μ + bᵢ + cⱼ + Uᵢ · Vⱼ

| Término | Qué captura |
|---|---|
| μ | El promedio de todas las notas |
| bᵢ | Si el usuario i califica más alto o más bajo que el promedio |
| cⱼ | Qué tan bien califica todo el mundo a la película j: su popularidad o calidad general |
| Uᵢ · Vⱼ | Lo que queda: cuánto le gusta la película j al usuario i más allá de lo esperable |

Como la calidad general ahora va en cⱼ, U y V ya no tienen que cargarla y
pueden dedicarse a diferencias de gusto: a quién le gusta el terror, a quién el
cine de autor, y así.

### Los dos ordenamientos

- **A, por r̂ completo:** incluye cⱼ, así que sigue favoreciendo a las
  películas que todos valoran bien. Casablanca sigue arriba, aunque menos:
  **34,0 %** de los top-10, con 261 películas distintas.
- **B, solo por Uᵢ · Vⱼ:** saca cⱼ y ordena únicamente por afinidad personal.
  Cada usuario recibe lo que le gusta a él más que al resto. Por eso se reparte:
  **433** películas distintas, y la más frecuente (Borat) aparece solo en el
  **13,1 %** de los top-10.

| Modelo (usuarios reales) | Películas en algún top-10 | La más frecuente |
|---|---:|---|
| `main` (k = 2, sin sesgos) | 127 de 534 | Casablanca, 43,9 % |
| Sesgos, ordenamiento A | 261 de 534 | Casablanca, 34,0 % |
| Sesgos, ordenamiento B | 433 de 534 | Borat, 13,1 % |

## Dos aclaraciones importantes

### 1. Los sesgos solos no alcanzan

El experimento 6 lo mostró: con sesgos pero con k = 2 y sin regularización
(las reglas de `main`), una sola película (Deadpool) dominó el **43,9 %** de los
top-10 con el ordenamiento B. El resultado bueno necesitó tres cosas juntas:

- **los sesgos**, para separar popularidad de gusto;
- **k = 20**, para tener suficientes dimensiones de gusto;
- **la regularización λ = 10**, para que las películas con pocas calificaciones
  no tengan factores exagerados.

### 2. Hay una contraprestación entre variedad y acierto

| | Precisión@10 |
|---|---:|
| `main` (línea de base) | 0,1116 |
| Sesgos, ordenamiento A | 0,1530 |
| Sesgos, ordenamiento B | 0,1122 |

B reparte mucho más, pero acierta apenas igual que `main` (0,1122 contra
0,1116, una diferencia mínima medida en una sola partición). A acierta bastante
más porque recomendar películas buenas para todos es una apuesta segura, aunque
sea repetitiva. Es la tensión clásica de los recomendadores entre recomendar lo
popular, que acierta, y recomendar lo personal, que es más variado.

**Una analogía:** A te recomienda "el mejor restaurante de la ciudad", que le
va a gustar a casi todos, y por eso se lo recomienda a casi todos. B te
recomienda "el que más te va a gustar a vos comparado con lo que le gusta a la
gente en general", y eso cambia mucho de persona a persona.

---

Los números salen de `experimentos/bitacora.md`: experimentos de concentración, 3, 4
y 6, el experimento de regularización (experimento 2) y el de sesgos. Nada de las ramas experimentales entra al informe sin aprobación de la
cátedra.
