# %% [markdown]
# # Simulated Annealing per il TSP
# 
# ### Problema: Travelling Salesman Problem (TSP)
#
# Il **Travelling Salesman Problem (TSP)** è uno dei problemi NP-hard più famosi:
# dato un insieme di città con distanze note, trovare il percorso più breve che
# visiti ogni città esattamente una volta e ritorni alla città di partenza.



# %%
import numpy as np
import matplotlib.pyplot as plt
import random
import math


# 15 città in posizioni 2D casuali
np.random.seed(42)
random.seed(42)
N_CITIES = 15
city_positions = np.random.rand(N_CITIES, 2)

def draw_tour(positions, tour=None, title="Città", color='#4f98a3'):
    """Visualizza le città e, se fornito, il percorso con frecce direzionali."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(positions[:, 0], positions[:, 1],
               s=300, c=color, zorder=5, edgecolors='white', linewidths=2)

    for city_idx, (x, y) in enumerate(positions):
        ax.annotate(str(city_idx), (x, y), ha='center', va='center',
                    fontweight='bold', color='white', fontsize=10, zorder=6)

    if tour is not None:
        tour_closed = tour + [tour[0]]
        for step in range(len(tour)):
            city_from = tour_closed[step]
            city_to   = tour_closed[step + 1]
            ax.annotate("",
                        xy=positions[city_to], xytext=positions[city_from],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


draw_tour(city_positions)
#%%

# Tour casuale iniziale per visualizzazione
initial_tour = list(range(N_CITIES))
random.shuffle(initial_tour)


draw_tour(city_positions, initial_tour,
          title=f"Tour casuale")




# %% [markdown]
# # Cost function
#
# Un **tour** è una sequenza ordinata di $N$ città: $\sigma = (\sigma_0, \sigma_1, \dots, \sigma_{N-1})$.
# Ad esempio con 4 città: $\sigma = (0, 2, 3, 1)$ significa "visita 0 → 2 → 3 → 1 → torna a 0".
#
# La distanza tra due città $i$ e $j$ è la distanza euclidea tra le loro posizioni:
#
# $$d_{i,i} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
#
# Il costo del tour è la somma di tutte le distanze percorse, incluso il ritorno alla città di partenza:
#
# $$C(\sigma) = \sum_{k=0}^{N-1} d_{\sigma_k,\; \sigma_{(k+1) \bmod N}}$$
#
# L'obiettivo è trovare il tour $\sigma^*$ che minimizza questo costo:
#
# $$\sigma^* = \arg\min_\sigma \; C(\sigma)$$
# ## Soluzione esatta

# La soluzione esatta può essere trovata con una ricerca esaustiva su tutte le permutazioni di città, 
# ma è impraticabile per $N$ grandi:
# Il numero di tour possibili cresce fattorialmente con $N$: ci sono $\frac{(N-1)!}{2}$ tour unici,
# quindi l'ordine del metodo è $O(N!)$

#%%

N_CITIES=15
print(f"Numero di città: {N_CITIES}")
print(f"Numero di tour possibili (unici) [(N-1)!/2]: {math.factorial(N_CITIES - 1) // 2:,}  ")

# %%
# COST FUNCTION: distanza totale del tour
def tsp_cost_function(tour, dist_matrix):
    """Calcola la distanza totale di un tour (incluso il ritorno alla città iniziale)."""
    n_cities_local = len(tour)
    total_distance = sum(
        dist_matrix[tour[i]][tour[(i + 1) % n_cities_local]]
        for i in range(n_cities_local)
    )
    return total_distance


def compute_distance_matrix(positions):
    """Calcola la matrice delle distanze euclidee tra tutte le coppie di città."""
    n = len(positions)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(positions[i] - positions[j])
    return dist_matrix


dist_matrix = compute_distance_matrix(city_positions)

tour = list(range(N_CITIES))
random.shuffle(tour)

costo_tour=tsp_cost_function(tour, dist_matrix)

draw_tour(city_positions, tour,
          title=f"Tour casuale, distanza : {costo_tour:.2f}")


# %% [markdown]
# ---
# ## Parte 2 — Simulated Annealing
#
# ### Soluzione Euristica?
#
# La soluzione esatta del TSP richiederebbe di valutare tutte le permutazioni possibili,
# con una complessità di $O(N!)$ — o al meglio $O(N^2 2^N)$ con l'algoritmo di Held-Karp.
#
# Si ricorre quindi a una **metaeuristica**: un algoritmo che non garantisce la soluzione
# ottima, ma trova soluzioni di buona qualità in tempo ragionevole. Il **Simulated Annealing**
# è una delle metaeuristiche più comuni strutturata in modo tale da  **uscire dai minimi locali** durante la ricerca.
#
# ### Simulated Annealing — l'idea intuitiva
#
# Immagina di essere perso in una catena montuosa e di voler raggiungere la valle
# più profonda (= il tour più corto). Un algoritmo "greedy" puro scenderebbe sempre
# verso il basso, ma si fermerebbe al primo avvallamento locale — non necessariamente
# la valle più profonda.
#
# Il **Simulated Annealing** risolve questo problema ispirandosi al processo fisico
# del *raffreddamento lento dei metalli*: quando un metallo fuso viene raffreddato
# gradualmente, gli atomi hanno il tempo di trovare la configurazione di minima energia
# globale invece di bloccarsi in stati disordinati locali.
#
# In pratica, l'algoritmo:
# 1. **Propone** una modifica casuale al tour corrente (es. inverte un segmento)
# 2. **Calcola** la variazione di costo `delta = costo_nuovo − costo_attuale`
# 3. **Decide** se accettare la modifica secondo la regola di Metropolis:
#    - se `delta < 0` (tour migliorato) → **accetta sempre**
#    - se `delta > 0` (tour peggiorato) → **accetta con probabilità** `exp(−delta / T)`
#
# Il parametro **T** è la *temperatura*: parte alta (esplorazione libera) e scende
# gradualmente verso zero (selezione sempre più severa).
#
# Con T alta, `exp(−delta / T)` è vicino a 1 → quasi ogni mossa viene accettata,
# anche le peggiori. L'algoritmo esplora liberamente l'intero spazio.
#
# Con T bassa, `exp(−delta / T)` → 0 → solo i miglioramenti vengono accettati.
# L'algoritmo converge verso un minimo locale.
#
# Questo meccanismo permette di **uscire dai minimi locali** nelle fasi iniziali,
# per poi stabilizzarsi sulla soluzione migliore trovata man mano che T scende.
#


# %% [markdown]
# ### Mossa 2-opt — come si genera un tour vicino
#
# Il TSP lavora su **permutazioni**: l'ordine in cui si visitano le città.
# Per "muoversi" nello spazio delle soluzioni bisogna modificare il tour corrente
# in modo strutturato. La mossa 2-opt è il metodo più semplice e diffuso.
#
# **Idea**: scegli due posizioni `i` e `j` nel tour, poi **inverti il segmento**
# compreso tra di esse.
#
# ```
# Tour originale:  [0]→[1]→[2]→[3]→[4]→[5]→(ritorno a 0)
#                            ↑_______________↑
#                            i=2             j=5   ← scelti a caso
#
# Dopo il 2-opt:   [0]→[1]→[5]→[4]→[3]→[2]→(ritorno a 0)
#                            (segmento invertito)
# ```
#
# In pratica rompe due archi e li ricollega in modo diverso.
# Se i due archi originali si **incrociavano** sulla mappa, la versione invertita
# non si incrocia — e due archi non incrociati sono sempre più corti
# (per la disuguaglianza triangolare).
#
# ### Come si integra nel SA
#
# Nel SA, la mossa 2-opt è la **proposta**: genera il tour candidato da valutare
# ad ogni iterazione. Lo schema di ogni step è:
#
# ```
# 1. tour_candidato = propose_2opt(tour_corrente)   ← genera un vicino casuale
# 2. delta = costo(tour_candidato) − costo(tour_corrente)
# 3. accetta o rifiuta con il criterio di Metropolis
# ```
#
# Ogni iterazione fa **una sola mossa 2-opt scelta a caso** — non la migliore
# possibile. Questo è intenzionale: cercare sempre la mossa migliore produrrebbe
# un algoritmo deterministico che converge rapidamente a un minimo locale senza
# mai riuscire a uscirne. La casualità + il criterio Metropolis sono ciò che
# permettono l'esplorazione globale.

# %%


def propose_2opt(tour):
    """Proposta 2-opt classica: inverte un segmento casuale del tour."""
    n = len(tour)
    pos_i, pos_j = sorted(random.sample(range(n), 2))
    new_tour = tour[:pos_i] + tour[pos_i:pos_j + 1][::-1] + tour[pos_j + 1:]
    return new_tour

def simulated_annealing_classic(dist_matrix, n_iter=1000, T_start=2.0, T_end=0.001):
    """
    SA standard con mossa 2-opt come proposta locale.
    Minimizza la distanza totale del tour (accetta peggioramenti con criterio Metropolis).
    """
    n_cities_local = dist_matrix.shape[0]
    current_tour = list(range(n_cities_local))
    random.shuffle(current_tour)
    current_cost = tsp_cost_function(current_tour, dist_matrix)
    best_tour, best_cost = current_tour[:], current_cost
    history_best = []
    history_current = []

    for iteration in range(n_iter):
        # Schedule di temperatura esponenziale
        temperature = T_start * (T_end / T_start) ** (iteration / n_iter)

        # Proposta: 2-opt casuale
        neighbor_tour = propose_2opt(current_tour)
        neighbor_cost = tsp_cost_function(neighbor_tour, dist_matrix)
        delta = neighbor_cost - current_cost

        # Metropolis per minimizzazione: accetta se delta < 0 (miglioramento)
        # oppure con probabilità exp(-delta/T) se delta > 0 (peggioramento)
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_tour, current_cost = neighbor_tour, neighbor_cost

        if current_cost < best_cost:
            best_tour, best_cost = current_tour[:], current_cost

        history_best.append(best_cost)
        history_current.append(current_cost)

    return best_tour, best_cost, history_best, history_current

# %%
best_tour_classic, best_cost_classic, history_classic, history_current_classic = simulated_annealing_classic(
    dist_matrix, n_iter=500, T_start=10.0, T_end=0.01
)

print(f"Miglior tour classico: {best_tour_classic}")
print(f"Miglior distanza (SA classico): {best_cost_classic:.3f}")

draw_tour(city_positions, best_tour_classic,
          title=f"SA Classico — distanza: {best_cost_classic:.3f}", color='#7a7974')

plt.figure(figsize=(8, 4))
plt.plot(history_current_classic, color='#b0a99f', linewidth=1.0, label='Soluzione corrente')
plt.plot(history_classic, color='#7a7974', linewidth=1.5, label='Miglior soluzione')
plt.xlabel('Iterazione')
plt.ylabel('Distanza TSP')
plt.title('Convergenza SA Classico')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# %% [markdown]
# ---
# ## Parte 3 — Video dell'ottimizzazione
#
# Registra l'andamento del SA in un video: a sinistra il tour corrente (best so far),
# a destra la curva di convergenza con un cursore che avanza iterazione per iterazione.

# %%
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def simulated_annealing_with_snapshots(
    dist_matrix, n_iter=500, T_start=2.0, T_end=0.001, snapshot_every=10
):
    """
    SA classico con 2-opt che salva snapshot del best tour a intervalli regolari.
    Ritorna best_tour, best_cost, history_best, history_current e la lista di snapshot per l'animazione.
    """
    n_cities_local = dist_matrix.shape[0]
    current_tour = list(range(n_cities_local))
    random.shuffle(current_tour)
    current_cost = tsp_cost_function(current_tour, dist_matrix)
    best_tour, best_cost = current_tour[:], current_cost
    history_best = []
    history_current = []
    snapshots = []

    for iteration in range(n_iter):
        temperature = T_start * (T_end / T_start) ** (iteration / n_iter)

        neighbor_tour = propose_2opt(current_tour)
        neighbor_cost = tsp_cost_function(neighbor_tour, dist_matrix)
        delta = neighbor_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_tour, current_cost = neighbor_tour, neighbor_cost

        if current_cost < best_cost:
            best_tour, best_cost = current_tour[:], current_cost

        history_best.append(best_cost)
        history_current.append(current_cost)

        if iteration % snapshot_every == 0 or iteration == n_iter - 1:
            snapshots.append({
                'iteration': iteration,
                'current_tour': current_tour[:],
                'current_cost': current_cost,
                'best_tour': best_tour[:],
                'best_cost': best_cost,
                'temperature': temperature,
                'history_best': history_best[:],
                'history_current': history_current[:],
            })

    return best_tour, best_cost, history_best, history_current, snapshots


def create_optimization_video(
    city_positions, snapshots, output_path='sa_optimization.mp4', fps=15
):
    """
    Crea un video con due subplot animati:
    - sinistra: tour best-so-far con frecce
    - destra:   curva di convergenza con cursore verticale
    """
    n_iter_total = snapshots[-1]['iteration'] + 1
    cost_min = min(s['best_cost'] for s in snapshots) * 0.95
    cost_max = max(s['current_cost'] for s in snapshots) * 1.05

    fig, (ax_tour, ax_conv) = plt.subplots(1, 2, figsize=(14, 6))

    current_color = '#aaaaaa'
    best_color = '#e05c5c'
    cursor_color = '#e05c5c'

    def draw_tour_arrows(ax, tour, color, lw):
        tour_closed = tour + [tour[0]]
        for step in range(len(tour)):
            city_from = tour_closed[step]
            city_to   = tour_closed[step + 1]
            ax.annotate(
                "",
                xy=city_positions[city_to], xytext=city_positions[city_from],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw)
            )

    def draw_frame(frame_idx):
        snap = snapshots[frame_idx]

        # ── Tour ──────────────────────────────────────────────────────────────
        ax_tour.clear()

        # Current tour in grey (background)
        draw_tour_arrows(ax_tour, snap['current_tour'], color=current_color, lw=1.5)

        # Best tour in red (foreground)
        draw_tour_arrows(ax_tour, snap['best_tour'], color=best_color, lw=2.0)

        # Cities on top
        ax_tour.scatter(
            city_positions[:, 0], city_positions[:, 1],
            s=300, c='#4f98a3', zorder=5, edgecolors='white', linewidths=2
        )
        for city_idx, (x, y) in enumerate(city_positions):
            ax_tour.annotate(
                str(city_idx), (x, y), ha='center', va='center',
                fontweight='bold', color='white', fontsize=10, zorder=6
            )

        # Dummy lines for legend
        from matplotlib.lines import Line2D
        ax_tour.legend(
            handles=[
                Line2D([0], [0], color=current_color, lw=1.5, label=f'Corrente: {snap["current_cost"]:.3f}'),
                Line2D([0], [0], color=best_color,    lw=2.0, label=f'Best: {snap["best_cost"]:.3f}'),
            ],
            fontsize=9, loc='upper right'
        )

        ax_tour.set_title(f"iter {snap['iteration']}", fontsize=11)
        ax_tour.set_xlim(-0.05, 1.05)
        ax_tour.set_ylim(-0.05, 1.05)
        ax_tour.grid(True, alpha=0.2)

        # ── Convergence ───────────────────────────────────────────────────────
        ax_conv.clear()
        ax_conv.plot(snap['history_current'], color=current_color, linewidth=1.0, label='Soluzione corrente')
        ax_conv.plot(snap['history_best'], color=best_color, linewidth=1.5, label='Miglior soluzione')
        ax_conv.axvline(
            x=snap['iteration'], color=cursor_color,
            linestyle='--', alpha=0.8, linewidth=1.5
        )
        ax_conv.set_xlim(0, n_iter_total)
        ax_conv.set_ylim(cost_min, cost_max)
        ax_conv.set_xlabel('Iterazione')
        ax_conv.set_ylabel('Distanza TSP')
        ax_conv.set_title(f"Convergenza SA  |  T = {snap['temperature']:.4f}", fontsize=11)
        ax_conv.legend(fontsize=9)
        ax_conv.grid(True, alpha=0.3)

        fig.tight_layout()

    anim = FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=1000 // fps)

    # Prova FFMpeg (mp4), fallback su Pillow (gif)
    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'SA TSP Optimization'})
        anim.save(output_path, writer=writer, dpi=120)
        print(f"Video salvato: {output_path}")
    except Exception as ffmpeg_error:
        gif_path = output_path.replace('.mp4', '.gif')
        print(f"FFMpeg non disponibile ({ffmpeg_error}), salvo come GIF: {gif_path}")
        writer = PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer, dpi=100)
        print(f"GIF salvata: {gif_path}")

    plt.close(fig)
    return anim


# %%
print("Esecuzione SA con raccolta snapshot...")
best_tour_video, best_cost_video, history_best_video, history_current_video, snapshots_video = (
    simulated_annealing_with_snapshots(
        dist_matrix, n_iter=600, T_start=5.0, T_end=0.01, snapshot_every=10
    )
)
print(f"Snapshot raccolti: {len(snapshots_video)}")
print(f"Miglior distanza: {best_cost_video:.3f}")

create_optimization_video(
    city_positions, snapshots_video,
    output_path='sa_optimization.mp4', fps=2
)

# %% [markdown]
# ---
# ## Parte 4 — Scalabilità: tempo di esecuzione vs numero di città
#
# Fissiamo `n_iter` e misuriamo quanto impiega un singolo run al variare di N città.
# Per ogni valore di N ripetiamo il run `N_TIMING_RUNS` volte (città casuali diversi)
# per stabilizzare la stima ed avere una banda di varianza.
#
# ### Costo computazionale atteso
# - `tsp_cost`:     O(N)  per iterazione
# - `propose_2opt`: O(N)  (slice + reverse)
# - loop SA:        n_iter iterazioni
#
# → Tempo totale ∝ **O(N · n_iter)** — crescita **lineare** in N a parità di iterazioni.

# %%
import time

def tsp_cost_function_fast(positions):
    """Distanza euclidea vettorizzata via broadcasting numpy — O(N²) ma senza loop Python."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))

CITY_COUNTS     = [5, 10, 20, 30, 50, 75, 100, 150, 200,250,500]
N_TIMING_RUNS   = 7    # run indipendenti per ogni N (città casuali diversi)
N_ITER_TIMING   = 1000  # iterazioni SA fisse per tutti i punti

df_timing = []  # ogni riga: {'n_cities', 'mean_ms', 'std_ms', 'min_ms'}

print(f"Benchmark SA classico  —  n_iter={N_ITER_TIMING}, {N_TIMING_RUNS} run/punto")
print(f"{'N città':>8}  {'media (ms)':>12}  {'std (ms)':>10}  {'min (ms)':>10}")
print("─" * 48)

for n_cities in CITY_COUNTS:
    run_times = []
    for _ in range(N_TIMING_RUNS):
        positions_bench = np.random.rand(n_cities, 2)
        dm_bench = tsp_cost_function_fast(positions_bench)

        t0 = time.perf_counter()
        simulated_annealing_classic(dm_bench, n_iter=N_ITER_TIMING)
        run_times.append(time.perf_counter() - t0)

    mean_ms = np.mean(run_times)  * 1000
    std_ms  = np.std(run_times)   * 1000
    min_ms  = np.min(run_times)   * 1000
    df_timing.append({'n_cities': n_cities, 'mean_ms': mean_ms,
                      'std_ms': std_ms, 'min_ms': min_ms})
    print(f"{n_cities:>8}  {mean_ms:>12.1f}  {std_ms:>10.1f}  {min_ms:>10.1f}")

# %%
n_cities_arr = np.array([row['n_cities'] for row in df_timing], dtype=float)
mean_ms_arr  = np.array([row['mean_ms']  for row in df_timing])
std_ms_arr   = np.array([row['std_ms']   for row in df_timing])

# Retta O(N) di riferimento: scala in modo che passi per il punto mediano
mid_idx     = len(n_cities_arr) // 2
linear_ref  = (mean_ms_arr[mid_idx] / n_cities_arr[mid_idx]) * n_cities_arr

fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5))

# ── Scala lineare ─────────────────────────────────────────────────────────────
ax_lin.plot(n_cities_arr, mean_ms_arr, marker='o', color='#4f98a3',
            linewidth=2, markersize=6, label='SA classico (media)')
ax_lin.fill_between(n_cities_arr,
                    mean_ms_arr - std_ms_arr,
                    mean_ms_arr + std_ms_arr,
                    alpha=0.25, color='#4f98a3', label='± 1 std')
ax_lin.plot(n_cities_arr, linear_ref, '--', color='#e05c5c',
            linewidth=1.5, alpha=0.8, label='O(N) riferimento')
ax_lin.set_xlabel('Numero di città (N)')
ax_lin.set_ylabel('Tempo (ms)')
ax_lin.set_title(f'Tempo SA vs N città  [n_iter={N_ITER_TIMING}]')
ax_lin.legend()
ax_lin.grid(True, alpha=0.3)

# ── Scala log-log ─────────────────────────────────────────────────────────────
ax_log.loglog(n_cities_arr, mean_ms_arr, marker='o', color='#4f98a3',
              linewidth=2, markersize=6, label='SA classico (media)')
ax_log.fill_between(n_cities_arr,
                    np.maximum(mean_ms_arr - std_ms_arr, 1e-3),
                    mean_ms_arr + std_ms_arr,
                    alpha=0.25, color='#4f98a3')
ax_log.loglog(n_cities_arr, linear_ref, '--', color='#e05c5c',
              linewidth=1.5, alpha=0.8, label='O(N) riferimento')
ax_log.set_xlabel('Numero di città (N)')
ax_log.set_ylabel('Tempo (ms)')
ax_log.set_title('Scala log-log — verifica del comportamento asintotico')
ax_log.legend()
ax_log.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.show()

# Stima empirica dell'esponente di scaling: pendenza sulla scala log-log
log_slope, log_intercept = np.polyfit(np.log(n_cities_arr), np.log(mean_ms_arr), 1)
print(f"\nEsponente di scaling empirico (fit log-log): {log_slope:.2f}  "
      f"→ tempo ∝ N^{log_slope:.2f}")

