# %% [markdown]
# # Simulated Annealing per l'Ottimizzazione di Portafoglio
#
# ### Problema: Portfolio Optimization (Markowitz)
#
# Dato un insieme di $N$ asset finanziari con rendimenti attesi $\mu$ e matrice di covarianza $\Sigma$,
# vogliamo trovare l'allocazione del capitale (i **pesi** del portafoglio) che
# **massimizza il Sharpe Ratio**:
#
# $$\text{Sharpe}(w) = \frac{R(w) - r_f}{\sigma(w)} = \frac{\mu^\top w - r_f}{\sqrt{w^\top \Sigma w}}$$
#
# dove $r_f$ è il tasso privo di rischio, $R(w) = \mu^\top w$ il rendimento atteso
# e $\sigma(w) = \sqrt{w^\top \Sigma w}$ la volatilità del portafoglio.
#
# ### Interpretazione del Sharpe Ratio
#
# Lo Sharpe Ratio misura il **rendimento in eccesso ottenuto per ogni unità di rischio assunto**:
# quante unità di rendimento extra (sopra $r_f$) si guadagnano per ogni punto percentuale
# di volatilità sopportata.
# Un Sharpe alto significa che il portafoglio è ben compensato per il rischio che prende.
#
# **Valori di riferimento tipici** (annualizzati):
#
# | Sharpe Ratio | Giudizio |
# |:---:|:---|
# | < 0 | Peggio del tasso privo di rischio |
# | 0 – 0.5 | Scarso |
# | 0.5 – 1.0 | Accettabile (tipico per portafogli azionari diversificati) |
# | 1.0 – 2.0 | Buono |
# | > 2.0 | Eccellente (raro su lunghi orizzonti temporali) |
#
# I fondi più performanti al mondo (es. Medallion Fund di Renaissance Technologies)
# hanno raggiunto Sharpe > 2 in modo consistente — un risultato eccezionale.
#
# I pesi devono soddisfare i vincoli del **simplex**:
#
# $$\sum_{i=1}^{N} w_i = 1, \qquad w_i \geq 0 \quad \forall i$$
#
# A differenza del TSP (spazio discreto delle permutazioni), qui lo spazio delle soluzioni è
# **continuo**: il simplex $N$-dimensionale. Il Simulated Annealing si adatta direttamente
# a entrambi i casi — cambia solo la mossa di proposta.



# %%
import numpy as np
import matplotlib.pyplot as plt
import random
import math

np.random.seed(42)
random.seed(42)

N_ASSETS = 8
RISK_FREE_RATE = 0.035  # 2% tasso privo di rischio annualizzato

# Rendimenti attesi casuali, ordinati crescenti (4%–20%)
expected_returns = np.sort(np.random.uniform(0.04, 0.20, N_ASSETS))

# Volatilità correlata al rendimento: σ_i ≈ 2 · μ_i + ε_i
# In media un asset con rendimento μ ha volatilità ~2μ (Sharpe ~0.5); il rumore rompe la linearità esatta
noise_vol = np.random.uniform(-0.04, 0.04, N_ASSETS)
sigmas = np.clip(2.0 * expected_returns + noise_vol, 0.05, None)

# Matrice di covarianza con correlazione uniforme ρ=0.35 tra tutti gli asset (mercato comune)
base_correlation = 0.35
corr_matrix = np.full((N_ASSETS, N_ASSETS), base_correlation)
np.fill_diagonal(corr_matrix, 1.0)
cov_matrix = np.diag(sigmas) @ corr_matrix @ np.diag(sigmas)

asset_names = [f"A{i+1}" for i in range(N_ASSETS)]

print(f"{'Asset':>6}  {'Rendimento':>12}  {'Volatilità':>12}")
print("─" * 36)
for i, name in enumerate(asset_names):
    print(f"  {name:4s}  {expected_returns[i]*100:>10.1f}%  {sigmas[i]*100:>10.1f}%")


# %% [markdown]
# ## Funzione di costo
#
# Il problema è formulato come **minimizzazione** del negativo dello Sharpe Ratio:
#
# $$\mathcal{C}(w) = -\text{Sharpe}(w) = -\frac{\mu^\top w - r_f}{\sqrt{w^\top \Sigma w}}$$
#
# ### Spazio delle soluzioni: il simplex
#
# L'insieme dei portafogli ammissibili è il simplex $(N-1)$-dimensionale:
#
# $$\Delta^{N-1} = \left\{ w \in \mathbb{R}^N : \sum_i w_i = 1,\; w_i \geq 0 \right\}$$
#
# Con $N = 10$ asset esistono **infinite** combinazioni convesse — non c'è un catalogo finito
# di soluzioni da cercare come nel TSP.

# %%
def portfolio_return(weights):
    """Rendimento atteso del portafoglio."""
    return np.dot(weights, expected_returns)

def portfolio_variance(weights):
    """Varianza del portafoglio."""
    return weights @ cov_matrix @ weights

def portfolio_sharpe(weights):
    """Sharpe Ratio del portafoglio."""
    ret = portfolio_return(weights)
    std = np.sqrt(portfolio_variance(weights))
    return (ret - RISK_FREE_RATE) / std

def portfolio_cost(weights):
    """Funzione di costo per SA: meno Sharpe Ratio (da minimizzare)."""
    return -portfolio_sharpe(weights)


# Portafoglio equipesato come baseline di riferimento
equal_weights = np.ones(N_ASSETS) / N_ASSETS
print(f"Portafoglio equipesato:")
print(f"  Rendimento: {portfolio_return(equal_weights)*100:.2f}%")
print(f"  Volatilità: {np.sqrt(portfolio_variance(equal_weights))*100:.2f}%")
print(f"  Sharpe Ratio:     {portfolio_sharpe(equal_weights):.3f}")

# %%
def draw_portfolio(weights, title="Portafoglio", color='#4f98a3'):
    """Visualizza i pesi del portafoglio e le metriche principali."""
    ret = portfolio_return(weights)
    vol = np.sqrt(portfolio_variance(weights))
    sharpe = portfolio_sharpe(weights)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(asset_names, weights * 100, color=color, edgecolor='white', linewidth=1.2)

    for bar, w in zip(bars, weights):
        if w > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{w*100:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Asset')
    ax.set_ylabel('Peso (%)')
    ax.set_title(
        f"{title}\n"
        f"Rendimento: {ret*100:.2f}%  |  Volatilità: {vol*100:.2f}%  |  Sharpe: {sharpe:.3f}"
    )
    ax.set_ylim(0, max(weights * 100) * 1.25 + 2)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()


draw_portfolio(equal_weights, title="Portafoglio Equipesato (baseline)")

# %%
# Frontiera efficiente approssimata via simulazione Monte Carlo
N_MC = 5000
mc_returns     = np.empty(N_MC)
mc_volatilities = np.empty(N_MC)
mc_sharpes     = np.empty(N_MC)

for k in range(N_MC):
    w = np.random.dirichlet(np.ones(N_ASSETS))
    mc_returns[k]      = portfolio_return(w) * 100
    mc_volatilities[k] = np.sqrt(portfolio_variance(w)) * 100
    mc_sharpes[k]      = portfolio_sharpe(w)

eq_return_pct = portfolio_return(equal_weights) * 100
eq_vol_pct    = np.sqrt(portfolio_variance(equal_weights)) * 100
eq_sharpe     = portfolio_sharpe(equal_weights)

fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(mc_volatilities, mc_returns, c=mc_sharpes,
                     cmap='viridis', alpha=0.4, s=8)
plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
ax.scatter(eq_vol_pct, eq_return_pct, color='#e05c5c', s=200, zorder=5, marker='*',
           label=f'Equipesato (Sharpe={eq_sharpe:.2f})')
ax.set_xlabel('Volatilità (%)')
ax.set_ylabel('Rendimento atteso (%)')
ax.set_title(f'Universo di portafogli — {N_MC:,} campioni Monte Carlo')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## Parte 2 — Simulated Annealing
#
# ### Mossa: trasferimento di peso tra due asset
#
# Nel TSP la mossa 2-opt invertiva un segmento della permutazione.
# Qui dobbiamo spostarci sul simplex, rispettando i vincoli $\sum w_i = 1$ e $w_i \geq 0$.
#
# **Idea**: scegli due asset $i$ e $j$ a caso; trasferisci una quota casuale
# $\delta \in [0, w_i]$ dal primo al secondo:
#
# $$w_i \leftarrow w_i - \delta, \qquad w_j \leftarrow w_j + \delta$$
#
# La somma rimane 1 e nessun peso diventa negativo — il vettore resta nel simplex.

# %%
def propose_weight_transfer(weights):
    """Mossa SA: trasferisce un peso casuale δ dall'asset i all'asset j."""
    new_weights = weights.copy()
    asset_i, asset_j = random.sample(range(len(weights)), 2)
    delta = random.uniform(0, new_weights[asset_i])
    new_weights[asset_i] -= delta
    new_weights[asset_j] += delta
    return new_weights


def simulated_annealing_portfolio(n_iter=2000, T_start=1.0, T_end=0.001):
    """
    SA per massimizzare lo Sharpe Ratio (minimizza il negativo).
    Proposta: trasferimento di peso tra due asset.
    Ritorna: best_weights, best_sharpe, history_best, history_current
      - history_best:    Sharpe migliore ad ogni iterazione (monotone crescente)
      - history_current: Sharpe della soluzione accettata ad ogni iterazione
    """
    current_weights = np.random.dirichlet(np.ones(N_ASSETS))
    current_cost    = portfolio_cost(current_weights)
    best_weights    = current_weights.copy()
    best_cost       = current_cost
    history_best    = []
    history_current = []

    for iteration in range(n_iter):
        temperature = T_start * (T_end / T_start) ** (iteration / n_iter)

        candidate_weights = propose_weight_transfer(current_weights)
        candidate_cost    = portfolio_cost(candidate_weights)
        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_weights, current_cost = candidate_weights, candidate_cost

        if current_cost < best_cost:
            best_weights, best_cost = current_weights.copy(), current_cost

        history_best.append(-best_cost)
        history_current.append(-current_cost)

    return best_weights, -best_cost, history_best, history_current


# %%
# np.random.seed(42)
# random.seed(42)

best_weights_sa, best_sharpe_sa, history_sa_best, history_sa_current = (
    simulated_annealing_portfolio(n_iter=1000, T_start=10.0, T_end=0.001)
)

print(f"Miglior Sharpe Ratio (SA):  {best_sharpe_sa:.4f}")
print(f"  Rendimento:               {portfolio_return(best_weights_sa)*100:.2f}%")
print(f"  Volatilità:               {np.sqrt(portfolio_variance(best_weights_sa))*100:.2f}%")
print(f"\nBaseline equipesato — Sharpe: {eq_sharpe:.4f}")
print(f"Miglioramento:  +{(best_sharpe_sa / eq_sharpe - 1)*100:.1f}%")

draw_portfolio(best_weights_sa,
               title="Portafoglio SA — Massimo Sharpe Ratio",
               color='#7a7974')

# %%
plt.figure(figsize=(8, 4))
plt.plot(history_sa_current, color='#b0a99f', linewidth=1.0, label='Sharpe corrente')
plt.plot(history_sa_best,    color='#7a7974', linewidth=1.5, label='Sharpe migliore')
plt.axhline(eq_sharpe, color='#e05c5c', linestyle='--', linewidth=1.2,
            alpha=0.8, label=f'Baseline equipesato ({eq_sharpe:.3f})')
plt.xlabel('Iterazione')
plt.ylabel('Sharpe Ratio')
plt.title('Convergenza SA — Ottimizzazione Portafoglio')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Simulazione delle performance — investimento di €10.000
#
# Dati i parametri del portafoglio ottimale ($\mu$, $\sigma$), simuliamo l'evoluzione
# del capitale nel tempo tramite il modello **Geometric Brownian Motion (GBM)**:
#
# $$S_{t+\Delta t} = S_t \cdot \exp\!\left[\left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\; Z\right], \quad Z \sim \mathcal{N}(0,1)$$
#
# Il termine $-\sigma^2/2$ è la correzione di Itô: garantisce che il rendimento atteso
# del processo sia $\mu$ (non $\mu + \sigma^2/2$), allineandolo con la media aritmetica
# dichiarata nel modello.
#
# Lanciamo $N$ scenari indipendenti e mostriamo la banda di confidenza 5°–95°.

# %%
INITIAL_INVESTMENT = 10_000  # euro
N_YEARS = 1
N_TRADING_DAYS = N_YEARS * 252
N_SIMULATIONS = 100


def simulate_gbm(annual_return, annual_vol, n_days, n_sims, initial_value):
    """Simula n_sims traiettorie GBM giornaliere; ritorna array (n_sims, n_days+1)."""
    dt = 1 / 252
    daily_drift = (annual_return - 0.5 * annual_vol ** 2) * dt
    daily_vol   = annual_vol * np.sqrt(dt)
    log_returns = daily_drift + daily_vol * np.random.randn(n_sims, n_days)
    values = np.empty((n_sims, n_days + 1))
    values[:, 0] = initial_value
    values[:, 1:] = initial_value * np.exp(np.cumsum(log_returns, axis=1))
    return values


np.random.seed(0)

sa_annual_return = portfolio_return(best_weights_sa)
sa_annual_vol    = np.sqrt(portfolio_variance(best_weights_sa))
eq_annual_return = portfolio_return(equal_weights)
eq_annual_vol    = np.sqrt(portfolio_variance(equal_weights))

sa_paths = simulate_gbm(sa_annual_return, sa_annual_vol,
                        N_TRADING_DAYS, N_SIMULATIONS, INITIAL_INVESTMENT)
eq_paths = simulate_gbm(eq_annual_return, eq_annual_vol,
                        N_TRADING_DAYS, N_SIMULATIONS, INITIAL_INVESTMENT)

time_axis = np.linspace(0, N_YEARS, N_TRADING_DAYS + 1)
rf_curve  = INITIAL_INVESTMENT * np.exp(RISK_FREE_RATE * time_axis)

fig, ax = plt.subplots(figsize=(10, 6))

# Banda SA
ax.fill_between(time_axis,
                np.percentile(sa_paths, 5,  axis=0),
                np.percentile(sa_paths, 95, axis=0),
                alpha=0.15, color='#4f98a3')
ax.plot(time_axis, np.percentile(sa_paths, 50, axis=0),
        color='#4f98a3', linewidth=2.0,
        label=f'SA ottimale — mediana  (μ={sa_annual_return*100:.1f}%, σ={sa_annual_vol*100:.1f}%)')

# Banda equipesato
ax.fill_between(time_axis,
                np.percentile(eq_paths, 5,  axis=0),
                np.percentile(eq_paths, 95, axis=0),
                alpha=0.12, color='#b0a99f')
ax.plot(time_axis, np.percentile(eq_paths, 50, axis=0),
        color='#7a7974', linewidth=1.5, linestyle='--',
        label=f'Equipesato — mediana  (μ={eq_annual_return*100:.1f}%, σ={eq_annual_vol*100:.1f}%)')

# Risk-free
ax.plot(time_axis, rf_curve,
        color='#e05c5c', linewidth=1.2, linestyle=':',
        label=f'Risk-free ({RISK_FREE_RATE*100:.1f}%)')

ax.axhline(INITIAL_INVESTMENT, color='grey', linewidth=0.8, linestyle='--', alpha=0.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'€{x:,.0f}'))
ax.set_xlabel('Anni')
ax.set_ylabel('Valore del portafoglio (€)')
ax.set_title(
    f'Simulazione Monte Carlo — {N_SIMULATIONS} scenari, orizzonte {N_YEARS} anni\n'
    f'Investimento iniziale: €{INITIAL_INVESTMENT:,}  |  banda = 5°–95° percentile'
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
sa_final = sa_paths[:, -1]
eq_final = eq_paths[:, -1]

print(f"Risultati dopo {N_YEARS} anni  (investimento iniziale: €{INITIAL_INVESTMENT:,})\n")
print(f"{'':35s}  {'SA ottimale':>13}  {'Equipesato':>12}")
print("─" * 65)
print(f"{'Mediana valore finale':35s}  "
      f"€{np.median(sa_final):>11,.0f}  €{np.median(eq_final):>10,.0f}")
print(f"{'Scenario pessimistico (5° percentile)':35s}  "
      f"€{np.percentile(sa_final, 5):>11,.0f}  €{np.percentile(eq_final, 5):>10,.0f}")
print(f"{'Scenario ottimistico (95° percentile)':35s}  "
      f"€{np.percentile(sa_final, 95):>11,.0f}  €{np.percentile(eq_final, 95):>10,.0f}")
print(f"{'Probabilità di perdita del capitale':35s}  "
      f"{(sa_final < INITIAL_INVESTMENT).mean()*100:>10.1f}%  "
      f"{(eq_final < INITIAL_INVESTMENT).mean()*100:>9.1f}%")


# %%
# Posizione della soluzione SA nella nuvola Monte Carlo
sa_return_pct = portfolio_return(best_weights_sa) * 100
sa_vol_pct    = np.sqrt(portfolio_variance(best_weights_sa)) * 100

fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(mc_volatilities, mc_returns, c=mc_sharpes,
                     cmap='viridis', alpha=0.3, s=8)
plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
ax.scatter(eq_vol_pct, eq_return_pct, color='#e05c5c', s=200, zorder=5, marker='*',
           label=f'Equipesato (Sharpe={eq_sharpe:.2f})')
ax.scatter(sa_vol_pct, sa_return_pct, color='#2ecc71', s=200, zorder=6, marker='D',
           label=f'SA ottimale (Sharpe={best_sharpe_sa:.2f})')
ax.set_xlabel('Volatilità (%)')
ax.set_ylabel('Rendimento atteso (%)')
ax.set_title('Portafoglio SA vs universo Monte Carlo')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ---
# ## Parte 3 — Video dell'ottimizzazione
#
# Registra l'evoluzione dei pesi durante il SA.
# A sinistra: istogramma pesi corrente (grigio) vs best-so-far (rosso).
# A destra: curva di convergenza dello Sharpe Ratio con cursore.

# %%
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

def simulated_annealing_portfolio_with_snapshots(
    n_iter=2000, T_start=1.0, T_end=0.001, snapshot_every=20
):
    """
    SA portafoglio con raccolta snapshot a intervalli regolari.
    Ritorna best_weights, best_sharpe, history_best, history_current, snapshots.
    Chiavi snapshot: current_weights, current_sharpe, best_weights, best_sharpe,
                     temperature, history_best, history_current.
    """
    current_weights = np.random.dirichlet(np.ones(N_ASSETS))
    current_cost    = portfolio_cost(current_weights)
    best_weights    = current_weights.copy()
    best_cost       = current_cost
    history_best    = []
    history_current = []
    snapshots       = []

    for iteration in range(n_iter):
        temperature = T_start * (T_end / T_start) ** (iteration / n_iter)

        candidate_weights = propose_weight_transfer(current_weights)
        candidate_cost    = portfolio_cost(candidate_weights)
        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_weights, current_cost = candidate_weights, candidate_cost

        if current_cost < best_cost:
            best_weights, best_cost = current_weights.copy(), current_cost

        history_best.append(-best_cost)
        history_current.append(-current_cost)

        if iteration % snapshot_every == 0 or iteration == n_iter - 1:
            snapshots.append({
                'iteration':       iteration,
                'current_weights': current_weights.copy(),
                'current_sharpe':  -current_cost,
                'best_weights':    best_weights.copy(),
                'best_sharpe':     -best_cost,
                'temperature':     temperature,
                'history_best':    history_best[:],
                'history_current': history_current[:],
            })

    return best_weights, -best_cost, history_best, history_current, snapshots


def create_portfolio_video(snapshots, output_path='sa_portfolio.mp4', fps=15):
    """
    Crea un video con due subplot animati:
    - sinistra: barre dei pesi (corrente grigio, best rosso) affiancate per asset
    - destra:   curva dello Sharpe Ratio con cursore verticale
    """
    n_iter_total    = snapshots[-1]['iteration'] + 1
    full_history_current = snapshots[-1]['history_current']
    full_history_best    = snapshots[-1]['history_best']
    sharpe_min = min(min(full_history_current), min(full_history_best)) * 0.90
    sharpe_max = max(max(full_history_current), max(full_history_best)) * 1.10

    current_color = '#aaaaaa'
    best_color    = '#e05c5c'
    x_positions   = np.arange(N_ASSETS)
    bar_width     = 0.38

    fig, (ax_bar, ax_conv) = plt.subplots(1, 2, figsize=(14, 6))

    def draw_frame(frame_idx):
        snap = snapshots[frame_idx]

        # ── Barre dei pesi ────────────────────────────────────────────────────
        ax_bar.clear()
        ax_bar.bar(x_positions - bar_width / 2,
                   snap['current_weights'] * 100, bar_width,
                   color=current_color,
                   label=f'Corrente: Sharpe={snap["current_sharpe"]:.3f}')
        ax_bar.bar(x_positions + bar_width / 2,
                   snap['best_weights'] * 100, bar_width,
                   color=best_color,
                   label=f'Best: Sharpe={snap["best_sharpe"]:.3f}')
        ax_bar.set_xticks(x_positions)
        ax_bar.set_xticklabels(asset_names)
        ax_bar.set_ylabel('Peso (%)')
        ax_bar.set_ylim(0, 100)
        ax_bar.set_title(f"iter {snap['iteration']}  |  T = {snap['temperature']:.4f}",
                         fontsize=11)
        ax_bar.legend(fontsize=9, loc='upper right')
        ax_bar.grid(True, alpha=0.2, axis='y')

        # ── Convergenza Sharpe ────────────────────────────────────────────────
        ax_conv.clear()
        ax_conv.plot(snap['history_current'], color=current_color,
                     linewidth=1.0, label='Sharpe corrente')
        ax_conv.plot(snap['history_best'], color=best_color,
                     linewidth=1.5, label='Sharpe migliore')
        ax_conv.axvline(x=snap['iteration'], color=best_color,
                        linestyle='--', alpha=0.8, linewidth=1.5)
        ax_conv.set_xlim(0, n_iter_total)
        ax_conv.set_ylim(sharpe_min, sharpe_max)
        ax_conv.set_xlabel('Iterazione')
        ax_conv.set_ylabel('Sharpe Ratio')
        ax_conv.set_title('Convergenza SA — Sharpe Ratio', fontsize=11)
        ax_conv.legend(fontsize=9)
        ax_conv.grid(True, alpha=0.3)

        fig.tight_layout()

    anim = FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=1000 // fps)

    try:
        writer = FFMpegWriter(fps=fps, metadata={'title': 'SA Portfolio Optimization'})
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
    simulated_annealing_portfolio_with_snapshots(
        n_iter=600, T_start=5.0, T_end=0.01, snapshot_every=10
    )
)
print(f"Snapshot raccolti: {len(snapshots_video)}")
print(f"Miglior distanza: {best_cost_video:.3f}")

create_portfolio_video(
    snapshots_video,
    output_path='sa_portfolio.mp4', fps=2
)
