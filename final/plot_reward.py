"""
plot_reward.py
Static 2D reward-function figures for StraightLineEnv (linear r_cross variant).

Figure 1 — two panels, both symmetric by construction:
  Left  : reward vs cross-track, family of course-error curves
          (closing speed = 0, h_err=0, Va_err=0)
  Right : reward vs cross-track, family of "toward-line" closing speeds
          (course_err=0, h_err=0, Va_err=0; closing speed is always toward the line)

Figure 2 — 3-variable view: reward vs cross-track for a grid of
          (closing speed, altitude error) combinations, one subplot per
          closing speed, one curve per altitude error.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── reward function ───────────────────────────────────────────────────────────

def compute_reward(cross_track, course_err, h_err, Va_err, closing_toward_line):
    """
    closing_toward_line : signed speed toward the line (positive = converging),
                          regardless of which side the aircraft is on.
                          Replaces raw d_cross to keep plots symmetric.
    """
    ct = abs(cross_track)

    # cross-track penalty — linear, capped at 1.0
    r_cross = min(1.0, ct / 200)

    # guidance terms
    blend     = np.exp(-ct**2 / 1250)
    r_closing = (1 - blend) * 0.5 * np.tanh(closing_toward_line / 5)
    r_course  = -blend * 0.5 * (1 - np.exp(-course_err**2 / 0.3))

    # secondary penalties
    r_h  = 0.2 * (1 - np.exp(-h_err**2  / 2500))
    r_va = 0.1 * (1 - np.exp(-Va_err**2 / 9))

    reward = 1.0 - r_cross + r_closing + r_course - r_h - r_va

    # proximity bonuses
    if ct < 1.0:
        reward += 1.0
    elif ct < 5.0:
        reward += 0.25

    return reward

compute_reward_v = np.vectorize(compute_reward)

# ── shared helpers ────────────────────────────────────────────────────────────

def style_ax(ax, xlabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('reward per step', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.axhline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.axvline(0, color='k', linewidth=0.6, linestyle='--', alpha=0.4)
    ax.set_xlim(-200, 200)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvspan(-5,  5,  alpha=0.08, color='gold')
    ax.axvspan(-1,  1,  alpha=0.15, color='green')

ct = np.linspace(-200, 200, 800)

# ── Figure 1 ─────────────────────────────────────────────────────────────────

fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
fig1.subplots_adjust(wspace=0.32)

# Left: vary course error, closing = 0 (symmetric by construction)
course_errors = [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3]
course_labels = [r'$0$', r'$\pi/12$', r'$\pi/6$', r'$\pi/4$', r'$\pi/3$']
course_colors = ['#1a6faf', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

ax = axes[0]
for chi, lbl, col in zip(course_errors, course_labels, course_colors):
    r = compute_reward_v(ct, chi, 0, 0, 0)
    ax.plot(ct, r, color=col, linewidth=2.0, label=f'χ err = {lbl} rad')
ax.legend(fontsize=9, loc='upper right', framealpha=0.85)
style_ax(ax, 'cross-track error  (m)',
         'Reward vs cross-track error\n'
         r'(closing=0, $h_{err}$=0, $V_{a,err}$=0)')

# Right: vary closing speed (always toward line), course_err = 0
closing_vals   = [-10, -5, 0, 5, 10]
closing_labels = ['-10 m/s (diverging)', '-5 m/s', '0 m/s',
                  '+5 m/s', '+10 m/s (converging)']
closing_colors = ['#d62728', '#ff7f0e', '#7f7f7f', '#2ca02c', '#1a6faf']

ax = axes[1]
for cv, lbl, col in zip(closing_vals, closing_labels, closing_colors):
    ls = '--' if cv < 0 else '-'
    r  = compute_reward_v(ct, 0, 0, 0, cv)
    ax.plot(ct, r, color=col, linewidth=2.0, linestyle=ls, label=lbl)
ax.legend(fontsize=9, loc='upper right', framealpha=0.85,
          title='closing speed (toward line)', title_fontsize=9)
style_ax(ax, 'cross-track error  (m)',
         'Reward vs cross-track error\n'
         r'($\chi_{err}$=0, $h_{err}$=0, $V_{a,err}$=0)')

fig1.text(0.5, -0.02,
          'Shaded bands: gold = |ct| < 5 m (+0.25 bonus), green = |ct| < 1 m (+1.00 bonus). '
          'Closing speed defined as positive toward the line.',
          ha='center', fontsize=9, color='#555555')

plt.tight_layout()
fig1.savefig('reward_fig1_symmetric.png', dpi=180, bbox_inches='tight', facecolor='white')
print('Saved: reward_fig1_symmetric.png')


# ── Figure 2 — cross-track x closing speed x altitude error ──────────────────
#
# Layout: 1 subplot per closing speed (3 speeds), curves = altitude errors

closing_panel  = [-5, 0, 5]
closing_titles = ['Diverging  (−5 m/s)', 'Neutral  (0 m/s)', 'Converging  (+5 m/s)']

h_errs   = [-40, -20, 0, 20, 40]
h_labels = ['-40 m', '-20 m', '0 m', '+20 m', '+40 m']
h_colors = ['#d62728', '#ff7f0e', '#1a6faf', '#2ca02c', '#9467bd']

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
fig2.suptitle('Reward vs cross-track error — effect of altitude error & closing speed\n'
              r'($\chi_{err}$=0, $V_{a,err}$=0)',
              fontsize=12, fontweight='bold', y=1.01)

for ax, cv, title in zip(axes2, closing_panel, closing_titles):
    for h, lbl, col in zip(h_errs, h_labels, h_colors):
        ls = '--' if h != 0 else '-'
        lw = 2.5  if h == 0 else 1.6
        r  = compute_reward_v(ct, 0, h, 0, cv)
        ax.plot(ct, r, color=col, linewidth=lw, linestyle=ls, label=lbl)
    style_ax(ax, 'cross-track error  (m)', title)
    if ax is axes2[0]:
        ax.set_ylabel('reward per step', fontsize=11)
    else:
        ax.set_ylabel('')

axes2[1].legend(fontsize=9, loc='upper right', framealpha=0.85,
                title='altitude error', title_fontsize=9)

fig2.text(0.5, -0.04,
          'Solid line = h_err=0. Dashed = altitude offset. '
          'Shaded bands: gold = |ct| < 5 m (+0.25), green = |ct| < 1 m (+1.00).',
          ha='center', fontsize=9, color='#555555')

plt.tight_layout()
fig2.savefig('reward_fig2_3var.png', dpi=180, bbox_inches='tight', facecolor='white')
print('Saved: reward_fig2_3var.png')

plt.show()