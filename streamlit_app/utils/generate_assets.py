import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def save_fig(fig, name):
    path = os.path.join(ASSETS_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    print(f"Saved {path}")

# 1. Shear Yielding
def gen_shear_yielding():
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Beam Web
    rect = patches.Rectangle((0, 0), 4, 2, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    # Flanges
    ax.plot([0, 4], [0, 0], 'k-', lw=4)
    ax.plot([0, 4], [2, 2], 'k-', lw=4)
    
    # 45 degree yield lines (Hatch)
    for i in np.linspace(-2, 6, 20):
        ax.plot([i, i+2], [0, 2], 'r-', lw=1, alpha=0.5)
        
    ax.text(2, 1, "SHEAR YIELDING\n(Pure Shear)", ha='center', va='center', fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    save_fig(fig, "shear_yielding.png")

# 2. Vierendeel Mechanism
def gen_vierendeel():
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Beam
    rect = patches.Rectangle((0, 0), 4, 2, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Flanges
    ax.plot([0, 4], [0, 0], 'k-', lw=3)
    ax.plot([0, 4], [2, 2], 'k-', lw=3)
    
    # Opening
    circle = patches.Circle((2, 1), 0.6, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(circle)
    
    # Plastic Hinges
    hinges = [(1.4, 0.4), (2.6, 0.4), (1.4, 1.6), (2.6, 1.6)]
    for h in hinges:
        ax.plot(h[0], h[1], 'ro', markersize=12)
        
    ax.text(2, 1, "VIERENDEEL", ha='center', va='center', fontweight='bold', fontsize=10)
    ax.text(2, -0.3, "Plastic Hinges Form at Corners", ha='center', fontsize=8)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')
    save_fig(fig, "vierendeel.png")

# 3. Web Post Buckling
def gen_web_buckling():
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Beam
    ax.plot([0, 4], [0, 0], 'k-', lw=3)
    ax.plot([0, 4], [2, 2], 'k-', lw=3)
    
    # Two Openings
    c1 = patches.Circle((1, 1), 0.6, linewidth=1, edgecolor='black', facecolor='white')
    c2 = patches.Circle((3, 1), 0.6, linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(c1)
    ax.add_patch(c2)
    
    # Web Post (Between holes)
    # Simulate buckling with curved lines
    x = np.linspace(1.8, 2.2, 20)
    y = np.linspace(0.2, 1.8, 20)
    ax.plot(x, y, 'b-', lw=1, alpha=0.3) # Straight ref
    
    # Buckled shape
    ax.plot([2, 1.9], [0.2, 0.9], 'r-', lw=2)
    ax.plot([1.9, 2.1], [0.9, 1.1], 'r-', lw=2)
    ax.plot([2.1, 2], [1.1, 1.8], 'r-', lw=2)
    
    ax.text(2, 1, "BUCKLING", ha='center', va='center', color='red', fontweight='bold', rotation=90)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.2, 2.2)
    ax.axis('off')
    save_fig(fig, "web_buckling.png")

# 4. Shear Yielding (2 Holes)
def gen_shear_yielding_2holes():
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    # Beam
    rect = patches.Rectangle((0, 0), 8, 2, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Flanges
    ax.plot([0, 8], [0, 0], 'k-', lw=3)
    ax.plot([0, 8], [2, 2], 'k-', lw=3)
    
    # 2 Holes
    c1 = patches.Circle((2.5, 1), 0.6, linewidth=2, edgecolor='black', facecolor='white')
    c2 = patches.Circle((5.5, 1), 0.6, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(c1)
    ax.add_patch(c2)
    
    # Yield Lines (Diagonal between holes)
    # X range 3.1 to 4.9 (between holes)
    for i in np.linspace(3, 5, 10):
        ax.plot([i, i+1], [0.2, 1.8], 'r--', lw=1)
        
    ax.text(4, 1, "SHEAR FAIL\nBETWEEN HOLES", ha='center', va='center', color='red', fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.2, 2.2)
    ax.axis('off')
    save_fig(fig, "shear_yielding_2holes.png")

if __name__ == "__main__":
    gen_shear_yielding()
    gen_vierendeel()
    gen_web_buckling()
    gen_shear_yielding_2holes()
