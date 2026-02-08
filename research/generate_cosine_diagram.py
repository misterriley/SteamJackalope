import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def create_cosine_sim_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set axis at origin
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # Vectors in different quadrants
    v1 = np.array([4, 3])   # Q1
    v2 = np.array([-2, 5])  # Q2
    
    # Plot vectors
    ax.quiver(0, 0, v1[0], v1[1], color='blue', label='Vector A', angles='xy', scale_units='xy', scale=1, width=0.015)
    ax.quiver(0, 0, v2[0], v2[1], color='red', label='Vector B', angles='xy', scale_units='xy', scale=1, width=0.015)

    # Angle arc
    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])
    angles = np.linspace(theta1, theta2, 50)
    radius = 1.5
    ax.plot(radius * np.cos(angles), radius * np.sin(angles), color='green', lw=2, linestyle='--')
    ax.text(radius * np.cos(np.mean(angles)) * 1.3, radius * np.sin(np.mean(angles)) * 1.3, r'$\theta$', fontsize=25, color='green', ha='center')

    # Formulas
    formula = r'$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$'
    ax.text(0.05, 0.90, formula, transform=ax.transAxes, fontsize=28, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    dot_formula = r'$\mathbf{A} \cdot \mathbf{B} = A_x B_x + A_y B_y$'
    ax.text(0.05, 0.82, dot_formula, transform=ax.transAxes, fontsize=20, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Axis labels
    ax.set_xlabel('Dimension 1', loc='right')
    ax.set_ylabel('Dimension 2', loc='top', rotation=0)
    
    # Limits
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 6)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.title("Cosine Similarity: Measuring the angle between vectors", fontsize=18, pad=30)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('cosine_similarity_diagram.png', dpi=300, bbox_inches='tight')
    print("Cosine similarity diagram saved to cosine_similarity_diagram.png")

if __name__ == "__main__":
    create_cosine_sim_diagram()
