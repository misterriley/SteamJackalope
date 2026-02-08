import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Define steps
    steps = [
        "Raw Steam Tags\n(Top 20 Counts)",
        "LOD Imputation\n(v_min / √2)",
        "Affinity Boost\n(Tag Correlations)",
        "Bayesian Smoothing\n(K / (N + K))",
        "CLR Transform\n(Log-Ratio)",
        "ZCA Whitening\n(Decorrelation)"
    ]
    
    colors = ['#f9f9f9', '#e1f5fe', '#b3e5fc', '#81d4fa', '#4fc3f2', '#29b6f6']
    
    x_pos = np.linspace(10, 90, len(steps))
    y_pos = 25
    width = 12
    height = 10

    for i, (step, color) in enumerate(zip(steps, colors)):
        # Draw box
        rect = patches.FancyBboxPatch((x_pos[i] - width/2, y_pos - height/2), width, height, 
                                     boxstyle="round,pad=0.3", ec="black", fc=color)
        ax.add_patch(rect)
        
        # Add text
        ax.text(x_pos[i], y_pos, step, ha='center', va='center', fontsize=9, weight='bold')
        
        # Draw arrow
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_pos[i+1] - width/2 - 0.5, y_pos), 
                        xytext=(x_pos[i] + width/2 + 0.5, y_pos),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))

    plt.title("Tag Vector Preprocessing Pipeline", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('tag_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    print("Pipeline diagram saved to tag_pipeline_diagram.png")

if __name__ == "__main__":
    import numpy as np
    create_pipeline_diagram()
