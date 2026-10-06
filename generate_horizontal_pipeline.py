import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Stages definition matching user exact request
stages = [
    {
        "num": "01",
        "title": "Input Image",
        "sub": "Base64 string",
        "tech": "HTTP REST / JSON\nBase64 encoded string",
        "stage_type": "INPUT"
    },
    {
        "num": "02",
        "title": "Decode to NumPy",
        "sub": "Array Conversion",
        "tech": "OpenCV cv2.imdecode\nBGR to RGB conversion",
        "stage_type": "PREPROCESSING"
    },
    {
        "num": "03",
        "title": "RetinaFace Detection",
        "sub": "Bounding Box + Landmarks",
        "tech": "InsightFace SCRFD / Retina\nQuality gate (det_score ≥ 0.6)",
        "stage_type": "DETECTION"
    },
    {
        "num": "04",
        "title": "Face Alignment",
        "sub": "5-point Landmark Norm",
        "tech": "Similarity transformation\nCanonical pose alignment",
        "stage_type": "ALIGNMENT"
    },
    {
        "num": "05",
        "title": "ArcFace Extraction",
        "sub": "ResNet + ArcFace Head",
        "tech": "Deep CNN (ResNet-50)\nAdditive angular margin",
        "stage_type": "FEATURE EXTRACTION"
    },
    {
        "num": "06",
        "title": "Embedding Vector",
        "sub": "512-d Feature Space",
        "tech": "Dense latent identity vector\n512 floating-point dims",
        "stage_type": "REPRESENTATION"
    },
    {
        "num": "07",
        "title": "L2 Normalisation",
        "sub": "Unit Sphere Projection",
        "tech": "Norm: v / ||v||₂\nUnit hypersphere mapping",
        "stage_type": "NORMALISATION"
    },
    {
        "num": "08",
        "title": "Qdrant Cosine Search",
        "sub": "Course Filter, Limit=1",
        "tech": "HNSW vector search\nPayload course_id filter",
        "stage_type": "VECTOR RETRIEVAL"
    },
    {
        "num": "09",
        "title": "Threshold Decision",
        "sub": "score >= 0.65 ?",
        "tech": "Cosine similarity cutoff\nVerification decision gate",
        "stage_type": "DECISION"
    },
    {
        "num": "10",
        "title": "Matched Student ID",
        "sub": "+ Score (or No Match)",
        "tech": "Final match classification\nor rejection response",
        "stage_type": "OUTPUT"
    }
]

# Color palette: Smooth gradient from Blue to Cyan to Teal to Green
# 10 steps gradient
hex_palette = [
    {"bg": "#EFF6FF", "border": "#1D4ED8", "badge": "#1E40AF", "text": "#1E3A8A", "accent": "#3B82F6"}, # 1: Deep Blue
    {"bg": "#F0F7FF", "border": "#2563EB", "badge": "#1D4ED8", "text": "#1E40AF", "accent": "#60A5FA"}, # 2: Royal Blue
    {"bg": "#F0F9FF", "border": "#0284C7", "badge": "#0369A1", "accent": "#38BDF8", "text": "#075985"}, # 3: Sky Blue
    {"bg": "#ECFEFF", "border": "#0891B2", "badge": "#0E7490", "accent": "#22D3EE", "text": "#155E75"}, # 4: Cyan
    {"bg": "#F0FDFA", "border": "#0D9488", "badge": "#0F766E", "accent": "#2DD4BF", "text": "#115E59"}, # 5: Dark Teal
    {"bg": "#ECFDF5", "border": "#059669", "badge": "#047857", "accent": "#34D399", "text": "#065F46"}, # 6: Teal / Mint
    {"bg": "#F0FDF4", "border": "#10B981", "badge": "#059669", "accent": "#4ADE80", "text": "#065F46"}, # 7: Emerald
    {"bg": "#F0FDF4", "border": "#16A34A", "badge": "#15803D", "accent": "#22C55E", "text": "#166534"}, # 8: Green
    {"bg": "#F7FEE7", "border": "#65A30D", "badge": "#4D7C0F", "accent": "#84CC16", "text": "#3F6212"}, # 9: Lime-Green
    {"bg": "#ECFDF5", "border": "#15803D", "badge": "#166534", "accent": "#22C55E", "text": "#14532D"}  # 10: Pure Output Green
]

def generate_horizontal_pipeline():
    fig_width = 32
    fig_height = 8.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis('off')

    # Figure Header
    plt.text(fig_width / 2, 7.8, "FIGURE 4.11: Facial Recognition Pipeline Flow Diagram", 
             fontsize=20, fontweight='bold', ha='center', va='center', color='#0F172A',
             fontfamily='DejaVu Sans')
    plt.text(fig_width / 2, 7.3, "Sequential stages of the InsightFace recognition process with labelled stage boxes, algorithms, and decision verification", 
             fontsize=12, ha='center', va='center', color='#64748B', style='italic',
             fontfamily='DejaVu Sans')

    n_stages = 10
    card_w = 2.65
    card_h = 4.2
    gap_x = 0.45
    margin_x = (fig_width - (n_stages * card_w + (n_stages - 1) * gap_x)) / 2
    y_card = 1.9

    # Category grouping brackets along top of cards
    categories = [
        ("INPUT STAGE", 0, 0, "#1D4ED8"),
        ("PREPROCESSING & DETECTION", 1, 3, "#0284C7"),
        ("FEATURE ENGINEERING & NORMALISATION", 4, 6, "#0D9488"),
        ("VECTOR SEARCH & DECISION", 7, 8, "#16A34A"),
        ("OUTPUT STAGE", 9, 9, "#15803D")
    ]

    for cat_title, start_idx, end_idx, cat_color in categories:
        cat_x1 = margin_x + start_idx * (card_w + gap_x)
        cat_x2 = margin_x + end_idx * (card_w + gap_x) + card_w
        cat_mid = (cat_x1 + cat_x2) / 2
        bar_y = y_card + card_h + 0.35
        
        # Draw category indicator pill
        pill_w = cat_x2 - cat_x1
        pill = FancyBboxPatch((cat_x1, bar_y), pill_w, 0.36,
                              boxstyle="round,pad=0.02,rounding_size=0.08",
                              facecolor=cat_color, edgecolor='none', alpha=0.9, zorder=5)
        ax.add_patch(pill)
        ax.text(cat_mid, bar_y + 0.18, cat_title,
                fontsize=8.5, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=6)

    for i, stage in enumerate(stages):
        x = margin_x + i * (card_w + gap_x)
        color = hex_palette[i]
        
        # Shadow
        shadow = FancyBboxPatch((x + 0.04, y_card - 0.04), card_w, card_h,
                                boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor='#CBD5E1', edgecolor='none', alpha=0.4, zorder=1)
        ax.add_patch(shadow)

        # Card body
        card = FancyBboxPatch((x, y_card), card_w, card_h,
                              boxstyle="round,pad=0.04,rounding_size=0.15",
                              facecolor='#FFFFFF', edgecolor=color["border"],
                              linewidth=2.2, zorder=2)
        ax.add_patch(card)

        # Stage Header background
        header_h = 2.1
        header_y = y_card + card_h - header_h
        header = FancyBboxPatch((x, header_y), card_w, header_h,
                                boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=color["bg"], edgecolor='none', zorder=3)
        ax.add_patch(header)

        # Divider line
        ax.plot([x, x + card_w], [header_y, header_y], color=color["border"], linewidth=1.2, alpha=0.4, zorder=4)

        # Step Number Badge
        badge_w = 0.72
        badge_h = 0.36
        badge_x = x + (card_w - badge_w) / 2
        badge_y = y_card + card_h - 0.52
        badge = FancyBboxPatch((badge_x, badge_y), badge_w, badge_h,
                               boxstyle="round,pad=0.02,rounding_size=0.08",
                               facecolor=color["badge"], edgecolor='none', zorder=5)
        ax.add_patch(badge)
        ax.text(badge_x + badge_w/2, badge_y + badge_h/2, f"STAGE {stage['num']}",
                fontsize=7.8, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=6)

        # Stage Title & Subtitle
        ax.text(x + card_w/2, y_card + card_h - 0.98, stage["title"],
                fontsize=10.5, fontweight='bold', color='#0F172A', ha='center', va='center', zorder=6,
                multialignment='center')
        ax.text(x + card_w/2, y_card + card_h - 1.52, stage["sub"],
                fontsize=8.5, fontweight='semibold', color=color["text"], ha='center', va='center', zorder=6,
                multialignment='center')

        # Annotation Title & Text (Bottom section)
        ann_box_y = y_card + 0.15
        ann_box_h = 1.75
        
        ax.text(x + card_w/2, ann_box_y + ann_box_h - 0.25, "ALGORITHM / TECH",
                fontsize=7.2, fontweight='bold', color='#64748B', ha='center', va='center', zorder=6)
        
        ax.text(x + card_w/2, ann_box_y + 0.7, stage["tech"],
                fontsize=8.0, color='#334155', ha='center', va='center', multialignment='center',
                linespacing=1.3, zorder=6)

        # Arrow to next stage
        if i < n_stages - 1:
            arrow_start_x = x + card_w + 0.05
            arrow_end_x = arrow_start_x + gap_x - 0.1
            arrow_y = y_card + card_h / 2
            ax.annotate('', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
                        arrowprops=dict(arrowstyle="-|>", color=color["border"], lw=2.4, mutation_scale=15),
                        zorder=10)

    # Footer Gradient Legend
    legend_y = 0.9
    ax.text(fig_width / 2, legend_y + 0.35, "PIPELINE GRADIENT STAGES (BLUE → GREEN)", 
            fontsize=9.5, fontweight='bold', color='#475569', ha='center', va='center', zorder=5)

    grad_w = 12.0
    grad_h = 0.16
    grad_x = (fig_width - grad_w) / 2
    
    # Create smooth gradient bar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    # cmap from blue to emerald
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("blue_to_green", ["#1D4ED8", "#0284C7", "#0D9488", "#10B981", "#15803D"])
    ax.imshow(gradient, extent=[grad_x, grad_x + grad_w, legend_y - 0.05, legend_y + 0.11], aspect='auto', cmap=custom_cmap, zorder=4)

    # Labels at ends of gradient bar
    ax.text(grad_x - 0.2, legend_y + 0.03, "Client Ingestion (Blue)", fontsize=8.2, fontweight='bold', color='#1D4ED8', ha='right', va='center')
    ax.text(grad_x + grad_w + 0.2, legend_y + 0.03, "Identification Match (Green)", fontsize=8.2, fontweight='bold', color='#15803D', ha='left', va='center')

    # Save to local workspace and artifact directory
    out_file = r"t:\FYP\Codes\Backend\py-microservice\figure_4_11_facial_recognition_pipeline.png"
    artifact_file = r"C:\Users\DICKSON\.gemini\antigravity-ide\brain\d930ea30-e4c7-4e00-a0da-564ac420fc26\figure_4_11_facial_recognition_pipeline.png"
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    shutil.copyfile(out_file, artifact_file)
    print(f"Generated single-row pipeline at {out_file} and {artifact_file}")

if __name__ == "__main__":
    generate_horizontal_pipeline()
