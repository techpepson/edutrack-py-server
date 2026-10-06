import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np

# Set figure size and DPI for publication quality
fig_width = 24
fig_height = 8.5
fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('#F8FAFC')

# Remove axes
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
ax.axis('off')

# Title & Header
plt.text(fig_width / 2, 7.8, "FIGURE 4.11: Facial Recognition Pipeline Flow Diagram", 
         fontsize=18, fontweight='bold', ha='center', va='center', color='#0F172A',
         family='sans-serif')
plt.text(fig_width / 2, 7.35, "Sequential stages of the InsightFace recognition process and Qdrant vector retrieval", 
         fontsize=11, ha='center', va='center', color='#64748B', style='italic',
         family='sans-serif')

# Stages definition
stages = [
    {
        "num": "01",
        "title": "Input Image",
        "sub": "Base64 string payload",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "HTTP Request / JSON\nBase64 encoded string",
        "category": "INPUT"
    },
    {
        "num": "02",
        "title": "Image Decoding",
        "sub": "Decode to NumPy Array",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "OpenCV cv2.imdecode\nBGR to RGB conversion",
        "category": "PREPROCESSING"
    },
    {
        "num": "03",
        "title": "Face Detection",
        "sub": "Bounding Box + Landmarks",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "InsightFace RetinaFace\nQuality gate (score ≥ 0.6)",
        "category": "DETECTION"
    },
    {
        "num": "04",
        "title": "Face Alignment",
        "sub": "Landmark Normalisation",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "5-point similarity transform\nAffine pose canonicalisation",
        "category": "PREPROCESSING"
    },
    {
        "num": "05",
        "title": "Feature Extraction",
        "sub": "ArcFace Representation",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "ResNet-50 / ArcFace head\nDeep CNN embedding",
        "category": "EXTRACTION"
    },
    {
        "num": "06",
        "title": "Embedding Vector",
        "sub": "512-d Identity Vector",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "Dense floating-point array\n512 latent dimensions",
        "category": "REPRESENTATION"
    },
    {
        "num": "07",
        "title": "L2 Normalisation",
        "sub": "Unit Sphere Projection",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "Unit norm v / ||v||₂\nHypersphere mapping",
        "category": "NORMALISATION"
    },
    {
        "num": "08",
        "title": "Qdrant Search",
        "sub": "Cosine Vector Retrieval",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "HNSW Cosine index (limit=1)\nCourse ID payload filter",
        "category": "SEARCH"
    },
    {
        "num": "09",
        "title": "Threshold Decision",
        "sub": "Score Verification",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "Decision rule (score ≥ 0.65)\nCosine distance check",
        "category": "VERIFICATION"
    },
    {
        "num": "10",
        "title": "Recognition Result",
        "sub": "Match / No Match Output",
        "tech_title": "TECHNOLOGY / ALGORITHM",
        "tech": "Matched Student ID + Score\nor No-Match rejection",
        "category": "OUTPUT"
    }
]

# We will arrange 10 stages in a sleek 2-row layout of 5 stages each with a smooth snake connection, OR 1 row with clean compact cards.
# Let's test a 2-row layout of 5 cards each:
# Row 1: Stages 01 -> 02 -> 03 -> 04 -> 05 (Left to Right)
# Row 2: Stages 06 -> 07 -> 08 -> 09 -> 10 (Left to Right, connected from Row 1)
# This gives HUGE clarity, ultra readable fonts, and premium spacing!

n_cols = 5
card_w = 3.9
card_h = 2.4
gap_x = 0.75
margin_x = (fig_width - (n_cols * card_w + (n_cols - 1) * gap_x)) / 2

# Blue-to-green gradient color palette across 10 steps
# Hex colors from deep royal blue -> cyan -> teal -> emerald green -> bright green
palette = [
    {"bg": "#EFF6FF", "border": "#2563EB", "badge_bg": "#1D4ED8", "accent": "#3B82F6", "text_dark": "#1E3A8A"}, # 1: Deep Blue
    {"bg": "#F0F9FF", "border": "#0284C7", "badge_bg": "#0369A1", "accent": "#0EA5E9", "text_dark": "#0C4A6E"}, # 2: Sky Blue
    {"bg": "#ECFEFF", "border": "#0891B2", "badge_bg": "#0E7490", "accent": "#06B6D4", "text_dark": "#164E63"}, # 3: Cyan
    {"bg": "#F0FDFA", "border": "#0D9488", "badge_bg": "#0F766E", "accent": "#14B8A6", "text_dark": "#134E4A"}, # 4: Dark Teal
    {"bg": "#F0FDF4", "border": "#059669", "badge_bg": "#047857", "accent": "#10B981", "text_dark": "#064E3B"}, # 5: Emerald / Teal
    {"bg": "#ECFDF5", "border": "#10B981", "badge_bg": "#059669", "accent": "#34D399", "text_dark": "#064E3B"}, # 6: Light Emerald
    {"bg": "#F0FDF4", "border": "#16A34A", "badge_bg": "#15803D", "accent": "#22C55E", "text_dark": "#14532D"}, # 7: Green
    {"bg": "#F7FEE7", "border": "#65A30D", "badge_bg": "#4D7C0F", "accent": "#84CC16", "text_dark": "#365314"}, # 8: Lime Green / Spring
    {"bg": "#ECFDF5", "border": "#059669", "badge_bg": "#047857", "accent": "#10B981", "text_dark": "#064E3B"}, # 9: Emerald
    {"bg": "#F0FDF4", "border": "#16A34A", "badge_bg": "#15803D", "accent": "#22C55E", "text_dark": "#14532D"}  # 10: Bright Green
]

row_y = [4.4, 1.4]  # Y positions (bottom of cards)

for idx, stage in enumerate(stages):
    row = 0 if idx < 5 else 1
    col = idx % 5
    
    x = margin_x + col * (card_w + gap_x)
    y = row_y[row]
    
    color_info = palette[idx]
    
    # Draw Stage Card Box (Main upper card)
    box_main_h = 1.3
    box_tech_h = 0.95
    
    # Outer container shadow / glow
    shadow_box = FancyBboxPatch((x + 0.04, y - 0.04), card_w, card_h,
                                boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor='#CBD5E1', edgecolor='none', alpha=0.35, zorder=1)
    ax.add_patch(shadow_box)
    
    # Outer container
    outer_box = FancyBboxPatch((x, y), card_w, card_h,
                               boxstyle="round,pad=0.04,rounding_size=0.15",
                               facecolor='#FFFFFF', edgecolor=color_info["border"], 
                               linewidth=2.0, zorder=2)
    ax.add_patch(outer_box)
    
    # Upper Stage Header Box
    header_box = FancyBboxPatch((x, y + card_h - box_main_h), card_w, box_main_h,
                                boxstyle="round,pad=0.04,rounding_size=0.15",
                                facecolor=color_info["bg"], edgecolor='none', zorder=3)
    ax.add_patch(header_box)
    
    # Divider line
    ax.plot([x, x + card_w], [y + card_h - box_main_h, y + card_h - box_main_h], 
            color=color_info["border"], linewidth=1.2, alpha=0.4, zorder=4)
    
    # Step Number Badge
    badge_w = 0.6
    badge_h = 0.32
    badge_x = x + 0.2
    badge_y = y + card_h - 0.42
    badge = FancyBboxPatch((badge_x, badge_y), badge_w, badge_h,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           facecolor=color_info["badge_bg"], edgecolor='none', zorder=5)
    ax.add_patch(badge)
    ax.text(badge_x + badge_w/2, badge_y + badge_h/2, stage["num"],
            fontsize=8.5, fontweight='bold', color='#FFFFFF', ha='center', va='center', zorder=6)
    
    # Category Tag
    ax.text(x + 0.9, badge_y + badge_h/2, stage["category"],
            fontsize=7.5, fontweight='bold', color=color_info["badge_bg"], ha='left', va='center', zorder=6)
    
    # Stage Title & Subtitle
    ax.text(x + card_w/2, y + card_h - 0.72, stage["title"],
            fontsize=11.5, fontweight='bold', color='#0F172A', ha='center', va='center', zorder=6)
    ax.text(x + card_w/2, y + card_h - 1.02, stage["sub"],
            fontsize=8.8, fontweight='medium', color=color_info["text_dark"], ha='center', va='center', zorder=6)
    
    # Technology / Algorithm Annotation Box (Bottom section)
    ax.text(x + card_w/2, y + 0.75, stage["tech_title"],
            fontsize=6.8, fontweight='bold', color='#94A3B8', ha='center', va='center', zorder=6)
    
    ax.text(x + card_w/2, y + 0.38, stage["tech"],
            fontsize=8.2, color='#334155', ha='center', va='center', multialignment='center',
            linespacing=1.25, zorder=6)

    # Connecting Arrows
    # Row 1 intra-row arrows: 0 -> 1 -> 2 -> 3 -> 4
    if row == 0 and col < 4:
        arrow_start_x = x + card_w + 0.08
        arrow_end_x = arrow_start_x + gap_x - 0.16
        arrow_y = y + card_h / 2
        ax.annotate('', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color=color_info["border"], lw=2.5, mutation_scale=16),
                    zorder=10)
    
    # Row 2 intra-row arrows: 5 -> 6 -> 7 -> 8 -> 9
    if row == 1 and col < 4:
        arrow_start_x = x + card_w + 0.08
        arrow_end_x = arrow_start_x + gap_x - 0.16
        arrow_y = y + card_h / 2
        ax.annotate('', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color=color_info["border"], lw=2.5, mutation_scale=16),
                    zorder=10)

# Connect Stage 5 (Row 1 End) to Stage 6 (Row 2 Start) via a curved return path or smooth link
s5_x = margin_x + 4 * (card_w + gap_x) + card_w / 2
s5_y = row_y[0]  # bottom of card 5
s6_x = margin_x + card_w / 2
s6_y = row_y[1] + card_h # top of card 6

# Return connection arrow from Stage 5 right to Stage 6 left
r_start_x = margin_x + 4 * (card_w + gap_x) + card_w
r_start_y = row_y[0] + card_h / 2
r_end_x = margin_x
r_end_y = row_y[1] + card_h / 2

# Draw a beautiful S-curve connector on the right margin or a clean connector pipe
conn_right_x = margin_x + 4 * (card_w + gap_x) + card_w + 0.42
mid_y = (row_y[0] + row_y[1] + card_h) / 2

# Path from Stage 5 right -> out -> down to mid_y -> across left -> down to row 2 -> into Stage 6 left
# Or connector right from Stage 5 around to Stage 6:
p1 = patches.ConnectionPatch(
    xyA=(r_end_x, r_end_y), coordsA='data',
    xyB=(r_start_x, r_start_y), coordsB='data',
    arrowstyle="-|>",
    connectionstyle="arc3,rad=-0.0",
    color="#059669", lw=2.5, mutation_scale=16, zorder=10
)
# Better: Draw explicit connecting spline / line
ax.annotate('', xy=(margin_x - 0.05, r_end_y), xytext=(margin_x - 0.45, r_end_y),
            arrowprops=dict(arrowstyle="-|>", color="#10B981", lw=2.5, mutation_scale=16), zorder=10)

# Pipe lines:
ax.plot([r_start_x, r_start_x + 0.4, r_start_x + 0.4], [r_start_y, r_start_y, mid_y], color="#059669", lw=2.2, linestyle="-", zorder=9)
ax.plot([r_start_x + 0.4, margin_x - 0.45], [mid_y, mid_y], color="#0D9488", lw=2.2, linestyle="--", zorder=9)
ax.plot([margin_x - 0.45, margin_x - 0.45], [mid_y, r_end_y], color="#10B981", lw=2.2, linestyle="-", zorder=9)

# Label on the connector pipe
ax.text((r_start_x + margin_x) / 2, mid_y + 0.15, "512-d Feature Representation Passing to Embedding Normalisation", 
        fontsize=8.5, fontweight='bold', color='#0F766E', ha='center', va='bottom',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#F0FDFA', edgecolor='#0D9488', lw=1.2), zorder=11)

# Phase bracket / legend at bottom
legend_y = 0.5
stages_legend = [
    ("Stage 01-02: Ingestion & Decode", "#2563EB"),
    ("Stage 03-04: Detection & Alignment", "#0891B2"),
    ("Stage 05-07: ArcFace & Normalisation", "#059669"),
    ("Stage 08-09: Qdrant Index Search & Scoring", "#16A34A"),
    ("Stage 10: Recognition Match Decision", "#15803D")
]

for i, (leg_text, leg_col) in enumerate(stages_legend):
    leg_x = margin_x + i * 4.3
    ax.add_patch(patches.Circle((leg_x + 0.1, legend_y), 0.08, facecolor=leg_col, edgecolor='none', zorder=5))
    ax.text(leg_x + 0.28, legend_y, leg_text, fontsize=8.2, fontweight='bold', color='#475569', va='center', zorder=5)

output_path = r"t:\FYP\Codes\Backend\py-microservice\figure_4_11_facial_recognition_pipeline.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
print(f"Saved diagram successfully to {output_path}")
