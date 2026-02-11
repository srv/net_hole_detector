string class_id      # "0", "person", etc.
float64 score        # Probabilidad (0.0 a 1.0)

# Coordenadas NORMALIZADAS (0.0 a 1.0)
# Es lo estándar en YOLO y facilita la vida al siguiente nodo
float64 x            # Centro X
float64 y            # Centro Y
float64 w            # Ancho
float64 h            # Alto

# --- NUEVO (Info de FastSAM) ---
float64 mask_area_px   # Área real en píxeles
float64 mask_x_norm    # Centroide real X (Normalizado 0-1)
float64 mask_y_norm    # Centroide real Y (Normalizado 0-1)
bool has_mask          # Para saber si FastSAM funcionó