string class_id      # "0", "person", etc.
float64 score        # Probabilidad (0.0 a 1.0)

# Coordenadas NORMALIZADAS (0.0 a 1.0)
# Es lo estándar en YOLO y facilita la vida al siguiente nodo
float64 x            # Centro X
float64 y            # Centro Y
float64 w            # Ancho
float64 h            # Alto

# Coordenades normalitzades del Punt de corrosió
float64 corr_x
float64 corr_y
float64 std_x
float64 std_y
float64 area
int16 width
int16 height
