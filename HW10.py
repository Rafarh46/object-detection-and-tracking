import cv2
import numpy as np
import argparse

def cli_interface() -> argparse.Namespace:
    """
    Función que maneja la interfaz de línea de comandos (CLI) para la entrada de usuario.

    Returns:
        argparse.Namespace: Los argumentos parseados desde la línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Detección y seguimiento de objetos en un video.")
    parser.add_argument('video', type=str, help='Ruta al archivo de video')
    args = parser.parse_args()
    return args

def detect_and_track_object(video_path):
    """
    Función principal que detecta y realiza el seguimiento de un objeto en un video.

    Args:
        video_path (str): Ruta al archivo de video.

    Returns:
        None
    """
    # Abrir el video para captura de fotogramas
    cap = cv2.VideoCapture(video_path)
    
    # Inicializar el detector de características FAST
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(20)  # Establecer el umbral para la detección de características
    
    # Leer el primer fotograma del video
    ret, old_frame = cap.read()
    if not ret:
        print("Error: no se pudo leer el primer fotograma del video.")
        return
    
    # Convertir el primer fotograma a escala de grises y detectar características
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    kp = fast.detect(old_gray, None)

    # Inicializar variables para el seguimiento del objeto
    last_box = None
    last_points = None
    count_left = 0
    count_right = 0
    last_position = None

    # Loop para procesar cada fotograma del video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir el fotograma actual a escala de grises y detectar características
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_next = fast.detect(frame_gray, None)
        
        # Calcular el flujo óptico entre el fotograma actual y el anterior
        p0 = np.array([kp[idx].pt for idx in range(len(kp))], dtype=np.float32).reshape(-1, 1, 2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        
        # Filtrar las características que se han movido significativamente
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        good_new = good_new.reshape(-1, 1, 2)
        good_old = good_old.reshape(-1, 1, 2)
        diff = np.linalg.norm(good_new - good_old, axis=2).reshape(-1)
        threshold = 2
        good_new = good_new[diff > threshold]
        good_old = good_old[diff > threshold]

        # Inicializar una máscara para dibujar
        mask = np.zeros_like(frame)

        # Actualizar el cuadro del objeto detectado
        if len(good_new) > 0:
            x, y, w, h = cv2.boundingRect(good_new.astype(int))
            last_box = (x, y, w, h)
            last_points = good_new
        
        # Dibujar el rectángulo del objeto detectado
        if last_box is not None:
            x, y, w, h = last_box
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Dibujar una línea en el centro del fotograma
            mid_line = frame.shape[1] // 2
            frame = cv2.line(frame, (mid_line, 0), (mid_line, frame.shape[0]), (255, 0, 0), 2)

            # Determinar la posición del objeto y contar su movimiento
            object_mid = x + w // 2
            if last_position is not None:
                if last_position < mid_line and object_mid > mid_line:
                    count_right += 1
                elif last_position > mid_line and object_mid < mid_line:
                    count_left += 1
            last_position = object_mid

            # Mostrar el conteo de movimientos en el fotograma
            cv2.putText(frame, f'Count Left: {count_left}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Count Right: {count_right}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Dibujar puntos verdes en las características detectadas
        if last_points is not None:
            for pt in last_points:
                a, b = pt.ravel().astype(int)
                frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

        # Fusionar el fotograma con la máscara
        img = cv2.add(frame, mask)
        
        # Mostrar el fotograma con la detección y seguimiento del objeto
        cv2.imshow('Frame', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        # Actualizar el fotograma anterior y las características para el siguiente fotograma
        old_gray = frame_gray.copy()
        kp = kp_next
    
    # Liberar los recursos y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Obtener la ruta del video desde la línea de comandos
    args = cli_interface()
    # Llamar a la función principal para detectar y realizar el seguimiento del objeto en el video
    detect_and_track_object(args.video)
